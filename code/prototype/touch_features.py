"""
Epee Fencing Bout Analysis - Labelled Windows for a Learned Touch Proposer
==========================================================================
Turn a bout's per-frame CSV and its hand-labelled touches into a feature matrix,
so the accept-or-reject decision currently made by two hand-set thresholds can
be made by a trained classifier instead and the two compared on equal terms.

WHY THIS EXISTS. `detect_touches.py` proposes a touch where inter-fencer
distance has a local minimum of at least 0.4 m prominence AND the fencers then
separate by at least 0.8 m. Both numbers were set by hand against clip 3. The
rule works, and the project has no way of saying whether it works because the
sport has that structure or because those two constants happen to suit four
recordings. A learned proposer answers that by competing with it.

WHAT IS LEARNED AND WHAT IS NOT. Candidate GENERATION stays a domain rule: this
module emits every local minimum in the distance series. What the classifier
replaces is the decision the rule makes about each candidate, accept or reject,
and the ranking it puts them in. That is the honest boundary and it should be
stated in the report rather than implied: this is not touch detection learned
from video, it is the proposal decision learned from geometry.

WHY THE CANDIDATE PROMINENCE IS LOWER THAN THE DETECTOR'S. Generating candidates
at the detector's own 0.4 m would hand the classifier exactly the set the rule
already accepts, so no model could ever recover a touch the rule missed and the
comparison would be rigged in the rule's favour. Generating at 0.1 m makes the
candidate set a SUPERSET of the rule's, which costs a worse class balance and
buys a fair contest. The price is measured rather than assumed: `ceiling()`
reports the recall available to any classifier built on this candidate set, and
no downstream result can be read without it.

WHY FEATURES COME IN RAW AND CLIP-NORMALISED PAIRS. The metre scale is derived
per clip from fencer height, and the four clips are framed differently, so 0.7 m
does not mean the same thing in each. A model trained on three clips and tested
on a fourth is therefore exposed to exactly that shift. Each distance feature is
emitted both as measured and as its percentile within its own bout, and the
ablation is left to say which the model actually uses. Deciding that in advance
would be a guess presented as a design.

Run with:
    python3 touch_features.py --results results_current
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

import detect_touches as dt

# Prominence for candidate GENERATION. Deliberately well below the detector's
# 0.4 m so the candidate set contains minima the rule rejects. See the module
# docstring: this is what makes the contest fair, and it is why `ceiling()`
# exists to report what it costs.
CANDIDATE_PROMINENCE_M = 0.1

# Seconds either side of a labelled touch within which a candidate counts as
# that touch. Matches evaluate_touches.DEFAULT_TOLERANCE_S so the learned
# proposer and the rule are scored against the same notion of a hit; the labels
# are recorded to the nearest second, and a touch's minimum can precede the
# referee's halt by a moment.
MATCH_TOLERANCE_S = dt.DEFAULT_TOLERANCE_S if hasattr(dt, "DEFAULT_TOLERANCE_S") else 2.0

# Windows used to describe each candidate, in seconds.
RATE_WIN_S = 0.5          # over which closing and opening speed are measured
DWELL_THRESH_M = 1.2      # "close" for the purpose of measuring time spent close
DWELL_WIN_S = 2.0         # window in which dwell is counted

FEATURE_NAMES = [
    "min_distance_m", "min_distance_pct",
    "prominence_m", "prominence_pct",
    "separation_after_m", "separation_after_pct",
    "separation_before_m",
    "closing_rate_ms", "opening_rate_ms",
    "dwell_close_s",
    "stance_max_m", "hip_height_min_m",
    "piste_position_pct",
    "time_since_prev_s", "bout_fraction",
]


def load_series(csv_path):
    """Time, smoothed distance, and the sparse per-fencer columns, as arrays."""
    rows = list(csv.DictReader(open(csv_path)))

    def col(name):
        out = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            v = r.get(name, "")
            if v not in ("", None):
                out[i] = float(v)
        return out

    t = np.array([float(r["time_s"]) for r in rows])
    return {
        "t": t,
        "d": col("distance_smooth_m"),
        "f1_stance": col("f1_stance_m"), "f2_stance": col("f2_stance_m"),
        "f1_hip": col("f1_hip_height_m"), "f2_hip": col("f2_hip_height_m"),
        "f1_pos": col("f1_pos_m"), "f2_pos": col("f2_pos_m"),
    }


def _win(t, v, lo, hi):
    """Finite samples of v where t is in [lo, hi)."""
    seg = v[(t >= lo) & (t < hi)]
    return seg[np.isfinite(seg)]


def _pct(value, pool):
    """Where `value` sits in `pool`, 0 to 100, or nan.

    This is the per-clip normalisation: a distance is converted into its rank
    within the bout it came from, so the feature means the same thing on a
    tightly framed clip and a wide one.
    """
    pool = pool[np.isfinite(pool)]
    if value is None or not np.isfinite(value) or pool.size == 0:
        return np.nan
    return float(100.0 * (pool < value).mean())


def candidate_times(s, prominence=CANDIDATE_PROMINENCE_M):
    """Every local minimum in the distance series, with its depth."""
    return dt.local_minima(s["t"], s["d"], prominence=prominence)


def prominence_at(t, d, ts, win_s=dt.LOCAL_MIN_WIN_S):
    """How far the minimum at ts sits below the higher of its two shoulders.

    Measured against the maximum on each side rather than the mean, matching
    `local_minima`, so the learned model sees the same quantity the rule uses
    rather than a differently defined lookalike.
    """
    left = _win(t, d, ts - win_s, ts)
    right = _win(t, d, ts, ts + win_s)
    here = _win(t, d, ts - 0.05, ts + 0.05)
    if left.size == 0 or right.size == 0 or here.size == 0:
        return np.nan
    return float(min(left.max(), right.max()) - here.min())


def features_at(s, ts, prev_ts, pools):
    """One candidate's feature vector, as a dict keyed by FEATURE_NAMES."""
    t, d = s["t"], s["d"]

    mind = dt.min_distance_near(t, d, ts, win=1.0)
    prom = prominence_at(t, d, ts)
    sep_after = dt.separation_after(t, d, ts)

    # The mirror of separation_after. A dip inside continuous close play is
    # approached from close range and left at close range; a touch is approached
    # from distance. Without this the model cannot tell the two apart on
    # geometry, and the rule cannot either.
    before_far = _win(t, d, ts - 2.5, ts - 0.8)
    before_near = _win(t, d, ts - 0.5, ts)
    sep_before = (float(before_far.mean() - before_near.mean())
                  if before_far.size >= 5 and before_near.size >= 3 else np.nan)

    pre = _win(t, d, ts - RATE_WIN_S, ts)
    post = _win(t, d, ts, ts + RATE_WIN_S)
    closing = float((pre[0] - pre[-1]) / RATE_WIN_S) if pre.size >= 3 else np.nan
    opening = float((post[-1] - post[0]) / RATE_WIN_S) if post.size >= 3 else np.nan

    near = _win(t, d, ts - DWELL_WIN_S, ts + DWELL_WIN_S)
    dwell = (float((near < DWELL_THRESH_M).mean() * 2 * DWELL_WIN_S)
             if near.size else np.nan)

    # Stance and hip height are sampled only on pose frames, so they are taken
    # as extremes over a window rather than read at the instant: the widest
    # stance and the lowest hip near the candidate are what a lunge looks like.
    st = np.concatenate([_win(t, s["f1_stance"], ts - 0.5, ts + 0.5),
                         _win(t, s["f2_stance"], ts - 0.5, ts + 0.5)])
    hp = np.concatenate([_win(t, s["f1_hip"], ts - 0.5, ts + 0.5),
                         _win(t, s["f2_hip"], ts - 0.5, ts + 0.5)])

    pos = np.concatenate([_win(t, s["f1_pos"], ts - 0.5, ts + 0.5),
                          _win(t, s["f2_pos"], ts - 0.5, ts + 0.5)])
    piste_pct = _pct(float(pos.mean()), pools["pos"]) if pos.size else np.nan

    duration = float(t[-1]) if t.size else 0.0
    return {
        "min_distance_m": np.nan if mind is None else mind,
        "min_distance_pct": _pct(mind, pools["d"]),
        "prominence_m": prom,
        "prominence_pct": _pct(prom, pools["prom"]),
        "separation_after_m": np.nan if sep_after is None else sep_after,
        "separation_after_pct": _pct(sep_after, pools["sep"]),
        "separation_before_m": sep_before,
        "closing_rate_ms": closing,
        "opening_rate_ms": opening,
        "dwell_close_s": dwell,
        "stance_max_m": float(st.max()) if st.size else np.nan,
        "hip_height_min_m": float(hp.min()) if hp.size else np.nan,
        "piste_position_pct": piste_pct,
        "time_since_prev_s": (ts - prev_ts) if prev_ts is not None else np.nan,
        "bout_fraction": (ts / duration) if duration else np.nan,
    }


def load_truth(path):
    """Awarded touch times, annulled hits excluded."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if not r.get("time_s"):
                continue
            if str(r.get("annulled", "0")).strip() == "1":
                continue
            out.append(float(r["time_s"]))
    return sorted(out)


def label(times, truth, tolerance=MATCH_TOLERANCE_S):
    """Label each candidate 1 positive, 0 negative, or -1 ignore.

    WHY THERE IS AN IGNORE CLASS. Labelling every candidate within the match
    tolerance as positive gave 1,326 positives for 27 touches, about 49 per
    touch, because minima are dense at this prominence and the tolerance is two
    seconds either side. That inflates the positive rate roughly fiftyfold and
    makes the task look far easier than it is: a model could score well by
    recognising "near a touch" rather than "is the touch".

    So exactly one candidate per labelled touch is positive, the nearest. The
    others inside the tolerance are neither: calling them negative would train
    the model that a candidate half a second from a real touch is not touch-like,
    which is false and is the one place the geometry genuinely is ambiguous.
    They are dropped from training and kept at inference, where the merge step
    collapses them exactly as it does for the rule.

    This keeps the positive count equal to the touch count, which is the number
    the report should quote, and it leaves `ceiling()` unchanged because the
    nearest candidate to a reachable touch is still labelled.
    """
    y = np.zeros(times.size, dtype=int)
    for g in truth:
        near = np.flatnonzero(np.abs(times - g) <= tolerance)
        if near.size == 0:
            continue
        y[near] = np.where(y[near] == 1, 1, -1)      # ignore, unless already a hit
        y[near[np.argmin(np.abs(times[near] - g))]] = 1
    return y


def build(csv_path, truth_path, prominence=CANDIDATE_PROMINENCE_M,
          tolerance=MATCH_TOLERANCE_S):
    """One clip's candidates as (X, y, times, truth). See `label` for y."""
    s = load_series(csv_path)
    truth = load_truth(truth_path)
    cands = candidate_times(s, prominence)

    # Pools for the per-clip percentile features, built from the candidates
    # themselves rather than from every frame: the comparison a percentile is
    # meant to make is against other candidates in this bout, not against the
    # long stretches of open distance between them.
    t, d = s["t"], s["d"]
    proms = np.array([prominence_at(t, d, ts) for ts, _ in cands])
    seps = np.array([(lambda v: np.nan if v is None else v)(
        dt.separation_after(t, d, ts)) for ts, _ in cands])
    pos_pool = np.concatenate([s["f1_pos"], s["f2_pos"]])
    pools = {"d": np.array([dd for _, dd in cands]), "prom": proms,
             "sep": seps, "pos": pos_pool[np.isfinite(pos_pool)]}

    X, times = [], []
    prev = None
    for ts, _ in cands:
        f = features_at(s, ts, prev, pools)
        X.append([f[k] for k in FEATURE_NAMES])
        times.append(ts)
        prev = ts
    times = np.array(times, dtype=float)
    return (np.array(X, dtype=float), label(times, truth, tolerance),
            times, truth)


def ceiling(times, truth, tolerance=MATCH_TOLERANCE_S):
    """Recall available to ANY classifier over this candidate set.

    Candidate generation is a domain rule, so a touch with no candidate near it
    is unreachable however good the model is. Every result downstream is capped
    by this number and reporting one without it would overstate what was
    measured.
    """
    if not truth:
        return None
    hit = sum(1 for g in truth
              if np.any(np.abs(times - g) <= tolerance)) if times.size else 0
    return hit / len(truth)


CLIPS = {
    "fencing_clip":  "ground_truth/fencing_clip1_touches.csv",
    "fencing_clip2": "ground_truth/fencing_clip2_touches.csv",
    "fencing_clip3": "ground_truth/fencing_clip3_touches.csv",
    "fencing_clip4": "ground_truth/fencing_clip4_touches.csv",
}


def build_all(results_dir, base="."):
    """Every clip with both metrics and labels, keyed by clip name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_distance.csv"))):
        name = os.path.basename(path).replace("_distance.csv", "")
        rel = CLIPS.get(name)
        if not rel:
            continue
        truth_path = os.path.join(base, rel)
        if not os.path.exists(truth_path):
            continue
        X, y, times, truth = build(path, truth_path)
        out[name] = {"X": X, "y": y, "times": times, "truth": truth}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build labelled touch candidates")
    ap.add_argument("--results", default="results_current")
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default=None, help="write the matrix to an .npz")
    args = ap.parse_args(argv)

    data = build_all(args.results, args.base)
    if not data:
        print(f"no clips with both metrics and labels in {args.results}")
        return 1

    print(f"{'clip':<16}{'cands':>7}{'pos':>6}{'ignore':>8}{'touches':>9}"
          f"{'ceiling':>9}{'pos rate':>10}{'missing':>9}")
    tot_c = tot_p = tot_t = tot_i = 0
    for name, dd in data.items():
        X, y, times, truth = dd["X"], dd["y"], dd["times"], dd["truth"]
        c = ceiling(times, truth)
        miss = float(np.isnan(X).mean()) if X.size else 0.0
        pos, ign = int((y == 1).sum()), int((y == -1).sum())
        keep = int((y >= 0).sum())
        print(f"{name:<16}{len(y):>7}{pos:>6}{ign:>8}{len(truth):>9}"
              f"{('n/a' if c is None else f'{c:.2f}'):>9}"
              f"{(pos / keep if keep else 0):>9.2%}{miss:>9.1%}")
        tot_c += len(y); tot_p += pos; tot_t += len(truth); tot_i += ign
    print(f"{'total':<16}{tot_c:>7}{tot_p:>6}{tot_i:>8}{tot_t:>9}")
    print(f"\n{len(FEATURE_NAMES)} features: {', '.join(FEATURE_NAMES)}")

    if args.out:
        np.savez(args.out, **{f"{n}_{k}": v for n, dd in data.items()
                              for k, v in (("X", dd["X"]), ("y", dd["y"]),
                                           ("times", dd["times"]))},
                 feature_names=np.array(FEATURE_NAMES))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
