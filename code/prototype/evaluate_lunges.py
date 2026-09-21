"""
Does a pose-derived stance feature mark a lunge?

An earlier test used awarded touches as a proxy for lunge times, which is weak in
both directions: most lunges miss, and some touches come from infighting. This
evaluates against lunge labels from the review interface instead, which is the
direct test.

Usage:
    python3 evaluate_lunges.py --csv results_pose/fencing_clip3_distance.csv \
                               --bout results_pose:fencing_clip3
"""

import argparse
import csv
import json
import os
from math import comb

import numpy as np

ANNOTATION_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "annotations")

# Windows tried, in seconds before and after the labelled peak. Several rather
# than one, because a result that only appears at a single window width is a
# window-tuning artefact and should not be believed. The label marks the peak, so
# these are tight.
WINDOWS = ((0.10, 0.10), (0.20, 0.20), (0.40, 0.20))

# How often a random window would beat the threshold by chance. The threshold is
# the matching percentile of the random-window distribution, so this is fixed by
# construction rather than estimated.
CHANCE = 0.10


def binomial_tail(k, n, p):
    """P(X >= k): how often chance alone would do this well or better."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def load_labels(bout_id):
    """Lunge labels for one bout, as (time_s, slot) pairs sorted by time."""
    # mirrors AnnotationStore._path; kept in step deliberately rather than
    # importing the backend, so this script has no dependency on FastAPI
    safe = bout_id.replace("/", "_").replace(":", "__")
    path = os.path.join(ANNOTATION_ROOT, f"{safe}.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"No annotation file for bout {bout_id!r} at {path}.\n"
            "Label some lunges in the review interface first: run the backend, "
            "pause on the frame of maximum extension and press 1 or 2.")
    with open(path) as f:
        data = json.load(f)
    lunges = data.get("lunges", [])
    if not lunges:
        raise SystemExit(f"Bout {bout_id!r} has no lunge labels yet.")
    return sorted((float(l["time_s"]), int(l["slot"]), l.get("note", ""))
                  for l in lunges)


def load_series(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["time_s"]) for r in rows])

    def col(name):
        if not rows or name not in rows[0]:
            raise SystemExit(
                f"{csv_path} has no {name!r} column. Re-run run_detection.py; "
                "the stance columns were added in TODO B1h.")
        return np.array([float(r[name]) if r[name] else np.nan for r in rows])

    series = {}
    for slot, key in ((0, "f1"), (1, "f2")):
        stance = col(f"{key}_stance_m")
        hip = col(f"{key}_hip_height_m")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = stance / hip
        ratio[~np.isfinite(ratio)] = np.nan
        # The ratio is dimensionless, so the per-fencer depth error cancels. B1h
        # measured that error at about 40 per cent between the two fencers,
        # because the scale is clip-wide and cannot know how far each fencer is
        # from the camera, so anything quoted in metres is not comparable.
        series[slot] = {"stance_m": stance, "hip_height_m": hip,
                        "stance_over_hip": ratio}
    return t, series


def extreme(series, t, centre, pre, post, mode):
    m = (t >= centre - pre) & (t <= centre + post)
    seg = series[m]
    seg = seg[~np.isnan(seg)]
    if not len(seg):
        return np.nan
    return float(np.nanmax(seg)) if mode == "max" else float(np.nanmin(seg))


def episode_rate(series, threshold, above, duration_s):
    """Separate excursions past the threshold per minute.

    Enrichment alone does not make a detector. A feature that is genuinely
    higher at lunges is still useless if it is also high constantly, and B1h's
    hip-drop signal failed exactly here: real at p = 0.001 and firing nine
    times more often than touches occurred.
    """
    runs, prev = 0, False
    for v in series:
        now = (not np.isnan(v)) and ((v > threshold) if above else (v < threshold))
        if now and not prev:
            runs += 1
        prev = now
    return 60.0 * runs / duration_s if duration_s else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Test pose stance features against hand-labelled lunges")
    ap.add_argument("--csv", required=True, help="Metrics CSV with stance columns")
    ap.add_argument("--bout", required=True,
                    help="Bout id as the interface shows it, e.g. results_pose:fencing_clip3")
    ap.add_argument("--seed", type=int, default=20260818,
                    help="Baseline RNG seed, fixed so the result is reproducible")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="Drop labels the labeller annotated as doubtful. Run BOTH "
                         "ways: if the conclusion depends on three borderline cases "
                         "out of thirty-six, it is not a conclusion.")
    args = ap.parse_args()

    labels = load_labels(args.bout)
    flagged = [l for l in labels if l[2]]
    if args.exclude_flagged:
        labels = [l for l in labels if not l[2]]
        print(f"excluding {len(flagged)} labels the labeller flagged as doubtful")
    elif flagged:
        print(f"including {len(flagged)} labels the labeller flagged as doubtful "
              f"(rerun with --exclude-flagged to see the effect)")
    t, series = load_series(args.csv)
    duration = float(t[-1]) if len(t) else 0.0
    rng = np.random.default_rng(args.seed)

    per_slot = {0: [x for x, sl, _ in labels if sl == 0],
                1: [x for x, sl, _ in labels if sl == 1]}
    print(f"{len(labels)} lunge labels over {duration:.0f}s: "
          f"{len(per_slot[0])} for F1, {len(per_slot[1])} for F2")
    if len(labels) < 15:
        print("NOTE: fewer than 15 labels. Treat every figure below as indicative; "
              "the binomial test has very little power at this size.")

    features = (("stance_m", "max", True),
                ("hip_height_m", "min", False),
                ("stance_over_hip", "max", True))

    for slot in (0, 1):
        times = per_slot[slot]
        if not times:
            print(f"\n=== F{slot+1}: no labels, skipped ===")
            continue
        note = ("  (too few for the binomial test to have any power; "
                "read the effect size, not p)" if len(times) < 12 else "")
        print(f"\n=== F{slot+1}: {len(times)} labelled lunges ==={note}")
        for name, mode, above in features:
            s = series[slot][name]
            avail = 100.0 * np.sum(~np.isnan(s)) / len(s)
            print(f"  {name} (available {avail:.0f}% of frames)")
            for pre, post in WINDOWS:
                base = np.array([v for v in (
                    extreme(s, t, rng.uniform(t[0] + pre, t[-1] - post),
                            pre, post, mode)
                    for _ in range(1500)) if not np.isnan(v)])
                if not len(base):
                    print("    no usable baseline")
                    continue
                thr = np.percentile(base, 90 if above else 10)
                peaks = np.array([v for v in (
                    extreme(s, t, x, pre, post, mode) for x in times)
                    if not np.isnan(v)])
                if not len(peaks):
                    print(f"    -{pre:.2f}/+{post:.2f}s: no pose data in any "
                          f"labelled window")
                    continue
                hits = int((peaks > thr).sum() if above else (peaks < thr).sum())
                p = binomial_tail(hits, len(peaks), CHANCE)
                rate = episode_rate(s, thr, above, duration)
                per_min = 60.0 * len(times) / duration
                print(f"    -{pre:.2f}/+{post:.2f}s: {hits}/{len(peaks)} past p"
                      f"{'90' if above else '10'} "
                      f"({100*hits/len(peaks):3.0f}%, chance {int(CHANCE*100)}%, "
                      f"p={p:.3f}) | fires {rate:.1f}/min vs {per_min:.1f} "
                      f"lunges/min ({rate/per_min:.1f}x)")
        # a label with no pose data nearby cannot support or refute anything, so
        # say how many were dropped rather than letting them shrink n silently
        blind = sum(1 for x in times
                    if np.isnan(extreme(series[slot]["stance_m"], t, x,
                                        0.20, 0.20, "max")))
        print(f"  {blind} of {len(times)} labelled windows had no stance data")

    print("\nReading this: a feature is worth keeping only if it is BOTH enriched "
          "at labels (low p) AND not firing far more often than lunges occur "
          "(multiplier near 1). B1h's hip-drop signal passed the first and failed "
          "the second by nine times, which is why it was rejected.")


if __name__ == "__main__":
    main()
