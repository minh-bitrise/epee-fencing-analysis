"""
Epee Fencing Bout Analysis - What Pose Estimation Contributes to Distance
=========================================================================
Compare the pose front-foot distance against the bounding-box distance ON THE
SAME FRAMES, and measure whether the difference reaches the decision that
distance exists to serve.

WHY THIS EXISTS. The report's only figure for the pose model is the proportion
of frames it succeeded on, and that figure is governed mostly by the deliberate
frame stride: pose runs every third frame, so a "success rate" near a third is
a configuration setting reported as a result. It says nothing about the model.

Pose and the bounding box estimate the SAME quantity, the distance between two
fencers, from different landmarks. On any frame where pose succeeded both are
available, so they can be compared directly. That is weaker than ground truth,
which would need a measured distance per frame and does not exist here, and it
is stronger than a coverage count because it asks what the model changes.

WHAT IS MEASURED, AND WHY IN THIS ORDER.

1. AGREEMENT. The signed difference between the two estimates, per frame,
   pooled per clip. If the two agree to within the system's own resolution then
   pose is refining a number the bounding box already determines, and the
   honest conclusion is that pose does not earn its place through distance.

2. NOISE. The median absolute change between consecutive pose frames. The real
   movement between two frames is identical for both estimators because it is
   the same pair of fencers over the same interval, so any excess is the
   estimator's own jitter. A refinement that is noisier than what it refines is
   a cost, not a contribution.

3. DISCRIMINATION. Distance exists in this system to support one decision, that
   a touch happened. Each clip is cut into non-overlapping windows, a window is
   positive if a hand-labelled touch falls inside it, and each estimator scores
   the window by its minimum distance. The rank statistic (AUC) then says how
   well each estimator separates the windows a touch occurred in from the rest,
   on the same windows and the same frames. This is the only one of the three
   that uses real labels, and it is the one that matters: agreement and noise
   describe the signal, discrimination asks whether the difference reaches the
   decision.

WHY PAIRED, AND WHY THE INTERVAL IS CLUSTERED BY CLIP. Both estimators see the
same windows, so the difference between their AUCs is a paired quantity and its
interval must be built by resampling windows, not by treating two independent
intervals as if their overlap settled anything. Windows are resampled WITHIN
each clip and the clips are held fixed, because four clips cannot support a
resample over clips and pretending otherwise would put a confidence interval on
a sample of four.

WHAT THIS CANNOT SAY. Neither estimator is validated against a measured
distance, so a difference here says which estimator serves the downstream
decision better, never which is closer to the truth. Both could be wrong in the
same direction and this would not see it.

Run with:
    python3 evaluate_pose.py --results results_posecmp
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

import clips

# Length of a scoring window, in seconds. Chosen as the approach-and-hit span the
# touch detector itself works over rather than for convenience: shorter and a
# labelled touch time, which is recorded to the nearest second, can fall outside
# the window holding its own approach; longer and a single window swallows both a
# touch and the unrelated play around it.
WINDOW_S = 2.0

# Tolerance on a labelled touch time, in seconds. The ground-truth files record
# touches to the nearest second, so a label can sit up to half a second from the
# moment of contact. A window is positive if the label falls inside it once
# widened by this much at each end.
LABEL_TOL_S = 0.5

BOOTSTRAP_N = 2000
SEED = 11


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def paired_series(rows):
    """Frames where pose succeeded AND both estimates exist.

    Returns (times, pose_m, bbox_m) as arrays. Restricting to pose frames is
    what makes the comparison paired: on a fallback frame the two columns hold
    the same number by construction, so including them would dilute every
    difference towards zero with frames that cannot differ.
    """
    t, p, b = [], [], []
    for r in rows:
        if r.get("method") != "pose":
            continue
        raw, bbox = r.get("distance_raw_m", ""), r.get("distance_bbox_m", "")
        if raw == "" or bbox == "":
            continue
        t.append(float(r["time_s"]))
        p.append(float(raw))
        b.append(float(bbox))
    return np.array(t), np.array(p), np.array(b)


def read_labels(path):
    """Awarded touch times from a ground-truth file, annulled hits excluded."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            if not r.get("time_s"):
                continue
            if str(r.get("annulled", "0")).strip() == "1":
                continue
            out.append(float(r["time_s"]))
    return sorted(out)


def windows(times, pose_m, bbox_m, labels, window_s=WINDOW_S, tol_s=LABEL_TOL_S):
    """Cut the clip into non-overlapping windows and score each one.

    Returns (is_touch, pose_min, bbox_min) arrays over windows that hold at
    least one paired frame. A window with no frames is dropped rather than
    scored as a miss: an interval the tracker lost both fencers over is an
    absence of measurement, and scoring it would credit whichever estimator
    happened to be undefined there.
    """
    if times.size == 0:
        return np.zeros(0, bool), np.zeros(0), np.zeros(0)
    edges = np.arange(0.0, times.max() + window_s, window_s)
    y, pm, bm = [], [], []
    for lo in edges:
        hi = lo + window_s
        m = (times >= lo) & (times < hi)
        if not m.any():
            continue
        y.append(any(lo - tol_s <= s < hi + tol_s for s in labels))
        pm.append(pose_m[m].min())
        bm.append(bbox_m[m].min())
    return np.array(y, bool), np.array(pm), np.array(bm)


def auc(scores, y):
    """Rank AUC for a classifier where a LOWER score means positive.

    Ties are handled by average ranks, which matters because minimum distance
    over a window is rounded to the millimetre and exact ties do occur.
    """
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(scores.size, float)
    ranks[order] = np.arange(1, scores.size + 1)
    # average the ranks within each group of equal scores
    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    # lower score = positive, so rank the NEGATIVES to keep the usual orientation
    r_neg = ranks[~y].sum()
    return float((r_neg - neg * (neg + 1) / 2) / (pos * neg))


def paired_auc_ci(clips, n=BOOTSTRAP_N, seed=SEED):
    """Percentile interval on (pose AUC - bbox AUC), resampling within clips.

    `clips` is a list of (y, pose_min, bbox_min) per clip. Windows are drawn
    with replacement inside each clip and the clips themselves are held fixed,
    so the interval covers sampling of play within these four bouts and makes
    no claim about bouts in general.
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        ys, ps, bs = [], [], []
        for y, pm, bm in clips:
            if y.size == 0:
                continue
            idx = rng.integers(0, y.size, y.size)
            ys.append(y[idx]); ps.append(pm[idx]); bs.append(bm[idx])
        if not ys:
            continue
        y = np.concatenate(ys)
        a_p, a_b = auc(np.concatenate(ps), y), auc(np.concatenate(bs), y)
        if a_p is None or a_b is None:
            continue
        diffs.append(a_p - a_b)
    if len(diffs) < n // 10:
        return None
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def clip_key(csv_path):
    """'fencing_clip3_distance.csv' -> 'fencing_clip3'."""
    return os.path.basename(csv_path).replace("_distance.csv", "")


GT_FOR = clips.CLIPS   # one definition of the evaluation set, see clips.py


def analyse(results_dir, gt_dir="."):
    per_clip, pooled = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_distance.csv"))):
        key = clip_key(path)
        gt = os.path.join(gt_dir, GT_FOR.get(key, ""))
        if not GT_FOR.get(key) or not os.path.exists(gt):
            continue
        t, p, b = paired_series(read_rows(path))
        if t.size == 0:
            per_clip.append({"clip": key, "n_paired": 0})
            continue
        d = p - b
        step = np.abs(np.diff(p)), np.abs(np.diff(b))
        labels = read_labels(gt)
        y, pm, bm = windows(t, p, b, labels)
        per_clip.append({
            "clip": key,
            "n_paired": int(t.size),
            "median_diff_m": float(np.median(d)),
            "iqr_diff_m": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "median_step_pose_m": float(np.median(step[0])) if step[0].size else None,
            "median_step_bbox_m": float(np.median(step[1])) if step[1].size else None,
            "n_windows": int(y.size),
            "n_touch_windows": int(y.sum()),
            "auc_pose": auc(pm, y),
            "auc_bbox": auc(bm, y),
        })
        pooled.append((y, pm, bm))

    out = {"clips": per_clip}
    if pooled:
        y = np.concatenate([c[0] for c in pooled])
        pm = np.concatenate([c[1] for c in pooled])
        bm = np.concatenate([c[2] for c in pooled])
        a_p, a_b = auc(pm, y), auc(bm, y)
        out["pooled"] = {
            "n_windows": int(y.size),
            "n_touch_windows": int(y.sum()),
            "auc_pose": a_p,
            "auc_bbox": a_b,
            "auc_diff": None if a_p is None or a_b is None else a_p - a_b,
            "auc_diff_ci": paired_auc_ci(pooled),
        }
    return out


def _f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def report(res):
    print("Pose against the bounding box, on the frames where both exist")
    print("=" * 62)
    print(f"{'clip':<16}{'frames':>8}{'med diff':>10}{'IQR':>8}"
          f"{'step P':>9}{'step B':>9}{'AUC P':>8}{'AUC B':>8}")
    for c in res["clips"]:
        if not c.get("n_paired"):
            print(f"{c['clip']:<16}{'none':>8}   no paired frames")
            continue
        print(f"{c['clip']:<16}{c['n_paired']:>8}{_f(c['median_diff_m']):>10}"
              f"{_f(c['iqr_diff_m']):>8}{_f(c['median_step_pose_m']):>9}"
              f"{_f(c['median_step_bbox_m']):>9}{_f(c['auc_pose']):>8}"
              f"{_f(c['auc_bbox']):>8}")

    p = res.get("pooled")
    if not p:
        return
    print()
    print(f"Pooled over {len(res['clips'])} clips: {p['n_windows']} windows, "
          f"{p['n_touch_windows']} holding a labelled touch")
    print(f"  AUC, pose front foot:   {_f(p['auc_pose'])}")
    print(f"  AUC, bounding box:      {_f(p['auc_bbox'])}")
    if p["auc_diff"] is not None:
        ci = p["auc_diff_ci"]
        ci_s = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        print(f"  Difference:             {p['auc_diff']:+.3f}  95% CI {ci_s}")
        if ci is not None and ci[0] <= 0 <= ci[1]:
            print("  The interval spans zero: on this footage pose does not measurably")
            print("  improve the decision distance exists to serve.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default="results_posecmp",
                    help="directory holding *_distance.csv with a distance_bbox_m column")
    ap.add_argument("--gt-dir", default=".", help="directory holding ground_truth/")
    args = ap.parse_args(argv)
    res = analyse(args.results, args.gt_dir)
    if not res["clips"]:
        print(f"No usable CSVs in {args.results}. Re-run run_detection.py: the "
              f"distance_bbox_m column is required and older runs do not have it.")
        return 1
    report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
