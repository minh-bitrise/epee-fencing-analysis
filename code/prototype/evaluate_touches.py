"""
Epee Fencing Bout Analysis - Touch Detection Evaluation
========================================================
Scores the candidates from detect_touches.py against hand-labelled ground
truth.

Matching is greedy one-to-one within a time tolerance: each labelled touch
can be claimed by at most one candidate and vice versa. Without that
constraint a burst of candidates around a single touch would inflate recall,
which is exactly the failure mode the clustered-candidate case produces.

Alongside precision and recall this reports **corrections required**, the
count of user actions needed to turn the proposals into a correct record:
one rejection per false positive plus one manual entry per missed touch.
That is the human-in-the-loop measure identified by Mosqueira-Rey et al.
(2023) and it is the number that actually matters for this design, since a
detector that halves a user's work is useful even at modest precision.

KNOWN FLAW IN THIS METRIC. It treats a rejection and a manual addition as
equally expensive, and they are not. Rejecting a proposal is one click on an
event the system has already located and timestamped. Adding a missed touch
requires scrubbing the video to find it, which is precisely the work the tool
exists to remove. Recall is therefore worth more than the raw count implies,
and settings should be compared with that asymmetry in mind rather than by
minimising corrections alone. Quantifying the real ratio needs the user study
described in the design chapter, which has not been run; until then this
number is a lower bound on the benefit, not a measure of it.

Usage:
    python3 evaluate_touches.py \
        --candidates results_after/fencing_clip3_distance_touches.csv \
        --truth ground_truth/fencing_clip3_touches.csv
"""

import argparse
import csv

DEFAULT_TOLERANCE_S = 2.0


def load_truth(path, exclude_annulled=False):
    """Read ground-truth touches, skipping comment lines.

    ANNULLED TOUCHES ARE COUNTED BY DEFAULT, AND THAT IS DELIBERATE. An annulled
    hit is one the apparatus registered and the referee then disallowed, for a
    corps-a-corps or a covered target or some other non-valid action. The
    physical event happened: the fencers closed, contact was made and they
    separated, which is exactly the geometry this detector looks for. Scoring it
    as a false positive would penalise the detector for finding something that
    was really there.

    The scoring evaluations take the opposite view and exclude them, because a
    touch that awarded no point cannot appear in a scoreline. Both are right for
    their own question, and the disagreement was accidental until clip 7a became
    the first clip in the set to contain an annulled touch: this function read
    the column and then ignored it.

    Pass exclude_annulled to score detection against awarded touches only, which
    is the stricter reading.
    """
    with open(path) as f:
        rows = [r for r in csv.DictReader(
            line for line in f if not line.startswith("#"))]
    out = [{"time_s": float(r["time_s"]),
            "scorer": r["scorer"],
            "annulled": (r.get("annulled") or "0").strip() == "1"} for r in rows]
    return [t for t in out if not t["annulled"]] if exclude_annulled else out


def load_candidates(path, min_confidence=0.0):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = [{"time_s": float(r["time_s"]),
            "confidence": float(r["confidence"]),
            "signals": r["signals"]} for r in rows]
    return [r for r in out if r["confidence"] >= min_confidence]


def match(truth, cands, tolerance=DEFAULT_TOLERANCE_S):
    """
    Greedy one-to-one matching, closest pairs first.
    Returns (pairs, unmatched_truth, unmatched_cands).
    """
    pairs = []
    remaining_t = list(range(len(truth)))
    remaining_c = list(range(len(cands)))

    # all feasible pairs, nearest first
    options = []
    for i in remaining_t:
        for j in remaining_c:
            gap = abs(truth[i]["time_s"] - cands[j]["time_s"])
            if gap <= tolerance:
                options.append((gap, i, j))
    options.sort()

    used_t, used_c = set(), set()
    for gap, i, j in options:
        if i in used_t or j in used_c:
            continue
        used_t.add(i)
        used_c.add(j)
        pairs.append((i, j, gap))

    miss = [i for i in remaining_t if i not in used_t]
    fp = [j for j in remaining_c if j not in used_c]
    return pairs, miss, fp


def score(truth, cands, tolerance=DEFAULT_TOLERANCE_S):
    pairs, miss, fp = match(truth, cands, tolerance)
    tp, n_fp, n_fn = len(pairs), len(fp), len(miss)
    precision = tp / (tp + n_fp) if (tp + n_fp) else 0.0
    recall = tp / (tp + n_fn) if (tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": tp, "fp": n_fp, "fn": n_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "corrections": n_fp + n_fn,
        "pairs": pairs, "missed": miss, "false_positives": fp,
    }


def report(truth, cand_path, tolerance=DEFAULT_TOLERANCE_S):
    all_cands = load_candidates(cand_path)
    print(f"ground truth : {len(truth)} awarded touches")
    print(f"candidates   : {len(all_cands)}")
    print(f"tolerance    : +/- {tolerance:.1f} s\n")

    print("Confidence sweep (corrections = rejections + manual additions):")
    print(f"  {'min conf':>9}{'cands':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'prec':>7}{'recall':>8}{'F1':>7}{'corrections':>13}")
    best = None
    for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        c = load_candidates(cand_path, min_confidence=thr)
        s = score(truth, c, tolerance)
        print(f"  {thr:>9.2f}{len(c):>7}{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}"
              f"{s['precision']:>7.2f}{s['recall']:>8.2f}{s['f1']:>7.2f}"
              f"{s['corrections']:>13}")
        if best is None or s["f1"] > best[1]["f1"]:
            best = (thr, s, c)

    thr, s, c = best
    print(f"\nBest F1 at min confidence {thr:.2f}: "
          f"precision {s['precision']:.2f}, recall {s['recall']:.2f}, "
          f"F1 {s['f1']:.2f}")
    print(f"Manual baseline would be {len(truth)} entries; "
          f"assisted requires {s['corrections']} corrections "
          f"({100 * s['corrections'] / len(truth):.0f}% of manual effort).")

    print("\nMatched touches:")
    for i, j, gap in sorted(s["pairs"], key=lambda p: truth[p[0]]["time_s"]):
        t = truth[i]
        print(f"  {t['time_s']:>6.0f}s {t['scorer']:<7} <- cand "
              f"{c[j]['time_s']:>6.1f}s (gap {gap:.1f}s, "
              f"conf {c[j]['confidence']:.2f})  {c[j]['signals']}")

    if s["missed"]:
        print("\nMissed touches:")
        for i in sorted(s["missed"], key=lambda i: truth[i]["time_s"]):
            t = truth[i]
            print(f"  {t['time_s']:>6.0f}s {t['scorer']:<7} MISSED")

    if s["false_positives"]:
        print("\nFalse positives:")
        for j in sorted(s["false_positives"], key=lambda j: c[j]["time_s"]):
            print(f"  {c[j]['time_s']:>6.1f}s conf {c[j]['confidence']:.2f}"
                  f"  {c[j]['signals']}")
    return s


def main():
    p = argparse.ArgumentParser(description="Evaluate touch candidates against ground truth")
    p.add_argument("--candidates", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    p.add_argument("--exclude-annulled", action="store_true",
                   help="score against awarded touches only. By default an "
                        "annulled hit counts, because the physical event the "
                        "detector looks for did happen; the scoring "
                        "evaluations take the opposite view for their own "
                        "reasons. See load_truth.")
    args = p.parse_args()
    report(load_truth(args.truth, args.exclude_annulled),
           args.candidates, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
