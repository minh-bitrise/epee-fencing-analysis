"""
Score the candidates from detect_touches.py against hand-labelled ground truth.

Matching is greedy one-to-one within a time tolerance, so a burst of candidates
around one touch cannot inflate recall. Alongside precision and recall it reports
CORRECTIONS REQUIRED: one rejection per false positive plus one manual entry per
miss, which is the number this design turns on.
"""

import argparse
import csv

DEFAULT_TOLERANCE_S = 2.0


def load_truth(path, exclude_annulled=False):
    """Read ground-truth touches, skipping comment lines.

    An annulled hit is one the apparatus registered and the referee then
    disallowed, for a corps-a-corps or a covered target or some other non-valid
    action.
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
