"""
Propose lunges from pose, using a threshold calibrated on the bout being watched.

`stance_m / hip_height_m` separates lunges well on the clip its operating point
came from, and does not survive a change of camera: fitted on clip 3 and applied
to clip 2 it reaches F1 0.22 while firing on a quarter of all windows. So this
stops trying to transfer, and fits the threshold from lunges the user has already
confirmed on the bout in hand.
"""

import argparse
import csv
import json
import sys

import numpy as np

# Half-width of the window a lunge is looked for in. A lunge lasts about ten
# frames at 30 fps, so a half-second window contains one comfortably without
# merging two.
WINDOW_S = 0.25

# The calibration threshold is the level the QUIETEST confirmed lunges still
# clear, not the median. A median would reject the smaller half of real lunges by
# construction; the 20th percentile keeps them while still sitting above the
# ordinary run of play.
CALIBRATION_PERCENTILE = 20

# Below this many confirmed lunges the threshold is not worth fitting. Measured:
# three gives recall 0.48 on clip 3 and five gives 0.80, so the difference
# between a poor calibration and a usable one is two examples.
MIN_CALIBRATION_LUNGES = 5

# Hip height below this is not a measurement, it is a pose failure, and dividing
# by it produces an enormous ratio that would clear any threshold.
MIN_HIP_HEIGHT_M = 0.1


def stance_ratio_series(csv_path, slot):
    """
    The dimensionless stance ratio over time for one fencer.

    Returns (times, ratios). Frames without pose are simply absent rather than
    interpolated: pose runs on a stride, so the gaps are the sampling, and
    inventing values between them would invent postures.
    """
    times, ratios = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            stance = row.get(f"f{slot + 1}_stance_m")
            hip = row.get(f"f{slot + 1}_hip_height_m")
            if not stance or not hip:
                continue
            hip = float(hip)
            if hip <= MIN_HIP_HEIGHT_M:
                continue
            times.append(float(row["time_s"]))
            ratios.append(float(stance) / hip)
    return np.array(times), np.array(ratios)


def peak_in_window(times, ratios, centre, half=WINDOW_S):
    """The largest stance ratio within `half` seconds of `centre`, or None."""
    mask = (times >= centre - half) & (times <= centre + half)
    return float(ratios[mask].max()) if mask.any() else None


def calibrate(times, ratios, confirmed_times):
    """Fit this bout's firing threshold from lunges the user has confirmed.

    Returns (threshold, n_used) or (None, n_used) when there is not enough to
    fit on.
    """
    peaks = [p for p in (peak_in_window(times, ratios, t) for t in confirmed_times)
             if p is not None]
    if len(peaks) < MIN_CALIBRATION_LUNGES:
        return None, len(peaks)
    return float(np.percentile(peaks, CALIBRATION_PERCENTILE)), len(peaks)


def propose(times, ratios, threshold, skip_before=None):
    """
    Every window whose stance ratio clears the threshold, as proposed lunges.

    Windows step by a full width so two proposals cannot describe the same
    moment. `skip_before` suppresses the stretch the calibration came from, so
    the user is not offered back the lunges they just confirmed.
    """
    if threshold is None or len(times) < 2:
        return []

    grid = np.arange(times.min() + WINDOW_S, times.max() - WINDOW_S, 2 * WINDOW_S)
    firing = []
    for centre in grid:
        if skip_before is not None and centre < skip_before:
            continue
        p = peak_in_window(times, ratios, centre)
        if p is not None and p >= threshold:
            firing.append((float(centre), p))

    # Adjacent firing windows are ONE lunge, not several. A lunge lasting about
    # a third of a second sits inside two neighbouring windows whenever it
    # straddles their boundary, so without merging every such lunge is proposed
    # twice: the user is asked the same question twice, and precision is
    # understated because the second proposal can never match a new label.
    merged = []
    for centre, peak in firing:
        if merged and centre - merged[-1]["_end"] <= 2 * WINDOW_S:
            if peak > merged[-1]["stance_ratio"]:
                merged[-1]["time_s"] = centre
                merged[-1]["stance_ratio"] = peak
            merged[-1]["_end"] = centre
            continue
        merged.append({"time_s": centre, "stance_ratio": peak, "_end": centre})

    out = []
    for m in merged:
        out.append({"time_s": round(m["time_s"], 2),
                    "stance_ratio": round(m["stance_ratio"], 3),
                    # How far above the bar, which is the only confidence this
                    # method can honestly offer.
                    "margin": round(m["stance_ratio"] / threshold, 2)})
    return out


def evaluate(times, ratios, confirmed, k=MIN_CALIBRATION_LUNGES):
    """
    Calibrate on the first k confirmed lunges and score the rest of the bout.

    The split is by TIME, not at random: calibration uses the first k lunges and
    the test region starts after them, so no lunge informs its own proposal and
    the protocol matches how the feature would actually be used.
    """
    confirmed = sorted(confirmed)
    if len(confirmed) <= k + 1:
        return None
    cal, rest = confirmed[:k], confirmed[k:]
    threshold, used = calibrate(times, ratios, cal)
    if threshold is None:
        return None

    start = cal[-1] + 2.0
    test = [t for t in rest if t >= start]
    if not test:
        return None

    proposals = propose(times, ratios, threshold, skip_before=start)
    matched = set()
    hits = 0
    for p in proposals:
        near = [t for t in test if abs(t - p["time_s"]) <= 2 * WINDOW_S]
        if near:
            hits += 1
            matched.update(near)

    precision = hits / len(proposals) if proposals else 0.0
    recall = len(matched) / len(test)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    n_windows = len(np.arange(start + WINDOW_S, times.max() - WINDOW_S,
                              2 * WINDOW_S))
    chance = len(test) / max(n_windows, 1)
    return {"threshold": round(threshold, 3), "calibrated_on": used,
            "test_lunges": len(test), "proposals": len(proposals),
            "precision": round(precision, 2), "recall": round(recall, 2),
            "f1": round(f1, 2), "chance_rate": round(chance, 3),
            "lift": round(precision / chance, 1) if chance else None}


def load_confirmed(annotation_path, slot):
    with open(annotation_path) as f:
        data = json.load(f)
    return sorted(l["time_s"] for l in data.get("lunges", [])
                  if l["slot"] == slot)


def main():
    p = argparse.ArgumentParser(
        description="Propose lunges, calibrated on this bout's confirmed ones")
    p.add_argument("--csv", required=True,
                   help="a *_distance.csv carrying the stance columns")
    p.add_argument("--annotations", required=True,
                   help="the bout's annotation JSON, for confirmed lunges")
    p.add_argument("--slot", type=int, default=1, choices=(0, 1))
    p.add_argument("--evaluate", action="store_true",
                   help="calibrate on the first few and score the rest")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    times, ratios = stance_ratio_series(args.csv, args.slot)
    confirmed = load_confirmed(args.annotations, args.slot)
    if len(times) == 0:
        print("no stance data in this CSV. Was it produced before the stance "
              "columns existed?")
        return 1
    print(f"{len(times)} pose samples, {len(confirmed)} confirmed lunges "
          f"for fencer {args.slot + 1}")

    if args.evaluate:
        result = evaluate(times, ratios, confirmed)
        if result is None:
            print(f"not enough confirmed lunges to evaluate: needs more than "
                  f"{MIN_CALIBRATION_LUNGES + 1}")
            return 1
        for k, v in result.items():
            print(f"  {k}: {v}")
        return 0

    threshold, used = calibrate(times, ratios, confirmed)
    if threshold is None:
        print(f"only {used} usable confirmed lunges; needs "
              f"{MIN_CALIBRATION_LUNGES}. Confirm a few more and try again.")
        return 1
    proposals = propose(times, ratios, threshold)
    print(f"threshold {threshold:.3f} from {used} confirmed lunges "
          f"-> {len(proposals)} proposals")
    for pr in proposals[:20]:
        print(f"  {pr['time_s']:>7.2f}s  ratio {pr['stance_ratio']}  "
              f"margin {pr['margin']}x")
    if len(proposals) > 20:
        print(f"  ... and {len(proposals) - 20} more")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(proposals, f, indent=2)
        print(f"Written -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
