"""
Attribute a touch to a fencer by reading the scoring machine's lamps.

The touch detector says WHEN a touch happened and never WHO scored. Lamps rather
than a score overlay, because club footage has no overlay, only the machine. This
is not a touch detector: lamps fire whenever the circuit closes, so it attributes
only touches that detection has already proposed.
"""

import argparse
import csv
import json
import sys

import cv2
import numpy as np

# A lamp is a small, very saturated, bright patch. These bounds are deliberately
# generous: the point is to catch the lamp on footage of any quality, and the
# local baseline below removes whatever else they let through.
SAT_MIN = 140
VAL_MIN = 110
RED_HUE_LOW = 8            # OpenCV hue is 0-179, so red wraps at both ends
RED_HUE_HIGH = 172
GREEN_HUE_LOW = 45
GREEN_HUE_HIGH = 90

# Where to look relative to a touch. The lamp lights on contact and stays lit
# until the referee resets the machine, so the window runs forward, and the
# baseline is taken from before the action that produced the touch.
PEAK_WINDOW_S = (0.0, 2.0)
PEAK_STEP_S = 0.4
BASELINE_OFFSETS_S = (2.0, 2.5, 3.0)


def lamp_counts(frame):
    """Saturated red and green pixel counts for one frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(int)
    s = hsv[..., 1].astype(int)
    v = hsv[..., 2].astype(int)
    bright = (s > SAT_MIN) & (v > VAL_MIN)
    red = bright & ((h <= RED_HUE_LOW) | (h >= RED_HUE_HIGH))
    green = bright & (h >= GREEN_HUE_LOW) & (h <= GREEN_HUE_HIGH)
    return int(red.sum()), int(green.sum())


def _at(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    return lamp_counts(frame) if ok else None


def lamp_response(video_path, times):
    """
    For each touch time, how much red and green switched ON around it.

    Returns a list of {"time_s", "red", "green"}, the counts being above the
    local baseline rather than absolute.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out = []
    for t in times:
        peak = (0, 0)
        offset = PEAK_WINDOW_S[0]
        while offset <= PEAK_WINDOW_S[1]:
            c = _at(cap, fps, t + offset)
            if c and sum(c) > sum(peak):
                peak = c
            offset += PEAK_STEP_S

        samples = [_at(cap, fps, t - d) for d in BASELINE_OFFSETS_S]
        samples = [s for s in samples if s]
        if samples:
            base = (float(np.median([s[0] for s in samples])),
                    float(np.median([s[1] for s in samples])))
        else:
            base = (0.0, 0.0)

        out.append({"time_s": float(t),
                    "red": peak[0] - base[0],
                    "green": peak[1] - base[1]})
    cap.release()
    return out


def fit_thresholds(responses, labels, green_is="right"):
    """Choose a per-bout firing threshold for each lamp from confirmed touches.

    Split at the midpoint between the highest value where the lamp should be
    off and the lowest where it should be on, which is the widest-margin split
    of a one-dimensional set.
    """
    red_side = "left" if green_is == "right" else "right"

    def split(values_on, values_off, fallback_from):
        if values_on and values_off:
            lo, hi = max(values_off), min(values_on)
            if lo < hi:
                return (lo + hi) / 2.0
            return (lo + hi) / 2.0
        if values_on:
            return max(1.0, min(values_on) / 2.0)
        if values_off:
            return max(values_off) + 1.0
        return float(fallback_from)

    green_on = [r["green"] for r, l in zip(responses, labels)
                if l in (green_is, "double")]
    green_off = [r["green"] for r, l in zip(responses, labels) if l == red_side]
    red_on = [r["red"] for r, l in zip(responses, labels)
              if l in (red_side, "double")]
    red_off = [r["red"] for r, l in zip(responses, labels) if l == green_is]

    return {"green": split(green_on, green_off, 1.0),
            "red": split(red_on, red_off, 1.0),
            "green_is": green_is}


def classify(response, thresholds):
    """Turn one lamp reading into a proposed scorer, with a confidence."""
    green_is = thresholds["green_is"]
    red_is = "left" if green_is == "right" else "right"
    g_on = response["green"] >= thresholds["green"]
    r_on = response["red"] >= thresholds["red"]

    if g_on and r_on:
        scorer = "double"
    elif g_on:
        scorer = green_is
    elif r_on:
        scorer = red_is
    else:
        # Neither lamp cleared its threshold. Reported as unknown rather than
        # guessed: the interface already has a way for the user to say who
        # scored, and a confident wrong answer costs them more than a blank.
        return {"scorer": "unknown", "confidence": 0.0}

    # Confidence is how far the deciding lamp cleared its threshold, capped at 1.
    margins = []
    if g_on:
        margins.append(response["green"] / max(thresholds["green"], 1.0))
    if r_on:
        margins.append(response["red"] / max(thresholds["red"], 1.0))
    return {"scorer": scorer, "confidence": round(min(1.0, min(margins) / 3.0), 2)}


def read_touches(path):
    rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))
    return ([float(r["time_s"]) for r in rows],
            [r.get("scorer", "unknown") for r in rows])


def leave_one_out(responses, labels, green_is):
    """Predict each touch from thresholds fitted WITHOUT it.

    The only honest protocol available at this sample size.
    """
    predictions = []
    for i in range(len(responses)):
        others = [r for j, r in enumerate(responses) if j != i]
        others_labels = [l for j, l in enumerate(labels) if j != i]
        th = fit_thresholds(others, others_labels, green_is=green_is)
        predictions.append(classify(responses[i], th))
    return predictions


def main():
    p = argparse.ArgumentParser(description="Attribute touches to a fencer")
    p.add_argument("--video", required=True)
    p.add_argument("--touches", required=True,
                   help="CSV of touch times; a scorer column enables --evaluate")
    p.add_argument("--green-is", default="right", choices=("left", "right"),
                   help="Which fencer the green lamp belongs to. Nothing in the "
                        "image says this, so it is confirmed once per bout.")
    p.add_argument("--evaluate", action="store_true",
                   help="Leave-one-out accuracy against the scorer column")
    p.add_argument("--output", default=None, help="Write proposals as JSON")
    args = p.parse_args()

    times, labels = read_touches(args.touches)
    responses = lamp_response(args.video, times)

    if args.evaluate:
        preds = leave_one_out(responses, labels, args.green_is)
        correct = sum(1 for pr, l in zip(preds, labels) if pr["scorer"] == l)
        unknown = sum(1 for pr in preds if pr["scorer"] == "unknown")
        print(f"{'t':>7} {'truth':>7} {'proposed':>9} {'conf':>6}  "
              f"{'dRed':>8} {'dGreen':>8}")
        for t, l, pr, r in zip(times, labels, preds, responses):
            mark = "ok" if pr["scorer"] == l else "MISS"
            print(f"{t:>7.0f} {l:>7} {pr['scorer']:>9} {pr['confidence']:>6.2f}  "
                  f"{r['red']:>8.0f} {r['green']:>8.0f}  {mark}")
        decided = len(preds) - unknown
        print(f"\nthree-way (left/right/double): correct {correct}/{len(preds)}"
              f"  ({100.0*correct/len(preds):.0f}%)"
              f"   decided {decided}/{len(preds)}"
              + (f", accuracy when decided {100.0*correct/decided:.0f}%"
                 if decided else ""))

        # The green channel on its own, which is the question it can actually
        # answer. Red is contaminated by everything permanently red in shot and
        # is where nearly every three-way error comes from; green is not.
        # Reported separately because "was this fencer involved" is a real,
        # useful answer even when "exactly who scored" is not available.
        green_is = args.green_is
        red_side = "left" if green_is == "right" else "right"
        g_correct = 0
        for i in range(len(responses)):
            others = [r for j, r in enumerate(responses) if j != i]
            others_labels = [l for j, l in enumerate(labels) if j != i]
            th = fit_thresholds(others, others_labels, green_is=green_is)
            involved = responses[i]["green"] >= th["green"]
            truly = labels[i] in (green_is, "double")
            g_correct += (involved == truly)
        determined = sum(1 for l in labels if l == red_side)
        print(f"green lamp alone (was the {green_is} fencer involved): "
              f"{g_correct}/{len(labels)} ({100.0*g_correct/len(labels):.0f}%)")
        print(f"  of which fully determined without the red lamp: {determined} "
              f"({red_side}-only touches); the rest narrow from three choices "
              f"to two")
    else:
        th = fit_thresholds(responses, labels, green_is=args.green_is)
        preds = [classify(r, th) for r in responses]
        for t, pr in zip(times, preds):
            print(f"{t:>7.1f}s  {pr['scorer']:>8}  confidence {pr['confidence']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([{"time_s": t, **pr} for t, pr in zip(times, preds)],
                      f, indent=2)
        print(f"\nWritten -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
