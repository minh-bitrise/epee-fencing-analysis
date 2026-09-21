"""
Measure how large the fencers are in the detector's input, per clip, and split
tracking loss into the detector's failures and the fencers' absences.

    python3 measure_framing.py

Coverage alone conflates the two. A fencer who walks off after a touch is absent,
not undetected, and that lands in coverage identically to a detection failure.
Gap length separates them: breaks are long and few, detector failures short and
many. Writes results_current/framing.csv, which the figures read, because these
numbers were four typed constants and went stale when the set grew.
"""

import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from clips import CLIPS
from run_detection import (CONF_THRESH, IOU_THRESH, MAX_FENCERS, MODEL_NAME,
                           box_height_pixels)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_current", "framing.csv")

SAMPLE_EVERY = 25    # one frame a second; the median is stable well before this matters
MAX_FRAMES = 3000    # two minutes of samples is plenty for a median, and bounds the runtime
SHORT_GAP_S = 1.0    # gaps at or under this are the detector faltering, not a break in play


def gap_runs(rows):
    """Lengths of the contiguous runs of frames where the pair was not tracked."""
    runs, n = [], 0
    for r in rows:
        if not r["distance_smooth_m"]:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def fencer_height_px(video_path, model):
    """
    Median fencer height in the detector's input frame, in pixels.

    The two tallest boxes per sampled frame are taken rather than every box, for the
    same reason the pipeline does: officials and spectators are people too, and on
    the wider clips there are more of them than there are fencers.
    """
    cap = cv2.VideoCapture(video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = 640.0 / max(w, h)
    heights, idx = [], 0
    while idx < min(n, MAX_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % SAMPLE_EVERY == 0:
            res = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                classes=[0], verbose=False)
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                boxes = res[0].boxes.xyxy.cpu().numpy()
                tallest = sorted(boxes, key=box_height_pixels, reverse=True)[:MAX_FENCERS]
                heights.extend(box_height_pixels(b) for b in tallest)
        idx += 1
    cap.release()
    if not heights:
        return None, int(w), int(h), 0
    return float(np.median(heights)) * scale, int(w), int(h), len(heights)


def main():
    model = YOLO(MODEL_NAME)
    rows_out = []
    for clip in CLIPS:
        video = os.path.join(HERE, f"{clip}.mp4")
        series = os.path.join(HERE, "results_current", f"{clip}_distance.csv")
        if not os.path.exists(video) or not os.path.exists(series):
            print(f"skipping {clip}: missing video or distance series")
            continue

        px, w, h, n = fencer_height_px(video, model)
        rows = list(csv.DictReader(open(series)))
        runs = gap_runs(rows)
        fps = 25.0
        short = sum(r for r in runs if r <= SHORT_GAP_S * fps)
        total = sum(runs)
        rows_out.append({
            "clip": clip,
            "source_w": w,
            "source_h": h,
            "fencer_px": round(px, 1) if px else "",
            "samples": n,
            "coverage_pct": round(100.0 * (1 - total / len(rows)), 1),
            "short_gap_pct": round(100.0 * short / len(rows), 2),
            "gaps": len(runs),
            "longest_gap_frames": max(runs) if runs else 0,
        })
        print(f"{clip:<16} {w}x{h}  fencer {px:5.0f} px  coverage "
              f"{rows_out[-1]['coverage_pct']:5.1f}%  short-gap loss "
              f"{rows_out[-1]['short_gap_pct']:5.2f}%")

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wr.writeheader()
        wr.writerows(rows_out)
    print(f"\nwrote {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
