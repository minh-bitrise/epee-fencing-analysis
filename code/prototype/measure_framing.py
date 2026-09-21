"""
Measure how large the fencers are in the detector's input, per clip.
====================================================================

Figure 4.4 originally carried these numbers as four hand-typed constants. They went
stale the moment the evaluation set grew, and because they were constants nothing
detected it: the figure kept plotting four clips while the chapter around it had
moved to nine. This script measures them instead, for every clip in the evaluation
set, and writes a CSV the figure reads.

    python3 measure_framing.py

Two quantities come out of it, and separating them is the point.

  fencer_px   Median detected person height, expressed in the network's input
              frame. The detector resizes so the longest side is 640, so source
              resolution is invisible to it and a 360p clip is not handicapped by
              being 360p. What can handicap it is the fencers occupying less of
              the frame.

  short_gap   Frames lost to gaps of at most one second, as a percentage of the
              clip. This is the detector losing and regaining the fencers.

  coverage    Frames where both fencers were tracked, as a percentage.

Coverage alone conflates two unrelated things. A fencer who walks off after a touch
while the referee confers is absent, not undetected, and that absence lands in the
coverage figure identically to a detector failure. Splitting the missing frames by
gap length separates them: breaks are long and few, detector failures are short and
many. The distinction matters because only one of them is a defect.
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
