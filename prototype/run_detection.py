"""
Epee Fencing Bout Analysis - Prototype
=======================================
Detects and tracks two fencers in a video using YOLOv8 + ByteTrack.
Computes the inter-fencer distance over time and outputs:
  - annotated video with bounding boxes, fencer IDs, and distance overlay
  - CSV of frame-by-frame distance data
  - distance-over-time plot (PNG)

Usage:
    python run_detection.py --video path/to/bout.mp4
    python run_detection.py --video path/to/bout.mp4 --output results/
"""

import argparse
import csv
import os
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# --- config -----------------------------------------------------------
MODEL_NAME   = "yolov8n.pt"   # nano model: small and fast; swap for yolov8s.pt for better accuracy
CONF_THRESH  = 0.4            # minimum detection confidence to keep a box
IOU_THRESH   = 0.5            # non-max suppression threshold
MAX_FENCERS  = 2              # we only care about 2 people per frame
# colours for the two fencers (BGR for OpenCV)
COLOURS = [(0, 200, 255), (0, 255, 100)]
# ----------------------------------------------------------------------


def get_box_centre(box):
    """Return the (x, y) centre of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_height_pixels(box):
    """Return the pixel height of a bounding box — used as a rough scale reference."""
    _, y1, _, y2 = box
    return y2 - y1


def pixel_distance(c1, c2):
    """Euclidean pixel distance between two centre points."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def normalise_distance(dist_px, height_px, real_height_m=1.75):
    """
    Convert pixel distance to an approximate real-world distance in metres.
    Uses the average fencer height (assumed 1.75 m) as a scale reference.
    This is a rough estimate — accuracy depends on camera angle and lens.
    """
    if height_px == 0:
        return None
    return (dist_px / height_px) * real_height_m


def draw_overlay(frame, track_data, dist_norm, frame_idx, fps):
    """Draw bounding boxes, fencer labels, and distance text on a frame."""
    for i, (box, track_id) in enumerate(track_data):
        x1, y1, x2, y2 = [int(v) for v in box]
        colour = COLOURS[i % len(COLOURS)]
        label  = f"Fencer {track_id}"

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # distance overlay (top-left corner)
    if dist_norm is not None:
        dist_text = f"Distance: {dist_norm:.2f} m"
    else:
        dist_text = "Distance: --"

    time_sec = frame_idx / fps if fps > 0 else 0
    time_text = f"Time: {time_sec:.1f}s"

    cv2.putText(frame, dist_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, time_text, (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame


def save_plot(times, distances, output_path):
    """Save a distance-over-time line chart as a PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, distances, color="#2196F3", linewidth=1.2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Estimated distance (metres)")
    ax.set_title("Inter-fencer Distance Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {output_path}")


def run(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base        = os.path.splitext(os.path.basename(video_path))[0]
    out_video   = os.path.join(output_dir, f"{base}_annotated.mp4")
    out_csv     = os.path.join(output_dir, f"{base}_distance.csv")
    out_plot    = os.path.join(output_dir, f"{base}_distance_plot.png")

    # load model (downloads yolov8n.pt automatically on first run)
    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total} frames")

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    # data collectors
    csv_rows   = []
    times      = []
    distances  = []

    frame_idx = 0
    print("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run YOLOv8 tracking on this frame
        # persist=True tells the tracker to keep IDs consistent across frames
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=[0],        # class 0 = "person" in COCO
            verbose=False,
        )

        # extract the top-2 person detections (by confidence)
        track_data   = []   # list of (box, track_id)
        dist_norm    = None

        if results[0].boxes is not None:
            boxes  = results[0].boxes
            # filter to detections that have a track ID
            mask   = boxes.id is not None
            if mask:
                ids    = boxes.id.cpu().numpy().astype(int)
                xyxys  = boxes.xyxy.cpu().numpy()
                confs  = boxes.conf.cpu().numpy()

                # sort by confidence descending, keep top MAX_FENCERS
                order  = np.argsort(confs)[::-1][:MAX_FENCERS]
                for rank, idx in enumerate(order):
                    track_data.append((xyxys[idx], ids[idx]))

        # compute distance if we have exactly 2 detections
        if len(track_data) == 2:
            c1 = get_box_centre(track_data[0][0])
            c2 = get_box_centre(track_data[1][0])
            h1 = box_height_pixels(track_data[0][0])
            h2 = box_height_pixels(track_data[1][0])
            avg_height = (h1 + h2) / 2

            dist_px   = pixel_distance(c1, c2)
            dist_norm = normalise_distance(dist_px, avg_height)

        # draw and write frame
        frame = draw_overlay(frame, track_data, dist_norm, frame_idx, fps)
        writer.write(frame)

        # record data
        time_sec = frame_idx / fps
        csv_rows.append({
            "frame":      frame_idx,
            "time_s":     round(time_sec, 3),
            "distance_m": round(dist_norm, 3) if dist_norm is not None else "",
            "fencers_detected": len(track_data),
        })
        if dist_norm is not None:
            times.append(time_sec)
            distances.append(dist_norm)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames processed")

    cap.release()
    writer.release()
    print(f"  Annotated video saved -> {out_video}")

    # save CSV
    with open(out_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "time_s", "distance_m", "fencers_detected"])
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)
    print(f"  CSV saved -> {out_csv}")

    # save plot
    if distances:
        save_plot(times, distances, out_plot)
    else:
        print("  No distance data to plot (were 2 fencers visible?)")

    # print summary stats
    if distances:
        print("\n--- Summary ---")
        print(f"  Frames processed:     {frame_idx}")
        print(f"  Frames with 2 fencers detected: {len(distances)}")
        print(f"  Mean distance:        {np.mean(distances):.2f} m")
        print(f"  Min distance:         {np.min(distances):.2f} m")
        print(f"  Max distance:         {np.max(distances):.2f} m")
        print(f"  Std deviation:        {np.std(distances):.2f} m")


def main():
    parser = argparse.ArgumentParser(description="Fencing bout fencer detection and distance analysis")
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument("--output", default="results", help="Output folder (default: results/)")
    args = parser.parse_args()
    run(args.video, args.output)


if __name__ == "__main__":
    main()
