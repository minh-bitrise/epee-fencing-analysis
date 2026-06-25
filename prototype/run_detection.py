"""
Epee Fencing Bout Analysis - Prototype v2
==========================================
Detects and tracks two fencers using YOLOv8 + ByteTrack, then runs
MediaPipe Pose on each fencer crop to extract body keypoints.

Distance is measured front-foot to front-foot (the most tactically
relevant measure in epee fencing). Falls back to bounding-box-centre
distance if pose estimation fails on a frame.

Outputs:
  - annotated video (boxes, fencer IDs, pose keypoints, distance overlay)
  - CSV of frame-by-frame distance data
  - distance-over-time plot (PNG)

Usage:
    python run_detection.py --video path/to/bout.mp4
    python run_detection.py --video path/to/bout.mp4 --output results/
"""

import argparse
import csv
import math
import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO


# --- config -----------------------------------------------------------
MODEL_NAME       = "yolov8n.pt"
POSE_MODEL_PATH  = "pose_landmarker.task"   # downloaded MediaPipe model file
CONF_THRESH      = 0.4
IOU_THRESH       = 0.5
MAX_FENCERS      = 2
REAL_HEIGHT_M    = 1.75       # assumed average fencer height for normalisation

COLOURS = [(0, 200, 255), (0, 255, 100)]  # BGR per fencer

# MediaPipe landmark indices
LM_LEFT_ANKLE     = 27
LM_RIGHT_ANKLE    = 28
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
# ----------------------------------------------------------------------

# lazy-loaded pose landmarker (created once inside run(), not at import time)
_pose_landmarker = None


def _get_pose_landmarker():
    """Return a shared PoseLandmarker instance, creating it on first call."""
    global _pose_landmarker
    if _pose_landmarker is None:
        base_options = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def get_box_centre(box):
    """Return the (x, y) centre of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_height_pixels(box):
    """Return the pixel height of a bounding box."""
    _, y1, _, y2 = box
    return y2 - y1


def pixel_distance(c1, c2):
    """Euclidean pixel distance between two points."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def normalise_distance(dist_px, ref_height_px, real_height_m=REAL_HEIGHT_M):
    """
    Convert a pixel distance to an estimated real-world distance in metres.
    Uses the average fencer bounding-box height as a scale reference.
    Returns None if ref_height_px is zero (avoids division by zero).
    """
    if ref_height_px == 0:
        return None
    return (dist_px / ref_height_px) * real_height_m


def get_pose_landmarks(frame_bgr, box):
    """
    Run MediaPipe Pose on the region of the frame defined by box.
    Returns a dict of {landmark_index: (abs_x, abs_y)} in full-frame
    pixel coordinates, or an empty dict if pose detection fails.
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    pad = 10
    h, w = frame_bgr.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return {}

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    landmarker = _get_pose_landmarker()
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {}

    crop_h, crop_w = crop.shape[:2]
    landmarks = {}
    for idx, lm in enumerate(result.pose_landmarks[0]):
        abs_x = cx1 + lm.x * crop_w
        abs_y = cy1 + lm.y * crop_h
        landmarks[idx] = (abs_x, abs_y)
    return landmarks


def get_front_foot(landmarks, fencer_centre_x, opponent_centre_x):
    """
    Determine which ankle is the 'front foot' (the one facing the opponent).
    In a side-on view, the front foot is the ankle closer to the opponent.
    Returns the (x, y) of the front ankle, or None if landmarks unavailable.
    """
    left_ankle  = landmarks.get(LM_LEFT_ANKLE)
    right_ankle = landmarks.get(LM_RIGHT_ANKLE)

    if left_ankle is None and right_ankle is None:
        return None
    if left_ankle is None:
        return right_ankle
    if right_ankle is None:
        return left_ankle

    # front foot = whichever ankle is closer to the opponent's side
    if opponent_centre_x > fencer_centre_x:
        # opponent is to the right — front foot is the one with higher x
        return left_ankle if left_ankle[0] > right_ankle[0] else right_ankle
    else:
        # opponent is to the left — front foot is the one with lower x
        return left_ankle if left_ankle[0] < right_ankle[0] else right_ankle


def draw_overlay(frame, track_data, pose_data, dist_m, dist_method, frame_idx, fps):
    """Draw bounding boxes, pose keypoints, and distance overlay on a frame."""
    for i, (box, track_id) in enumerate(track_data):
        x1, y1, x2, y2 = [int(v) for v in box]
        colour = COLOURS[i % len(COLOURS)]
        label  = f"Fencer {track_id}"

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # draw key pose landmarks if available
        if i < len(pose_data) and pose_data[i]:
            lms = pose_data[i]
            key_points = [
                LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
                LM_LEFT_HIP, LM_RIGHT_HIP,
                LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
            ]
            for lm_idx in key_points:
                if lm_idx in lms:
                    px, py = int(lms[lm_idx][0]), int(lms[lm_idx][1])
                    cv2.circle(frame, (px, py), 4, colour, -1)

            # draw front foot marker (larger dot)
            centres = [get_box_centre(t[0]) for t in track_data]
            opp_idx = 1 - i
            if opp_idx < len(centres):
                front = get_front_foot(lms, centres[i][0], centres[opp_idx][0])
                if front:
                    cv2.circle(frame, (int(front[0]), int(front[1])), 7, (255, 255, 255), 2)

    # distance overlay (top-left)
    method_label = "pose" if dist_method == "pose" else "bbox"
    if dist_m is not None:
        dist_text = f"Distance ({method_label}): {dist_m:.2f} m"
    else:
        dist_text = "Distance: --"

    time_sec  = frame_idx / fps if fps > 0 else 0
    time_text = f"Time: {time_sec:.1f}s"

    cv2.putText(frame, dist_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, time_text, (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame


def save_plot(times, distances, methods, output_path):
    """Save a distance-over-time chart, colouring pose vs bbox frames differently."""
    pose_t  = [t for t, m in zip(times, methods) if m == "pose"]
    pose_d  = [d for d, m in zip(distances, methods) if m == "pose"]
    bbox_t  = [t for t, m in zip(times, methods) if m == "bbox"]
    bbox_d  = [d for d, m in zip(distances, methods) if m == "bbox"]

    fig, ax = plt.subplots(figsize=(12, 4))
    if pose_t:
        ax.scatter(pose_t, pose_d, s=2, color="#2196F3", label="Pose (front foot)", alpha=0.7)
    if bbox_t:
        ax.scatter(bbox_t, bbox_d, s=2, color="#FF9800", label="Fallback (bbox centre)", alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Estimated distance (metres)")
    ax.set_title("Inter-fencer Distance Over Time (front foot to front foot)")
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {output_path}")


def run(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base      = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base}_annotated.mp4")
    out_csv   = os.path.join(output_dir, f"{base}_distance.csv")
    out_plot  = os.path.join(output_dir, f"{base}_distance_plot.png")

    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    csv_rows  = []
    times     = []
    distances = []
    methods   = []

    pose_success = 0
    frame_idx    = 0
    print("Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=[0],
            verbose=False,
        )

        track_data = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes  = results[0].boxes
            ids    = boxes.id.cpu().numpy().astype(int)
            xyxys  = boxes.xyxy.cpu().numpy()
            confs  = boxes.conf.cpu().numpy()
            order  = np.argsort(confs)[::-1][:MAX_FENCERS]
            for idx in order:
                track_data.append((xyxys[idx], ids[idx]))

        # run pose on each fencer crop
        pose_data = []
        for box, _ in track_data:
            lms = get_pose_landmarks(frame, box)
            pose_data.append(lms)

        # compute distance
        dist_m  = None
        method  = None

        if len(track_data) == 2:
            c0 = get_box_centre(track_data[0][0])
            c1 = get_box_centre(track_data[1][0])
            h0 = box_height_pixels(track_data[0][0])
            h1 = box_height_pixels(track_data[1][0])
            avg_height = (h0 + h1) / 2

            # try pose-based front-foot distance first
            front0 = get_front_foot(pose_data[0], c0[0], c1[0]) if pose_data[0] else None
            front1 = get_front_foot(pose_data[1], c1[0], c0[0]) if pose_data[1] else None

            if front0 and front1:
                dist_px = pixel_distance(front0, front1)
                dist_m  = normalise_distance(dist_px, avg_height)
                method  = "pose"
                pose_success += 1
            else:
                # fallback to bounding box centres
                dist_px = pixel_distance(c0, c1)
                dist_m  = normalise_distance(dist_px, avg_height)
                method  = "bbox"

        frame = draw_overlay(frame, track_data, pose_data, dist_m, method, frame_idx, fps)
        writer.write(frame)

        time_sec = frame_idx / fps
        csv_rows.append({
            "frame":            frame_idx,
            "time_s":           round(time_sec, 3),
            "distance_m":       round(dist_m, 3) if dist_m is not None else "",
            "method":           method or "",
            "fencers_detected": len(track_data),
        })
        if dist_m is not None:
            times.append(time_sec)
            distances.append(dist_m)
            methods.append(method)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames processed")

    cap.release()
    writer.release()
    print(f"  Annotated video saved -> {out_video}")

    with open(out_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(
            f, fieldnames=["frame", "time_s", "distance_m", "method", "fencers_detected"])
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)
    print(f"  CSV saved -> {out_csv}")

    if distances:
        save_plot(times, distances, methods, out_plot)

    if distances:
        pose_pct = (pose_success / len(distances)) * 100
        print("\n--- Summary ---")
        print(f"  Frames processed:          {frame_idx}")
        print(f"  Frames with 2 fencers:     {len(distances)}")
        print(f"  Pose-based distance:       {pose_success} frames ({pose_pct:.1f}%)")
        print(f"  Fallback (bbox) distance:  {len(distances) - pose_success} frames")
        print(f"  Mean distance:             {np.mean(distances):.2f} m")
        print(f"  Min distance:              {np.min(distances):.2f} m")
        print(f"  Max distance:              {np.max(distances):.2f} m")
        print(f"  Std deviation:             {np.std(distances):.2f} m")
    else:
        print("  No distance data recorded.")


def main():
    parser = argparse.ArgumentParser(description="Fencing bout fencer detection and distance analysis")
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument("--output", default="results", help="Output folder (default: results/)")
    args = parser.parse_args()
    run(args.video, args.output)


if __name__ == "__main__":
    main()
