"""
Epee Fencing Bout Analysis - Prototype v3
==========================================
Detects and tracks two fencers using YOLOv8 + ByteTrack, then runs
MediaPipe Pose on each fencer crop to extract body keypoints.

Improvements over v2:
  - Fencer identity is locked onto the first two persistent track IDs,
    so "Fencer 1" / "Fencer 2" labels stay stable for the whole bout.
  - Cumulative advance ("push") and retreat ("pull") tracked per fencer.
  - Displayed distance is rolling-median smoothed (raw values still in CSV).
  - Distance overlay colour-coded by tactical zone (close / medium / far).
  - Bounding-box fallback uses bottom-centre (feet proxy), not box centre.
  - Per-frame movement clamped at a biomechanical limit to reduce the
    impact of camera panning on the push/pull metric.

Outputs:
  - annotated video (boxes, fencer IDs, pose keypoints, distance overlay,
    live push/pull readout)
  - CSV of frame-by-frame distance, smoothed distance, and per-fencer
    cumulative push / pull
  - distance-over-time plot (PNG)

Usage:
    python3 run_detection.py --video path/to/bout.mp4
    python3 run_detection.py --video path/to/bout.mp4 --output results/
"""

import argparse
import csv
import json
import math
import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO


# --- config -----------------------------------------------------------
MODEL_NAME       = "yolov8n.pt"
POSE_MODEL_PATH  = "pose_landmarker.task"
CONF_THRESH      = 0.4
IOU_THRESH       = 0.5
MAX_FENCERS      = 2
REAL_HEIGHT_M    = 1.75

# pose estimation is the expensive step; run it every Nth frame and fall
# back to bounding-box-bottom-centre distance on the in-between frames.
# at 60fps and stride=3 this still gives ~20 pose samples per second.
DEFAULT_POSE_STRIDE = 3

# rolling-median window for the distance value shown on screen
SMOOTH_WINDOW    = 5

# tactical zones for the coloured distance overlay (metres)
DIST_CLOSE_M     = 1.0    # within touch range
DIST_MEDIUM_M    = 1.8    # engagement range
COLOUR_CLOSE     = (0,   0,   255)    # red    (close, touch range)
COLOUR_MEDIUM    = (0,   165, 255)    # orange (engagement range)
COLOUR_FAR       = (0,   220, 100)    # green  (safe distance)

# max plausible per-frame fencer motion in metres
# (at 60fps, > 0.15 m/frame = > 9 m/s which exceeds the fastest lunges)
# anything beyond this is treated as camera motion or detection jitter
MAX_FRAME_MOVEMENT_M = 0.15

# minimum movement (m) to count toward push/pull (ignore detection jitter)
PUSH_PULL_NOISE_FLOOR_M = 0.03

# rolling median window applied to each fencer's reference x before
# computing frame-to-frame movement, to suppress bounding-box jitter
PUSH_PULL_SMOOTH_WINDOW = 5

# fencer colours (BGR)
COLOURS = [(0, 200, 255), (0, 255, 100)]

# MediaPipe landmark indices
LM_LEFT_ANKLE     = 27
LM_RIGHT_ANKLE    = 28
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
# ----------------------------------------------------------------------

# lazy-loaded pose landmarker (created on first use, not at import time)
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


# --- geometry helpers -------------------------------------------------

def get_box_centre(box):
    """Return the (x, y) centre of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_bottom_centre(box):
    """Return (x, y) of the bottom-centre of a bounding box - a feet-position proxy."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def box_height_pixels(box):
    """Return the pixel height of a bounding box."""
    _, y1, _, y2 = box
    return y2 - y1


def pixel_distance(c1, c2):
    """Euclidean pixel distance between two points."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def normalise_distance(dist_px, ref_height_px, real_height_m=REAL_HEIGHT_M):
    """
    Convert a pixel distance to an estimated real-world distance in metres,
    using the average fencer bounding-box height as a scale reference.
    Returns None if ref_height_px is zero (avoids division by zero).
    """
    if ref_height_px == 0:
        return None
    return (dist_px / ref_height_px) * real_height_m


def distance_zone_colour(dist_m):
    """Pick an overlay colour based on the tactical range of the current distance."""
    if dist_m is None:
        return (200, 200, 200)
    if dist_m <= DIST_CLOSE_M:
        return COLOUR_CLOSE
    if dist_m <= DIST_MEDIUM_M:
        return COLOUR_MEDIUM
    return COLOUR_FAR


# --- pose --------------------------------------------------------------

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
    return {
        idx: (cx1 + lm.x * crop_w, cy1 + lm.y * crop_h)
        for idx, lm in enumerate(result.pose_landmarks[0])
    }


def get_hip_centre(landmarks):
    """Return the midpoint between left and right hips, or None if unavailable."""
    lh = landmarks.get(LM_LEFT_HIP)
    rh = landmarks.get(LM_RIGHT_HIP)
    if lh is None and rh is None:
        return None
    if lh is None:
        return rh
    if rh is None:
        return lh
    return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)


def get_front_foot(landmarks, fencer_ref_x, opponent_ref_x):
    """
    Determine which ankle is the 'front foot' (closest to the opponent).
    Returns the (x, y) of the front ankle, or None if no ankles found.
    """
    left_ankle  = landmarks.get(LM_LEFT_ANKLE)
    right_ankle = landmarks.get(LM_RIGHT_ANKLE)

    if left_ankle is None and right_ankle is None:
        return None
    if left_ankle is None:
        return right_ankle
    if right_ankle is None:
        return left_ankle

    if opponent_ref_x > fencer_ref_x:
        # opponent to the right -> front foot has the higher x
        return left_ankle if left_ankle[0] > right_ankle[0] else right_ankle
    else:
        # opponent to the left -> front foot has the lower x
        return left_ankle if left_ankle[0] < right_ankle[0] else right_ankle


# --- piste region -----------------------------------------------------

class PisteRegion:
    """
    A polygon in pixel coordinates that marks the fencing strip (piste)
    in the frame. A detection is accepted only if its feet-proxy point
    (bounding-box bottom-centre) lies inside the polygon.

    This filter is applied BEFORE the FencerTracker matching stage, so
    it can also reject bystanders on the very first frame - the case
    where the tracker's motion / size gates have no history to work
    with and would otherwise let anything through.

    Piste polygons are loaded from a small JSON file so they can be
    hand-authored once per clip:

        { "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] }

    Any convex or concave polygon is supported. cv2.pointPolygonTest
    is used for the point-in-polygon check because OpenCV is already a
    dependency; there is no need to reimplement it.
    """

    def __init__(self, polygon):
        if polygon is None or len(polygon) < 3:
            raise ValueError("Piste polygon must have at least 3 vertices")
        self.polygon = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)

    @classmethod
    def from_json_file(cls, path):
        """Load a PisteRegion from a JSON file. Returns None if path is None."""
        if path is None:
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(data["polygon"])

    def contains(self, point):
        """Return True if point (x, y) is inside (or on) the polygon."""
        # cv2.pointPolygonTest returns +1 inside, 0 on the edge, -1 outside
        result = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def draw(self, frame, colour=(255, 200, 0), thickness=2):
        """
        Outline the piste region on a frame. Purely diagnostic: the filter
        is otherwise invisible in the output, so without this the only
        evidence it ran is the absence of boxes on bystanders.
        """
        pts = self.polygon.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness)
        label_pt = pts.reshape(-1, 2).min(axis=0)
        cv2.putText(frame, "piste region", (int(label_pt[0]) + 6, int(label_pt[1]) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return frame

    def filter_detections(self, ids, xyxys, confs):
        """
        Keep only detections whose bounding-box bottom-centre falls inside
        the piste polygon. Returns filtered (ids, xyxys, confs) arrays.
        """
        if len(ids) == 0:
            return ids, xyxys, confs
        keep = [self.contains(get_box_bottom_centre(b)) for b in xyxys]
        keep_idx = np.array([i for i, k in enumerate(keep) if k], dtype=int)
        if len(keep_idx) == 0:
            return (np.array([], dtype=int),
                    np.empty((0, 4)),
                    np.array([]))
        return ids[keep_idx], xyxys[keep_idx], confs[keep_idx]


# --- fencer identity --------------------------------------------------

class FencerTracker:
    """
    Maintains stable Fencer 1 / Fencer 2 slots by spatial continuity
    rather than by ByteTrack ID alone, which is unreliable across
    occlusions and rapid motion.

    On the first frame with at least two person detections, the
    leftmost is assigned to slot 0 (Fencer 1) and the rightmost to
    slot 1 (Fencer 2). On subsequent frames, the top-2 most confident
    person detections are matched to slots by minimising total
    distance from each slot's last known position.

    Each candidate assignment must additionally pass two gates,
    designed to keep background bystanders out of the fencer slots:

      * spatial gate - the candidate centre must be within
        GATE_DISTANCE_RATIO box-heights of the slot's last known
        centre (rejects detections that have jumped across the frame).
      * size gate - the candidate box height must be within
        [MIN_SIZE_RATIO, MAX_SIZE_RATIO] of the slot's last accepted
        height (rejects much smaller background people further from
        the camera).

    If a candidate fails its gate, that slot is left empty for the
    frame (no fake label) rather than swapping in a bystander.

    select() always returns a length-2 list (slots [A, B]); a slot
    is None if no detection was matched to it in this frame.
    """

    GATE_DISTANCE_RATIO = 3.5   # max jump from predicted position, in box-heights
    MIN_SIZE_RATIO      = 0.4   # candidate height must be >= this * last height
    MAX_SIZE_RATIO      = 2.2   # candidate height must be <= this * last height

    # Velocity is only extrapolated from two commits this close together.
    # Differencing two positions recorded many frames apart does not measure
    # velocity - it measures the total displacement over the gap - and
    # extrapolating from it throws the prediction far outside the frame.
    VELOCITY_MAX_GAP_FRAMES = 3

    # After this many consecutive frames with no match, a slot's history is
    # discarded so the slot can re-acquire a fencer. Without this, a slot
    # that stops matching can never recover: its stale history keeps
    # rejecting every candidate, and because nothing is committed the
    # history never updates.
    STALE_RESET_FRAMES = 30

    def __init__(self):
        self.last_pos  = [None, None]   # last (x, y) centre per slot
        self.prev_pos  = [None, None]   # centre one commit before last_pos
        self.last_h    = [None, None]   # last accepted box height per slot
        self.last_seen = [None, None]   # frame number of the last commit
        self.prev_seen = [None, None]   # frame number of the commit before that
        self.frame_no  = 0

    def predicted_pos(self, slot_idx):
        """
        Constant-velocity prediction of where slot_idx should be this frame.

        Velocity is estimated from the last two committed centres, but only
        when those commits were close enough together in time to represent
        an actual per-frame velocity. Otherwise the last known position is
        returned unextrapolated. Returns None if the slot has no history.
        """
        last = self.last_pos[slot_idx]
        if last is None:
            return None
        prev = self.prev_pos[slot_idx]
        if prev is None:
            return last
        gap = self.last_seen[slot_idx] - self.prev_seen[slot_idx]
        if gap > self.VELOCITY_MAX_GAP_FRAMES:
            return last
        return (last[0] + (last[0] - prev[0]),
                last[1] + (last[1] - prev[1]))

    def _expire_stale_slots(self):
        """Discard the history of any slot that has gone unmatched too long."""
        for slot in (0, 1):
            if self.last_seen[slot] is None:
                continue
            if self.frame_no - self.last_seen[slot] > self.STALE_RESET_FRAMES:
                self.last_pos[slot]  = None
                self.prev_pos[slot]  = None
                self.last_h[slot]    = None
                self.last_seen[slot] = None
                self.prev_seen[slot] = None

    def _passes_gate(self, slot_idx, new_centre, new_height):
        """Return True if a candidate is consistent with the slot's history."""
        if self.last_pos[slot_idx] is None or self.last_h[slot_idx] is None:
            return True  # no history yet, accept
        ref_h = self.last_h[slot_idx]
        if ref_h <= 0 or new_height <= 0:
            return False
        # gate distance is measured against the predicted position rather
        # than the last-known one - this catches the "bystander stepping
        # into the old position while the fencer has moved on" case, which
        # last-known-position gating cannot see.
        anchor = self.predicted_pos(slot_idx)
        if pixel_distance(new_centre, anchor) > self.GATE_DISTANCE_RATIO * ref_h:
            return False
        ratio = new_height / ref_h
        if ratio < self.MIN_SIZE_RATIO or ratio > self.MAX_SIZE_RATIO:
            return False
        return True

    def _commit(self, result, slot, box, tid):
        """Record a successful match into the slot and update history."""
        result[slot] = (box, tid)
        # shift last -> prev before overwriting, so predicted_pos() can
        # use a two-point history to estimate velocity next frame. The
        # frame numbers are shifted alongside the positions so the gap
        # between the two samples is known.
        self.prev_pos[slot]  = self.last_pos[slot]
        self.prev_seen[slot] = self.last_seen[slot]
        self.last_pos[slot]  = get_box_centre(box)
        self.last_seen[slot] = self.frame_no
        self.last_h[slot]    = box_height_pixels(box)

    def select(self, ids, xyxys, confs, max_fencers=MAX_FENCERS):
        self.frame_no += 1
        self._expire_stale_slots()

        result = [None, None]
        if len(ids) == 0:
            return result

        # take the two most confident person detections this frame
        order      = np.argsort(confs)[::-1][:max_fencers]
        boxes      = [xyxys[i]              for i in order]
        chosen_ids = [int(ids[i])           for i in order]
        centres    = [get_box_centre(b)     for b in boxes]
        heights    = [box_height_pixels(b)  for b in boxes]

        # case 1: not initialised yet - no gates, just establish slots
        if self.last_pos[0] is None and self.last_pos[1] is None:
            if len(boxes) >= 2:
                if centres[0][0] <= centres[1][0]:
                    self._commit(result, 0, boxes[0], chosen_ids[0])
                    self._commit(result, 1, boxes[1], chosen_ids[1])
                else:
                    self._commit(result, 0, boxes[1], chosen_ids[1])
                    self._commit(result, 1, boxes[0], chosen_ids[0])
            elif len(boxes) == 1:
                self._commit(result, 0, boxes[0], chosen_ids[0])
            return result

        # anchors used for matching costs. The predicted position (last
        # position projected forward by the slot's velocity) is preferred
        # over the raw last-known position - it makes the matcher choose
        # the detection consistent with where each fencer WAS GOING,
        # which is exactly what disambiguates close-range crossings.
        anchor0 = self.predicted_pos(0) or self.last_pos[0]
        anchor1 = self.predicted_pos(1) or self.last_pos[1]

        # case 2: one detection - assign to the closer slot, then gate
        if len(boxes) == 1:
            c, hgt = centres[0], heights[0]
            d0 = pixel_distance(c, anchor0) if anchor0 else float("inf")
            d1 = pixel_distance(c, anchor1) if anchor1 else float("inf")
            slot = 0 if d0 <= d1 else 1
            if self._passes_gate(slot, c, hgt):
                self._commit(result, slot, boxes[0], chosen_ids[0])
            return result

        # case 3: two detections, both slots have history
        # pick the cheaper assignment, then gate each match independently
        if self.last_pos[0] is not None and self.last_pos[1] is not None:
            cost_straight = (pixel_distance(centres[0], anchor0) +
                             pixel_distance(centres[1], anchor1))
            cost_swapped  = (pixel_distance(centres[0], anchor1) +
                             pixel_distance(centres[1], anchor0))
            pairs = [(0, 0), (1, 1)] if cost_straight <= cost_swapped else [(0, 1), (1, 0)]
            for det_idx, slot_idx in pairs:
                if self._passes_gate(slot_idx, centres[det_idx], heights[det_idx]):
                    self._commit(result, slot_idx, boxes[det_idx], chosen_ids[det_idx])
            return result

        # case 4: only one slot has history - assign closer detection there
        # (gated), and the other to the empty slot (no gate possible).
        known   = 0 if self.last_pos[0] is not None else 1
        unknown = 1 - known
        known_anchor = self.predicted_pos(known) or self.last_pos[known]
        d0 = pixel_distance(centres[0], known_anchor)
        d1 = pixel_distance(centres[1], known_anchor)
        best_for_known, other = (0, 1) if d0 <= d1 else (1, 0)
        if self._passes_gate(known, centres[best_for_known], heights[best_for_known]):
            self._commit(result, known, boxes[best_for_known], chosen_ids[best_for_known])
        self._commit(result, unknown, boxes[other], chosen_ids[other])
        return result


# --- push / pull (advance / retreat) tracking -------------------------

class PushPullTracker:
    """
    Accumulates how far each fencer has advanced (pushed forward toward
    the opponent) and retreated (moved backward) over the bout, in metres.

    Each fencer's reference x is first smoothed with a rolling median
    window to suppress bounding-box jitter; movement is then computed
    against the previous smoothed value. Per-frame movements greater
    than MAX_FRAME_MOVEMENT_M (treated as camera motion) or smaller
    than PUSH_PULL_NOISE_FLOOR_M (treated as noise) are ignored.
    """

    def __init__(self, n_fencers=2, smooth_window=PUSH_PULL_SMOOTH_WINDOW):
        self.smooth_window = smooth_window
        self.x_history     = [deque(maxlen=smooth_window) for _ in range(n_fencers)]
        self.prev_smooth   = [None] * n_fencers
        self.advance_m     = [0.0]  * n_fencers
        self.retreat_m     = [0.0]  * n_fencers

    def update(self, idx, fencer_x, opponent_x, scale_px_per_m):
        if fencer_x is None or scale_px_per_m == 0:
            return

        self.x_history[idx].append(fencer_x)
        smooth_x = float(np.median(self.x_history[idx]))

        prev = self.prev_smooth[idx]
        self.prev_smooth[idx] = smooth_x
        if prev is None:
            return

        dx_px = smooth_x - prev

        # signed advance: positive = moving toward opponent
        if opponent_x is not None and opponent_x < prev:
            advance_px = -dx_px
        else:
            advance_px = dx_px

        advance_m = advance_px / scale_px_per_m

        if abs(advance_m) > MAX_FRAME_MOVEMENT_M:
            return
        if abs(advance_m) < PUSH_PULL_NOISE_FLOOR_M:
            return

        if advance_m > 0:
            self.advance_m[idx] += advance_m
        else:
            self.retreat_m[idx] += -advance_m


# --- smoothing --------------------------------------------------------

def smooth_distance(buffer, new_value, window=SMOOTH_WINDOW):
    """
    Append new_value to a sliding buffer (mutated in place) and return
    the rolling median of the most recent `window` values.
    """
    buffer.append(new_value)
    if len(buffer) > window:
        buffer.popleft()
    return float(np.median(buffer))


# --- drawing ----------------------------------------------------------

def _draw_panel(frame, x, y, w, h, alpha=0.55):
    """Draw a semi-transparent dark panel for overlay text legibility."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # thin outline so the panel reads cleanly against any background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)


def draw_overlay(frame, slots, pose_data, dist_display_m, dist_raw_m,
                 dist_method, push_pull, frame_idx, fps):
    """Draw bounding boxes, pose keypoints, distance overlay, and push/pull stats."""

    h, w = frame.shape[:2]

    # bounding boxes + pose dots
    for slot_idx, slot in enumerate(slots):
        if slot is None:
            continue
        box, track_id = slot
        x1, y1, x2, y2 = [int(v) for v in box]
        colour = COLOURS[slot_idx]
        label  = f"Fencer {slot_idx + 1}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        lms = pose_data[slot_idx] if slot_idx < len(pose_data) else None
        if lms:
            for lm_idx in (LM_LEFT_ANKLE, LM_RIGHT_ANKLE,
                           LM_LEFT_HIP, LM_RIGHT_HIP,
                           LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER):
                if lm_idx in lms:
                    px, py = int(lms[lm_idx][0]), int(lms[lm_idx][1])
                    cv2.circle(frame, (px, py), 4, colour, -1)

    # ----- HUD panel along the bottom of the frame -----
    panel_h     = 116
    panel_y     = h - panel_h - 16
    panel_w     = w - 32
    panel_x     = 16
    _draw_panel(frame, panel_x, panel_y, panel_w, panel_h, alpha=0.6)

    inner_top = panel_y + 12
    col_w     = panel_w // 3

    # column 1: time + distance
    time_text = f"Time: {frame_idx / fps:5.1f} s"
    if dist_display_m is not None:
        method_label = "pose" if dist_method == "pose" else "bbox"
        dist_text    = f"{dist_display_m:.2f} m  ({method_label})"
    else:
        dist_text    = "-- m"
    zone_colour = distance_zone_colour(dist_display_m)

    col1_x = panel_x + 16
    cv2.putText(frame, time_text, (col1_x, inner_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
    cv2.putText(frame, "Distance:", (col1_x, inner_top + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, dist_text, (col1_x, inner_top + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, zone_colour, 2)

    # columns 2 & 3: per-fencer push / pull
    for i in range(2):
        col_x = panel_x + 16 + col_w * (i + 1)
        adv = push_pull.advance_m[i]
        ret = push_pull.retreat_m[i]

        cv2.putText(frame, f"Fencer {i + 1}", (col_x, inner_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[i], 2)
        cv2.putText(frame, f"push  {adv:6.2f} m", (col_x, inner_top + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, f"pull  {ret:6.2f} m", (col_x, inner_top + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

    return frame


def save_plot(times, distances, methods, output_path):
    """Save a distance-over-time chart, with pose vs bbox samples coloured separately."""
    pose_t = [t for t, m in zip(times, methods) if m == "pose"]
    pose_d = [d for d, m in zip(distances, methods) if m == "pose"]
    bbox_t = [t for t, m in zip(times, methods) if m == "bbox"]
    bbox_d = [d for d, m in zip(distances, methods) if m == "bbox"]

    fig, ax = plt.subplots(figsize=(12, 4))
    if pose_t:
        ax.scatter(pose_t, pose_d, s=2, color="#2196F3", label="Pose (front foot)", alpha=0.7)
    if bbox_t:
        ax.scatter(bbox_t, bbox_d, s=2, color="#FF9800", label="Fallback (bbox feet)", alpha=0.5)
    # also draw zone thresholds for context
    ax.axhline(DIST_CLOSE_M,  color="red",    linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(DIST_MEDIUM_M, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Estimated distance (metres)")
    ax.set_title("Inter-fencer Distance Over Time (front foot to front foot)")
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {output_path}")


# --- main pipeline ----------------------------------------------------

def run(video_path, output_dir, pose_stride=DEFAULT_POSE_STRIDE, piste_config=None,
        show_piste=False):
    os.makedirs(output_dir, exist_ok=True)

    base      = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base}_annotated.mp4")
    out_csv   = os.path.join(output_dir, f"{base}_distance.csv")
    out_plot  = os.path.join(output_dir, f"{base}_distance_plot.png")

    piste = PisteRegion.from_json_file(piste_config)
    if piste is not None:
        print(f"Piste region loaded from: {piste_config}")

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

    fencer_tracker = FencerTracker()
    push_pull      = PushPullTracker(n_fencers=2)
    dist_buffer    = deque()

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
            frame, persist=True,
            conf=CONF_THRESH, iou=IOU_THRESH,
            classes=[0], verbose=False,
        )

        # gather detections this frame
        ids   = np.array([], dtype=int)
        xyxys = np.empty((0, 4))
        confs = np.array([])
        if results[0].boxes is not None and results[0].boxes.id is not None:
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            xyxys = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

        # drop detections outside the piste region (referees, adjacent
        # pistes, audience). No-op when no piste config was provided.
        if piste is not None:
            ids, xyxys, confs = piste.filter_detections(ids, xyxys, confs)

        # map detections to stable Fencer 1 / Fencer 2 slots
        slots = fencer_tracker.select(ids, xyxys, confs)

        # run pose on each slot, but only every `pose_stride` frames
        # to keep total wall-clock time manageable on long videos
        pose_data = [None, None]
        run_pose_this_frame = (frame_idx % pose_stride == 0)
        if run_pose_this_frame:
            for i, slot in enumerate(slots):
                if slot is not None:
                    pose_data[i] = get_pose_landmarks(frame, slot[0])

        # compute distance + update push/pull when both fencers visible
        dist_raw_m  = None
        dist_method = None
        if slots[0] is not None and slots[1] is not None:
            box0, _ = slots[0]
            box1, _ = slots[1]

            h0 = box_height_pixels(box0)
            h1 = box_height_pixels(box1)
            avg_height_px   = (h0 + h1) / 2
            scale_px_per_m  = avg_height_px / REAL_HEIGHT_M if avg_height_px > 0 else 0

            # per-fencer reference x: prefer mid-hip from pose, else box bottom-centre
            ref0 = get_hip_centre(pose_data[0]) if pose_data[0] else None
            ref1 = get_hip_centre(pose_data[1]) if pose_data[1] else None
            if ref0 is None:
                ref0 = get_box_bottom_centre(box0)
            if ref1 is None:
                ref1 = get_box_bottom_centre(box1)

            # front-foot distance if pose available on both; else feet-of-bbox
            front0 = get_front_foot(pose_data[0], ref0[0], ref1[0]) if pose_data[0] else None
            front1 = get_front_foot(pose_data[1], ref1[0], ref0[0]) if pose_data[1] else None

            if front0 and front1:
                dist_px     = pixel_distance(front0, front1)
                dist_raw_m  = normalise_distance(dist_px, avg_height_px)
                dist_method = "pose"
                pose_success += 1
            else:
                # fallback: bbox bottom-centres (feet proxy) - more accurate than centre
                p0 = get_box_bottom_centre(box0)
                p1 = get_box_bottom_centre(box1)
                dist_px     = pixel_distance(p0, p1)
                dist_raw_m  = normalise_distance(dist_px, avg_height_px)
                dist_method = "bbox"

            # update push/pull using each fencer's reference x and the scale
            push_pull.update(0, ref0[0], ref1[0], scale_px_per_m)
            push_pull.update(1, ref1[0], ref0[0], scale_px_per_m)

        # smoothed value for the on-screen overlay
        dist_display_m = None
        if dist_raw_m is not None:
            dist_display_m = smooth_distance(dist_buffer, dist_raw_m)

        # outline the piste before the boxes, so boxes draw on top of it
        if piste is not None and show_piste:
            piste.draw(frame)

        frame = draw_overlay(frame, slots, pose_data,
                             dist_display_m, dist_raw_m, dist_method,
                             push_pull, frame_idx, fps)
        writer.write(frame)

        time_sec = frame_idx / fps
        csv_rows.append({
            "frame":             frame_idx,
            "time_s":            round(time_sec, 3),
            "distance_raw_m":    round(dist_raw_m, 3)     if dist_raw_m     is not None else "",
            "distance_smooth_m": round(dist_display_m, 3) if dist_display_m is not None else "",
            "method":            dist_method or "",
            "f1_advance_m":      round(push_pull.advance_m[0], 3),
            "f1_retreat_m":      round(push_pull.retreat_m[0], 3),
            "f2_advance_m":      round(push_pull.advance_m[1], 3),
            "f2_retreat_m":      round(push_pull.retreat_m[1], 3),
        })
        if dist_raw_m is not None:
            times.append(time_sec)
            distances.append(dist_raw_m)
            methods.append(dist_method)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} frames processed")

    cap.release()
    writer.release()
    print(f"  Annotated video saved -> {out_video}")

    with open(out_csv, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=[
            "frame", "time_s",
            "distance_raw_m", "distance_smooth_m", "method",
            "f1_advance_m", "f1_retreat_m",
            "f2_advance_m", "f2_retreat_m",
        ])
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)
    print(f"  CSV saved -> {out_csv}")

    if distances:
        save_plot(times, distances, methods, out_plot)
        pose_pct = (pose_success / len(distances)) * 100
        print("\n--- Summary ---")
        print(f"  Frames processed:           {frame_idx}")
        print(f"  Frames with both fencers:   {len(distances)}")
        print(f"  Pose-based distance:        {pose_success} ({pose_pct:.1f}%)")
        print(f"  Fallback (bbox) distance:   {len(distances) - pose_success}")
        print(f"  Mean distance:              {np.mean(distances):.2f} m")
        print(f"  Min  distance:              {np.min(distances):.2f} m")
        print(f"  Max  distance:              {np.max(distances):.2f} m")
        print(f"  Std deviation:              {np.std(distances):.2f} m")
        print(f"  Fencer 1   total advance:   {push_pull.advance_m[0]:.2f} m")
        print(f"  Fencer 1   total retreat:   {push_pull.retreat_m[0]:.2f} m")
        print(f"  Fencer 2   total advance:   {push_pull.advance_m[1]:.2f} m")
        print(f"  Fencer 2   total retreat:   {push_pull.retreat_m[1]:.2f} m")
    else:
        print("  No distance data recorded.")


def main():
    parser = argparse.ArgumentParser(description="Fencing bout fencer detection and distance analysis")
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument("--output", default="results", help="Output folder (default: results/)")
    parser.add_argument("--pose-stride", type=int, default=DEFAULT_POSE_STRIDE,
                        help=f"Run pose estimation every Nth frame (default {DEFAULT_POSE_STRIDE}, "
                             f"set to 1 for every frame)")
    parser.add_argument("--piste-config", default=None,
                        help="Path to a JSON file describing the piste polygon "
                             "(pixel-space vertices). Detections outside the "
                             "polygon are rejected before tracking.")
    parser.add_argument("--show-piste", action="store_true",
                        help="Draw the piste polygon on the annotated video. "
                             "Diagnostic only; useful for checking that a "
                             "polygon actually matches the strip.")
    args = parser.parse_args()
    run(args.video, args.output, pose_stride=args.pose_stride,
        piste_config=args.piste_config, show_piste=args.show_piste)


if __name__ == "__main__":
    main()
