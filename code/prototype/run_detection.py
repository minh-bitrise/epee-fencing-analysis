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

# Tactical distance bands, in metres, measured front foot to front foot.
#
# These follow the standard fencing taxonomy (close / lunge / advance-lunge /
# out of distance) and are derived from weapon geometry rather than chosen by
# eye. An epee blade is 90 cm and arm extension adds roughly 60 cm from the
# shoulder, so reach from the front foot is about 1.2 m standing and about
# 2.3 m through a lunge, which advances the front foot a further 0.6 to 0.9 m.
# Subtracting the offset between the defender's front foot and their torso
# puts a lunge-scored touch at roughly 2.0 to 2.6 m of front-foot separation.
#
# NOTE: an earlier version used 1.0 m for "touch range" and 1.8 m for
# "engagement range". Both were invented rather than derived and were far too
# tight: they placed almost every real touch outside "touch range" entirely,
# which made the generated summaries report that close-range fencing was
# essentially absent when it was simply mis-binned. See TODO B1a.
DIST_CLOSE_M         = 1.5   # infighting; a touch lands without a lunge
DIST_LUNGE_M         = 2.6   # lunge distance; a touch can land with a lunge
DIST_ADVANCE_LUNGE_M = 3.5   # needs a step plus a lunge to reach

COLOUR_CLOSE         = (0,   0,   255)    # red     (infighting)
COLOUR_LUNGE         = (0,   165, 255)    # orange  (touch possible)
COLOUR_ADVANCE_LUNGE = (0,   255, 255)    # yellow  (needs a step first)
COLOUR_OUT           = (0,   220, 100)    # green   (out of distance)

# max plausible per-frame fencer motion in metres
# (at 60fps, > 0.15 m/frame = > 9 m/s which exceeds the fastest lunges)
# anything beyond this is treated as camera motion or detection jitter
MAX_FRAME_MOVEMENT_M = 0.15

# minimum movement (m) to count toward push/pull (ignore detection jitter)
PUSH_PULL_NOISE_FLOOR_M = 0.03
# Below this magnitude a net displacement says only "finished where they started",
# and its sign is not meaningful. Kept in step with the same constant in
# generate_summary.py and the review interface.
NET_SIGN_FLOOR_M = 1.0

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
LM_LEFT_KNEE      = 25
LM_RIGHT_KNEE     = 26
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


def distance_zone(dist_m):
    """
    Name the tactical band a distance falls into, front foot to front foot.
    Returns one of "close", "lunge", "advance_lunge", "out", or None.
    """
    if dist_m is None:
        return None
    if dist_m <= DIST_CLOSE_M:
        return "close"
    if dist_m <= DIST_LUNGE_M:
        return "lunge"
    if dist_m <= DIST_ADVANCE_LUNGE_M:
        return "advance_lunge"
    return "out"


def distance_zone_colour(dist_m):
    """Pick an overlay colour based on the tactical range of the current distance."""
    return {
        None:            (200, 200, 200),
        "close":         COLOUR_CLOSE,
        "lunge":         COLOUR_LUNGE,
        "advance_lunge": COLOUR_ADVANCE_LUNGE,
        "out":           COLOUR_OUT,
    }[distance_zone(dist_m)]


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


def get_stance_features(landmarks, scale_px_per_m):
    """
    Stance geometry for one fencer, in metres, or None where unavailable.

    WHY THIS EXISTS. The resolution ablation in TODO B1c found that pose success
    can fall from 27 per cent of frames to 5.7 per cent with no effect at all on
    touch-detection F1, which means MediaPipe is currently not load-bearing for
    anything the system reports. Distance uses the hip midpoint but falls back to
    the bounding box, so pose only refines a number it does not determine. Either
    pose earns a job or it should be dropped, and a lunge is the obvious job: it is
    the action that produces most epee touches and it is defined by stance rather
    than by position.

    Two features, both chosen to be readable off a noisy skeleton rather than to
    be precise:

    - `stance_m`, the horizontal separation of the ankles. A fencer on guard keeps
      the feet roughly shoulder width apart and a lunge throws the front foot out,
      so this should roughly double. Horizontal only, because a lunge extends along
      the piste and the piste runs across the frame.
    - `hip_height_m`, the hip midpoint above the ankle line. A lunge drops the hips
      as the rear leg extends, so this should fall while `stance_m` rises. Having
      two features that move in opposite directions matters: it discriminates a
      lunge from a fencer simply being detected at a different scale, which would
      move both the same way.

    Returns a dict so callers can record what was available rather than having to
    treat a partial skeleton as a total failure.
    """
    if not landmarks or not scale_px_per_m:
        return None

    la = landmarks.get(LM_LEFT_ANKLE)
    ra = landmarks.get(LM_RIGHT_ANKLE)
    hip = get_hip_centre(landmarks)

    stance_m = None
    if la is not None and ra is not None:
        stance_m = abs(la[0] - ra[0]) / scale_px_per_m

    hip_height_m = None
    if hip is not None and (la is not None or ra is not None):
        ankles_y = [a[1] for a in (la, ra) if a is not None]
        # image y grows downward, so the ankles sit at a LARGER y than the hips
        hip_height_m = (sum(ankles_y) / len(ankles_y) - hip[1]) / scale_px_per_m

    if stance_m is None and hip_height_m is None:
        return None
    return {"stance_m": stance_m, "hip_height_m": hip_height_m}


def _fmt_stance(features, key):
    """CSV cell for one stance feature: rounded metres, or empty when absent."""
    if not features or features.get(key) is None:
        return ""
    return round(features[key], 4)


# --- scale calibration ------------------------------------------------

def calibrate_fixed_scale(video_path, sample_every=15, max_frames=4500,
                          min_height_px=60):
    """
    Estimate one pixels-per-metre scale for the whole clip, from the median
    apparent height of the two largest person detections.

    WHY A FIXED SCALE RATHER THAN A PER-FRAME ONE. The pipeline originally
    derived scale from the current frame's mean bounding-box height, on the
    reasoning that a fencer's height is a known quantity and apparent height
    therefore encodes depth. Measurement shows that reasoning is wrong, and
    wrong in a way that biases the result rather than merely adding noise.

    Bounding-box height tracks posture more strongly than depth. On clip 3 its
    correlation with feet-y, the ground-plane depth cue, is +0.267, while its
    correlation with inter-fencer distance is +0.502; fencers are 1.31 times
    taller when in the furthest quartile of separation than in the nearest.
    That is the sport itself: en garde is shorter than standing and a lunge
    shorter again.

    The consequence is a self-reinforcing error. Closing distance to attack
    lowers both fencers, which shrinks the scale reference, which inflates every
    computed metre value, and it does so exactly during the exchanges that matter
    most. Because both fencers crouch together, the bias does not cancel between
    them, which is why clip 3 reported both fencers net-advancing a combined 32 m
    on a 14 m piste. Camera panning was investigated first and ruled out: pan
    bias moves the two fencers' net displacement in opposite directions, and
    clip 3's were both positive.

    A median over the whole clip removes the frame-to-frame posture variation.
    It does not make the absolute scale correct, since it still rests on an
    assumed 1.75 m fencer and takes no account of perspective. Doing better
    requires a scale reference that is not a fencer, which means calibrating
    against the piste. Note that this cannot assume the whole piste is visible:
    in practice a camera shows only a segment of the strip, so a four-corner
    homography is not generally available. The strip's two long edges plus one
    transverse line of known separation would be, and that is the route to a
    properly metric calibration.

    Returns pixels per metre, or None if too few usable detections were found,
    in which case callers should fall back to the per-frame estimate.
    """
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    heights = []
    idx = 0
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                    classes=[0], verbose=False)
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                tallest = sorted(boxes, key=box_height_pixels, reverse=True)[:MAX_FENCERS]
                heights.extend(box_height_pixels(b) for b in tallest
                               if box_height_pixels(b) >= min_height_px)
        idx += 1
    cap.release()
    if len(heights) < 20:
        return None
    return float(np.median(heights)) / REAL_HEIGHT_M


# --- camera motion ----------------------------------------------------

class CameraMotionEstimator:
    """
    Estimates per-frame horizontal camera motion so it can be removed from the
    fencers' apparent movement.

    WHY THIS IS NEEDED. Push and pull are accumulated from each fencer's
    horizontal displacement in image coordinates, which conflates the fencer
    moving with the camera moving. The existing safeguards do not catch this: the
    per-frame clamp only rejects jumps too large to be biomechanical, and the
    noise floor only rejects movements too small to matter, so a slow steady pan
    passes straight through and is accumulated as fencer motion.

    The consequence is measurable and it is not subtle. On hand-held footage the
    metric reported both fencers net-advancing a combined 32 m on a 14 m piste,
    which is impossible, while the distance record stayed flat throughout.
    Measured pan across the evaluation clips ranges from 20 px in total on a
    locked-off broadcast to 3,900 px with 1,070 px of net drift on a hand-held
    club recording.

    HOW IT WORKS. Good features are tracked between consecutive frames with
    Lucas-Kanade optical flow, and the median horizontal displacement is taken as
    the camera's motion. The median matters: the fencers also move, but they
    occupy a small minority of tracked features, so a median is robust to them
    where a mean would not be. Features falling inside a tracked fencer's
    bounding box are excluded as well, which removes the bias directly rather
    than relying on robustness alone.

    WHY THIS VALIDATES ITSELF. Over a whole bout each fencer returns roughly to
    where they started, since play resets to the guard lines after every touch.
    Net displacement should therefore be near zero, and that physical constraint
    gives a correctness check requiring no ground-truth labels at all.

    WHY IT IS OFF BY DEFAULT. That check says this helps three of the four
    evaluation clips slightly and harms the fourth badly. A controlled comparison
    isolated it as the cause: on the club clip the worst net displacement is 21.88 m
    with everything off, 21.90 m with fixed scale and movement banking enabled but
    stabilisation off, and 37.34 m with stabilisation on.

    The reason is that the club camera is hand-held and FOLLOWS the action. When an
    operator pans to keep the fencers in frame, camera motion becomes correlated
    with fencer motion, so subtracting it subtracts the very displacement being
    measured. Stabilisation is therefore valid only for a camera that is
    essentially fixed and pans incidentally, which describes the three broadcast
    and competition clips and not the club one.

    This matters beyond a default. A following camera has no fixed relationship to
    the piste, so no amount of frame-to-frame compensation recovers world
    coordinates from it. Measuring displacement on such footage requires a
    reference in the scene rather than in the camera, which is the argument for
    calibrating against the piste itself.
    """

    # Lucas-Kanade needs enough features to make a median meaningful; below this
    # the estimate is discarded rather than trusted.
    MIN_FEATURES = 12

    # A camera cannot pan faster than this between frames in practice, and a
    # larger apparent shift means the flow estimate has failed (a cut, a flash,
    # or near-total occlusion). Expressed as a fraction of frame width.
    MAX_PAN_FRACTION = 0.08

    FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=10)
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self):
        self.prev_gray = None
        self.cumulative_dx = 0.0     # running camera offset in pixels
        self.frames_estimated = 0
        self.frames_failed = 0

    @staticmethod
    def _mask_excluding(shape, boxes, pad=20):
        """Feature-search mask with tracked fencers blanked out."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        for box in boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = shape[:2]
            cv2.rectangle(mask,
                          (max(0, x1 - pad), max(0, y1 - pad)),
                          (min(w, x2 + pad), min(h, y2 + pad)),
                          0, -1)
        return mask

    def update(self, frame_bgr, fencer_boxes=()):
        """
        Return the estimated horizontal camera displacement since the previous
        frame, in pixels, or 0.0 when no reliable estimate is available. Also
        maintains a cumulative offset for converting image x to a stabilised x.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        mask = self._mask_excluding(frame_bgr.shape, fencer_boxes)
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.FEATURE_PARAMS)
        dx = 0.0
        if p0 is not None and len(p0) >= self.MIN_FEATURES:
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None, **self.LK_PARAMS)
            if p1 is not None and status is not None:
                good0 = p0[status.ravel() == 1].reshape(-1, 2)
                good1 = p1[status.ravel() == 1].reshape(-1, 2)
                if len(good0) >= self.MIN_FEATURES:
                    candidate = float(np.median(good1[:, 0] - good0[:, 0]))
                    limit = self.MAX_PAN_FRACTION * frame_bgr.shape[1]
                    if abs(candidate) <= limit:
                        dx = candidate
                        self.frames_estimated += 1
                    else:
                        self.frames_failed += 1
                else:
                    self.frames_failed += 1
            else:
                self.frames_failed += 1
        else:
            self.frames_failed += 1

        self.prev_gray = gray
        self.cumulative_dx += dx
        return dx

    def stabilise(self, x):
        """Convert an image x coordinate into the stabilised reference frame."""
        return None if x is None else x - self.cumulative_dx


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
        # Sub-threshold movement waiting to be committed. See update().
        self.pending_m     = [0.0]  * n_fencers
        # First and last smoothed position per fencer, in pixels, from which NET
        # displacement is computed at the end of the bout.
        #
        # WHY NET IS REPORTED SEPARATELY FROM THE CUMULATIVE TOTALS. They differ
        # in how much the data supports them, and the difference is large enough
        # that presenting them together without comment would be misleading.
        #
        # Net displacement is a difference between two positions, so a bad frame
        # in the middle affects it only if it is the first or last. Measured on
        # the club clip it agrees with the sum of per-frame deltas to the
        # centimetre, which is the arithmetic identity a correct accumulator must
        # satisfy.
        #
        # Cumulative push and pull sum the magnitude of every frame's movement, so
        # every tracking error adds to them and none cancels. On the club clip 63
        # to 65 frames out of 5,110 carry apparent jumps averaging half a metre in
        # a single frame, which is 15 m/s and not a fencer moving; they are
        # tracker discontinuities, permitted because the identity gate allows a
        # candidate to move up to 3.5 bounding-box heights between frames. How
        # those frames are handled changes the answer by tens of metres on a 14 m
        # piste: discarding them biases the net by about 10 m and 23 m for the two
        # fencers, capping them by about the same, and counting them inflates the
        # path length outright. Three defensible treatments disagreeing by that
        # much means the cumulative total is not determined by the data.
        #
        # So net displacement is reported as a measurement and the cumulative
        # totals as indicative only. Recovering a trustworthy path length needs
        # tracking without metre-scale discontinuities, which is a tracking
        # problem rather than an accumulation one.
        self.first_smooth = [None] * n_fencers
        self.last_smooth  = [None] * n_fencers

        # Counters for the share of moving frames spent closing distance.
        #
        # This is the well-defined replacement for "how far did each fencer
        # advance in total". That question, as a distance, has no answer in this
        # data: path length sums the magnitude of every frame's change, so
        # measurement noise adds to it and never cancels. Re-measuring clip 3's
        # position series under median windows from 1 to 121 frames moved the path
        # length from 161 m to 33 m for one fencer, a factor of five, with no
        # asymptote, while net displacement stayed at exactly +3.43 m throughout.
        # A quantity that changes fivefold with an arbitrary smoothing parameter
        # is not a measurement of the fencer; it is a measurement of the filter.
        #
        # Counting the SIGN of each frame's movement rather than its magnitude
        # avoids that, because noise contributes symmetrically to both directions.
        # Under the same 1-to-121 window sweep the closing share moved only from
        # 52.4 to 60.9 per cent, so it is mildly window-dependent rather than
        # scale-free, and the window should be reported alongside it.
        self.closing_frames = [0] * n_fencers
        self.moving_frames  = [0] * n_fencers

        # Which image direction counts as "toward the opponent" for each fencer,
        # +1 for rightward and -1 for leftward. Established once from the first
        # frame in which both fencers are located, and then held for the bout.
        #
        # WHY THIS IS FIXED RATHER THAN RE-EVALUATED PER FRAME. The original
        # implementation compared the opponent's current x against this fencer's
        # previous x on every frame, which sounds harmless because fencers do not
        # change ends during a bout. Measured on the club clip, that comparison
        # returned the wrong side on 3 frames out of 5,109, a rate of 0.06 per
        # cent. Those three frames cost 10 metres.
        #
        # The reason the damage is so disproportionate is that the comparison
        # only fails when a tracked position jumps, and a jump is exactly when
        # the frame's displacement is large. A large movement given the wrong
        # sign contributes twice its magnitude as error, once for the value it
        # should have had and once for the value it got. So the metric was most
        # vulnerable at precisely the moments the rest of the pipeline is built
        # to tolerate.
        #
        # With the side fixed, the accumulated net displacement matches the
        # difference between the first and last tracked position exactly, which
        # is the arithmetic identity any correct accumulator must satisfy. On the
        # club clip that changed Fencer 2's net from +9.18 m to -0.88 m, the
        # latter agreeing with the endpoint measurement to the centimetre.
        self.toward_opponent = [None] * n_fencers

    def update(self, idx, fencer_x, opponent_x, scale_px_per_m):
        if fencer_x is None or scale_px_per_m == 0:
            return

        # Establish which way is "forward" for this fencer, once. See the note in
        # __init__ for why this must not be re-evaluated per frame.
        if self.toward_opponent[idx] is None and opponent_x is not None:
            self.toward_opponent[idx] = 1.0 if opponent_x > fencer_x else -1.0

        self.x_history[idx].append(fencer_x)
        smooth_x = float(np.median(self.x_history[idx]))
        if self.first_smooth[idx] is None:
            self.first_smooth[idx] = smooth_x
        self.last_smooth[idx] = smooth_x

        prev = self.prev_smooth[idx]
        self.prev_smooth[idx] = smooth_x
        if prev is None:
            return

        dx_px = smooth_x - prev

        # signed advance: positive = moving toward the opponent
        direction = self.toward_opponent[idx]
        if direction is None:
            return          # side not yet known, so the sign is undetermined
        advance_m = (dx_px * direction) / scale_px_per_m

        # A single-frame movement this large is not biomechanically possible and
        # indicates camera motion or a detection failure. It is CAPPED rather
        # than discarded.
        #
        # Discarding it was the earlier behaviour and it biased the result for the
        # same reason the noise floor did: what gets removed is not
        # direction-neutral. On the club clip the cap threshold was exceeded on
        # only 52 and 55 frames out of 5,110, about one per cent, but those frames
        # carried a net of -9.25 m and -12.71 m, almost entirely retreat. Removing
        # one per cent of frames therefore injected roughly ten metres of false
        # advance.
        #
        # Capping keeps the sign and a plausible magnitude, which bounds the
        # influence of a glitch without deleting the movement underneath it.
        #
        # It does NOT preserve the identity that accumulated net displacement
        # equals the difference between first and last position. An earlier
        # version of this comment claimed it did, and that was wrong. Capping
        # discards everything above the threshold, so it breaks the identity for
        # the same reason trimming does, just less. Measured by replaying clip 3's
        # recorded positions through this exact logic, the cap fires on 63 and 65
        # frames of 5,246 and destroys -9.26 m and -23.94 m of signed movement,
        # which is the ENTIRE divergence between the cumulative totals and the
        # endpoint measurement (F2: accumulated +23.01 m against an endpoint
        # -0.94 m). The residue left unbanked is 0.015 m, so nothing else
        # contributes.
        #
        # No cap value repairs this. The movements being capped are tracking
        # glitches of up to 5.4 m in a single frame, so the true displacement
        # underneath them is not recoverable from the series at all; a larger cap
        # admits the glitch and a smaller one destroys more real retreat. That is
        # why the cumulative totals were abandoned as measurements rather than
        # retuned, and why the metrics that survive are the ones immune to this by
        # construction: net displacement, which reads only the endpoints, and
        # closing share, which counts signs.
        if abs(advance_m) > MAX_FRAME_MOVEMENT_M:
            advance_m = MAX_FRAME_MOVEMENT_M if advance_m > 0 else -MAX_FRAME_MOVEMENT_M

        # Sub-threshold movement is BANKED, not discarded.
        #
        # The threshold exists to stop bounding-box jitter accumulating: an early
        # version summed raw displacement and reported 216 m of push per fencer in
        # a three-minute bout. Discarding small movements fixed that number and
        # introduced a directional bias, because in fencing advances and retreats
        # do not have the same speed. An attack is explosive and clears the
        # threshold every frame; the recovery and walk-back are slow and clear it
        # on none. Measured on synthetic input with a fencer returning to its
        # exact starting position, the discarding version reported +5.25 m of net
        # advance with pull recorded as 0.00 m, having thrown away every retreat
        # frame. Because both fencers attack fast and recover slowly, the bias
        # does not cancel between them, which is why clip 3 reported both fencers
        # net-advancing a combined 32 m on a 14 m piste.
        #
        # Banking keeps the jitter rejection and removes the bias. Jitter
        # oscillates around zero, so the buffer rarely reaches the threshold and
        # commits only the true net when it does. Genuine slow movement is
        # one-directional, so the buffer fills and commits at the correct
        # magnitude. Nothing real is lost; it is only delayed.
        #
        # Camera panning and per-frame scale variation were both investigated as
        # causes of the same symptom before this was found, and neither was it.
        # Both "fixes" made clip 3 worse, because each reduced the noise that had
        # been accidentally pushing some slow retreats over the threshold.
        self.pending_m[idx] += advance_m
        if abs(self.pending_m[idx]) < PUSH_PULL_NOISE_FLOOR_M:
            return

        committed = self.pending_m[idx]
        self.pending_m[idx] = 0.0

        self.moving_frames[idx] += 1
        if committed > 0:
            self.advance_m[idx] += committed
            self.closing_frames[idx] += 1
        else:
            self.retreat_m[idx] += -committed

    def closing_share(self, idx):
        """
        Share of committed movements that were toward the opponent, in [0, 1].

        Answers "who pressed forward more" without depending on distance
        magnitudes, which is what makes it usable where the push and pull totals
        are not. Returns None before any movement has been committed.

        The magnitude-independence holds for movements that clear
        PUSH_PULL_NOISE_FLOOR_M. Below it the banking buffer decides when a
        movement commits, so magnitude does influence the count. The metric is
        therefore scale-insensitive in the regime that matters rather than
        universally, and it should be quoted with the smoothing window, since a
        1-to-121 frame sweep moved it from 52.4 to 60.9 per cent on clip 3.
        """
        if self.moving_frames[idx] == 0:
            return None
        return self.closing_frames[idx] / self.moving_frames[idx]

    def net_displacement_m(self, idx, scale_px_per_m):
        """
        Net movement toward the opponent, from first to last tracked position.

        This is the reliable movement figure. Unlike the cumulative push and pull
        totals it is a difference between two positions rather than a sum over
        every frame, so a tracking discontinuity in the middle of the bout does
        not accumulate into it. See the note in __init__.
        """
        if (self.first_smooth[idx] is None or self.last_smooth[idx] is None
                or self.toward_opponent[idx] is None or not scale_px_per_m):
            return None
        delta_px = (self.last_smooth[idx] - self.first_smooth[idx])
        return (delta_px * self.toward_opponent[idx]) / scale_px_per_m


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
                 dist_method, push_pull, frame_idx, fps, net_scale=None):
    """
    Draw bounding boxes, pose keypoints, the distance readout and per-fencer
    movement.

    The movement columns show net displacement and closing share, not the
    cumulative push and pull totals they used to show. The totals are wrong rather
    than approximate: B1g traced their error to the per-frame movement cap and
    measured it at 24 m on a 14 m piste. Burning them into the video was the last
    place they still appeared as though they were measurements, and an annotated
    clip is the most quotable artefact the project produces, so it should not
    caption a fencer with a figure the project has withdrawn.

    net_scale is the clip-wide fixed scale in pixels per metre. It defaults to None
    so a caller without one still gets an overlay, showing the closing share and
    marking the displacement unavailable rather than inventing it.
    """

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

    # columns 2 & 3: per-fencer net displacement and closing share
    for i in range(2):
        col_x = panel_x + 16 + col_w * (i + 1)

        net = push_pull.net_displacement_m(i, net_scale) if net_scale else None
        # Under a metre the sign is not meaningful: the same fencer measures
        # +0.43 m from raw positions and -0.94 m from smoothed endpoints, so the
        # two readings agree on the substance and disagree on the direction.
        if net is None:
            net_text = "net      -- "
        elif abs(net) < NET_SIGN_FLOOR_M:
            net_text = "net    ~0 m"
        else:
            net_text = f"net  {net:+5.2f} m"

        share = push_pull.closing_share(i)
        share_text = "closing   -- " if share is None else f"closing {share*100:4.1f}%"

        cv2.putText(frame, f"Fencer {i + 1}", (col_x, inner_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[i], 2)
        cv2.putText(frame, net_text, (col_x, inner_top + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, share_text, (col_x, inner_top + 92),
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
    # band boundaries, drawn for context (see the constants for the derivation)
    ax.axhline(DIST_CLOSE_M,         color="red",    linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(DIST_LUNGE_M,         color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(DIST_ADVANCE_LUNGE_M, color="gold",   linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Estimated distance (metres)")
    ax.set_title("Inter-fencer Distance Over Time (front foot to front foot)")
    ax.legend(markerscale=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {output_path}")


def _fmt_net(v):
    """Format a net displacement, or say why it is unavailable."""
    return f"{v:+.2f} m" if v is not None else "unavailable (no fixed scale)"


# --- main pipeline ----------------------------------------------------

def run(video_path, output_dir, pose_stride=DEFAULT_POSE_STRIDE, piste_config=None,
        show_piste=False, stabilise_camera=False, fixed_scale_calibration=True):
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
    camera         = CameraMotionEstimator() if stabilise_camera else None

    fixed_scale = None
    if fixed_scale_calibration:
        print("Calibrating a fixed pixels-per-metre scale (pre-pass)...")
        fixed_scale = calibrate_fixed_scale(video_path)
        if fixed_scale:
            print(f"  fixed scale: {fixed_scale:.1f} px/m")
        else:
            print("  too few detections; falling back to per-frame scale")
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

        # estimate camera motion with the tracked fencers masked out, so
        # their movement does not bias the global estimate
        if camera is not None:
            camera.update(frame, [s[0] for s in slots if s is not None])

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
        f1_pos_m    = None
        f2_pos_m    = None
        stance      = [None, None]
        if slots[0] is not None and slots[1] is not None:
            box0, _ = slots[0]
            box1, _ = slots[1]

            h0 = box_height_pixels(box0)
            h1 = box_height_pixels(box1)
            avg_height_px   = (h0 + h1) / 2
            scale_px_per_m  = avg_height_px / REAL_HEIGHT_M if avg_height_px > 0 else 0
            # Push/pull uses the clip-wide fixed scale when available. The
            # per-frame scale tracks posture, not depth, so it biases
            # accumulated movement toward the moments fencers crouch.
            movement_scale = fixed_scale if fixed_scale else scale_px_per_m

            # per-fencer reference x: prefer mid-hip from pose, else box bottom-centre
            ref0 = get_hip_centre(pose_data[0]) if pose_data[0] else None
            ref1 = get_hip_centre(pose_data[1]) if pose_data[1] else None

            # stance geometry, recorded rather than discarded. See
            # get_stance_features for why: pose is currently not load-bearing and
            # this is the data needed to decide whether it should be.
            stance = [get_stance_features(pose_data[i], movement_scale)
                      for i in (0, 1)]
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

            # Push/pull uses the bounding-box bottom-centre ONLY, never the hip.
            #
            # ref0 and ref1 above prefer the mid-hip when pose is available and
            # fall back to the box bottom-centre otherwise, which is right for
            # distance: the hip is a better body-position estimate and distance is
            # computed independently on each frame, so switching landmarks changes
            # accuracy but not consistency.
            #
            # Displacement is different. It is a difference between successive
            # frames, so it requires the SAME point to be tracked over time. Pose
            # runs every third frame, so the reference was alternating between two
            # landmarks that sit a measured 6 to 10 px apart, and the offset is not
            # constant: it varies with stance, being largest in a lunge, when the
            # hips stay back and the front foot travels. Every switch therefore
            # injected a spurious displacement correlated with the action.
            #
            # The box bottom-centre is available on every frame, so using it alone
            # keeps the reference consistent. This is not a preference between two
            # good options: differencing a signal whose definition changes between
            # samples measures the definition change as movement.
            pp_x0 = get_box_bottom_centre(box0)[0]
            pp_x1 = get_box_bottom_centre(box1)[0]

            # Remove camera motion before accumulating, when enabled. Both
            # fencers are shifted by the same offset, so the left/right
            # relationship and therefore the sign of "advance" are unchanged.
            if camera is not None:
                sx0 = camera.stabilise(pp_x0)
                sx1 = camera.stabilise(pp_x1)
            else:
                sx0, sx1 = pp_x0, pp_x1

            push_pull.update(0, sx0, sx1, movement_scale)
            push_pull.update(1, sx1, sx0, movement_scale)

            # Raw position of each fencer along the image x axis, in metres.
            #
            # Written to the CSV so downstream metrics are derived from the
            # measurement rather than from the filtered totals. Net displacement
            # and closing share computed from these columns are independent of the
            # noise floor, the movement cap and the banking buffer; computed from
            # the cumulative advance/retreat columns they inherit all three. That
            # distinction is not academic: the two disagreed by 3.5 m on clip 3
            # before these columns existed.
            if movement_scale:
                f1_pos_m = sx0 / movement_scale
                f2_pos_m = sx1 / movement_scale

        # smoothed value for the on-screen overlay
        dist_display_m = None
        if dist_raw_m is not None:
            dist_display_m = smooth_distance(dist_buffer, dist_raw_m)

        # outline the piste before the boxes, so boxes draw on top of it
        if piste is not None and show_piste:
            piste.draw(frame)

        frame = draw_overlay(frame, slots, pose_data,
                             dist_display_m, dist_raw_m, dist_method,
                             push_pull, frame_idx, fps,
                             net_scale=fixed_scale if fixed_scale else None)
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
            "f1_pos_m":          round(f1_pos_m, 4) if f1_pos_m is not None else "",
            "f2_pos_m":          round(f2_pos_m, 4) if f2_pos_m is not None else "",
            "f1_stance_m":       _fmt_stance(stance[0], "stance_m"),
            "f2_stance_m":       _fmt_stance(stance[1], "stance_m"),
            "f1_hip_height_m":   _fmt_stance(stance[0], "hip_height_m"),
            "f2_hip_height_m":   _fmt_stance(stance[1], "hip_height_m"),
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
            "f1_pos_m", "f2_pos_m",
            "f1_stance_m", "f2_stance_m",
            "f1_hip_height_m", "f2_hip_height_m",
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
        net_scale = fixed_scale if fixed_scale else None
        print(f"  Fencer 1   net displacement: "
              f"{_fmt_net(push_pull.net_displacement_m(0, net_scale))}")
        print(f"  Fencer 2   net displacement: "
              f"{_fmt_net(push_pull.net_displacement_m(1, net_scale))}")
        print(f"  (net is a first-to-last position difference and is the reliable"
              f" movement figure)")
        for k in (0, 1):
            cs = push_pull.closing_share(k)
            shown = f"{100*cs:.1f}%" if cs is not None else "unavailable"
            print(f"  Fencer {k+1}   closing share:    {shown}")
        print(f"  (share of moving frames spent closing distance; smoothing window"
              f" {PUSH_PULL_SMOOTH_WINDOW})")
        print(f"  Fencer 1   total advance:   {push_pull.advance_m[0]:.2f} m  [indicative]")
        print(f"  Fencer 1   total retreat:   {push_pull.retreat_m[0]:.2f} m  [indicative]")
        print(f"  Fencer 2   total advance:   {push_pull.advance_m[1]:.2f} m  [indicative]")
        print(f"  Fencer 2   total retreat:   {push_pull.retreat_m[1]:.2f} m  [indicative]")
        print(f"  (cumulative totals sum every frame, so tracking discontinuities"
              f" accumulate into them; see PushPullTracker)")
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
    parser.add_argument("--no-fixed-scale", action="store_true",
                        help="Use the per-frame bounding-box scale for push/pull instead\n"
                             "of a clip-wide fixed one. The per-frame scale tracks posture\n"
                             "rather than depth and biases movement totals; retained so the\n"
                             "before/after comparison stays reproducible.")
    parser.add_argument("--stabilise", action="store_true",
                        help="Enable camera-motion compensation. OFF by default: it is "
                             "only valid for a camera that is essentially fixed and pans "
                             "incidentally. On a hand-held camera that FOLLOWS the action, "
                             "camera motion is correlated with fencer motion, so removing "
                             "it removes the signal being measured. Measured on the club "
                             "clip, enabling it made the worst net displacement worse, "
                             "21.88 -> 37.34 m, while helping the three fixed-camera clips "
                             "only slightly.")
    parser.add_argument("--show-piste", action="store_true",
                        help="Draw the piste polygon on the annotated video. "
                             "Diagnostic only; useful for checking that a "
                             "polygon actually matches the strip.")
    args = parser.parse_args()
    run(args.video, args.output, pose_stride=args.pose_stride,
        piste_config=args.piste_config, show_piste=args.show_piste,
        stabilise_camera=args.stabilise,
        fixed_scale_calibration=not args.no_fixed_scale)


if __name__ == "__main__":
    main()
