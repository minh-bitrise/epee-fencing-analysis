"""
Unit tests for the fencing detection prototype (v3).
Tests the pure utility functions and stateful trackers without
requiring a real video or AI model.

Run with:
    python3 -m pytest test_detection.py -v
"""

import json
import math
import os
import tempfile
from collections import deque

import numpy as np
import pytest

from run_detection import (
    # geometry helpers
    get_box_centre,
    get_box_bottom_centre,
    box_height_pixels,
    pixel_distance,
    normalise_distance,
    distance_zone,
    distance_zone_colour,
    # pose helpers
    get_front_foot,
    get_hip_centre,
    LM_LEFT_ANKLE,
    LM_RIGHT_ANKLE,
    LM_LEFT_HIP,
    LM_RIGHT_HIP,
    # trackers and smoothing
    FencerTracker,
    PisteRegion,
    PushPullTracker,
    smooth_distance,
    # constants
    COLOUR_CLOSE,
    COLOUR_LUNGE,
    COLOUR_ADVANCE_LUNGE,
    COLOUR_OUT,
    DIST_CLOSE_M,
    DIST_LUNGE_M,
    DIST_ADVANCE_LUNGE_M,
    MAX_FRAME_MOVEMENT_M,
    PUSH_PULL_NOISE_FLOOR_M,
)


# -------------------- geometry --------------------

class TestGetBoxCentre:
    def test_simple_box(self):
        assert get_box_centre([0, 0, 100, 100]) == (50.0, 50.0)

    def test_non_square_box(self):
        assert get_box_centre([10, 20, 50, 80]) == (30.0, 50.0)

    def test_single_pixel_box(self):
        assert get_box_centre([5, 5, 5, 5]) == (5.0, 5.0)


class TestGetBoxBottomCentre:
    def test_returns_bottom_y(self):
        # bottom-centre y should be the bottom edge of the box
        assert get_box_bottom_centre([0, 0, 100, 200]) == (50.0, 200.0)

    def test_centre_x_unchanged(self):
        # x should be the same as the regular centre
        cx, _   = get_box_centre([10, 20, 50, 80])
        bx, by  = get_box_bottom_centre([10, 20, 50, 80])
        assert bx == cx
        assert by == 80


class TestBoxHeightPixels:
    def test_normal_box(self):
        assert box_height_pixels([0, 10, 100, 110]) == 100

    def test_zero_height(self):
        assert box_height_pixels([0, 50, 100, 50]) == 0

    def test_tall_box(self):
        assert box_height_pixels([0, 0, 50, 480]) == 480


class TestPixelDistance:
    def test_zero(self):
        assert pixel_distance((10, 10), (10, 10)) == 0.0

    def test_horizontal(self):
        assert pixel_distance((0, 0), (100, 0)) == 100.0

    def test_vertical(self):
        assert pixel_distance((0, 0), (0, 50)) == 50.0

    def test_345_triangle(self):
        assert math.isclose(pixel_distance((0, 0), (3, 4)), 5.0)

    def test_float_coords(self):
        assert math.isclose(pixel_distance((0.5, 0.5), (3.5, 4.5)), 5.0)


class TestNormaliseDistance:
    def test_equal_distance_and_height(self):
        assert math.isclose(normalise_distance(200, 200, real_height_m=1.75), 1.75)

    def test_zero_height_returns_none(self):
        assert normalise_distance(100, 0) is None

    def test_close_fencers(self):
        assert math.isclose(normalise_distance(100, 200, real_height_m=1.75), 0.875)

    def test_far_fencers(self):
        assert math.isclose(normalise_distance(400, 200, real_height_m=1.75), 3.5)

    def test_custom_real_height(self):
        assert math.isclose(normalise_distance(100, 100, real_height_m=2.0), 2.0)


class TestDistanceZone:
    """
    Bands are front foot to front foot and follow the fencing taxonomy:
    close/infighting, lunge distance (where a touch can actually land),
    advance-lunge, and out of distance.
    """

    def test_none_distance(self):
        assert distance_zone(None) is None

    def test_close_band(self):
        assert distance_zone(0.5) == "close"
        assert distance_zone(DIST_CLOSE_M) == "close"          # boundary inclusive

    def test_lunge_band(self):
        assert distance_zone(DIST_CLOSE_M + 0.01) == "lunge"
        assert distance_zone(2.2) == "lunge"
        assert distance_zone(DIST_LUNGE_M) == "lunge"          # boundary inclusive

    def test_advance_lunge_band(self):
        assert distance_zone(DIST_LUNGE_M + 0.01) == "advance_lunge"
        assert distance_zone(DIST_ADVANCE_LUNGE_M) == "advance_lunge"

    def test_out_of_distance_band(self):
        assert distance_zone(DIST_ADVANCE_LUNGE_M + 0.01) == "out"
        assert distance_zone(10.0) == "out"

    def test_a_typical_lunge_touch_is_in_the_lunge_band(self):
        """
        Regression guard for the bug this taxonomy replaced. A touch scored
        with a lunge lands at roughly 2.0-2.6 m of front-foot separation; the
        previous thresholds (1.0 / 1.8 m) binned all of those as "far", which
        made the generated summaries report close-range fencing as absent.
        """
        for d in (2.0, 2.2, 2.4, 2.6):
            assert distance_zone(d) == "lunge", f"{d} m should be a scoring distance"


class TestDistanceZoneColour:
    def test_none_returns_neutral(self):
        c = distance_zone_colour(None)
        assert isinstance(c, tuple) and len(c) == 3

    def test_each_band_has_its_colour(self):
        assert distance_zone_colour(0.5) == COLOUR_CLOSE
        assert distance_zone_colour(2.2) == COLOUR_LUNGE
        assert distance_zone_colour(3.0) == COLOUR_ADVANCE_LUNGE
        assert distance_zone_colour(5.0) == COLOUR_OUT

    def test_colours_are_distinct(self):
        cols = {COLOUR_CLOSE, COLOUR_LUNGE, COLOUR_ADVANCE_LUNGE, COLOUR_OUT}
        assert len(cols) == 4


# -------------------- pose helpers --------------------

class TestGetHipCentre:
    def test_both_hips_present(self):
        lms = {LM_LEFT_HIP: (100, 400), LM_RIGHT_HIP: (140, 410)}
        assert get_hip_centre(lms) == (120.0, 405.0)

    def test_only_left_hip(self):
        assert get_hip_centre({LM_LEFT_HIP: (50, 300)}) == (50, 300)

    def test_only_right_hip(self):
        assert get_hip_centre({LM_RIGHT_HIP: (90, 320)}) == (90, 320)

    def test_no_hips(self):
        assert get_hip_centre({}) is None


class TestGetFrontFoot:
    """Front foot = the ankle nearest the opponent in image x-coordinates."""

    def _landmarks(self, left_ankle, right_ankle):
        return {LM_LEFT_ANKLE: left_ankle, LM_RIGHT_ANKLE: right_ankle}

    def test_left_fencer_front_foot_is_higher_x(self):
        lms = self._landmarks(left_ankle=(80, 400), right_ankle=(120, 400))
        # left fencer (x=100), opponent on the right (x=500) -> front foot has higher x
        assert get_front_foot(lms, 100, 500) == (120, 400)

    def test_right_fencer_front_foot_is_lower_x(self):
        lms = self._landmarks(left_ankle=(480, 400), right_ankle=(520, 400))
        # right fencer (x=500), opponent on the left (x=100) -> front foot has lower x
        assert get_front_foot(lms, 500, 100) == (480, 400)

    def test_missing_left_returns_right(self):
        assert get_front_foot({LM_RIGHT_ANKLE: (120, 400)}, 100, 500) == (120, 400)

    def test_missing_right_returns_left(self):
        assert get_front_foot({LM_LEFT_ANKLE: (80, 400)}, 100, 500) == (80, 400)

    def test_no_ankles_returns_none(self):
        assert get_front_foot({}, 100, 500) is None


# -------------------- FencerTracker --------------------

class TestFencerTracker:
    """
    The tracker assigns Fencer 1 = leftmost on the first multi-person
    frame, then uses spatial continuity to keep that assignment stable
    even when ByteTrack IDs change.
    """

    def test_initial_assignment_by_position(self):
        t = FencerTracker()
        ids   = np.array([5, 7])
        # leftmost box first
        boxes = np.array([[ 50, 100, 150, 400],
                          [400, 100, 500, 400]])
        confs = np.array([0.9, 0.8])
        slots = t.select(ids, boxes, confs)

        assert slots[0] is not None and slots[0][1] == 5   # leftmost
        assert slots[1] is not None and slots[1][1] == 7   # rightmost

    def test_initial_assignment_independent_of_confidence_order(self):
        # higher-confidence detection is on the right;
        # leftmost should still go into slot 0.
        t = FencerTracker()
        ids   = np.array([5, 7])
        boxes = np.array([[400, 100, 500, 400],     # rightmost, conf=0.95
                          [ 50, 100, 150, 400]])    # leftmost,  conf=0.7
        confs = np.array([0.95, 0.7])
        slots = t.select(ids, boxes, confs)

        assert slots[0][1] == 7   # leftmost
        assert slots[1][1] == 5   # rightmost

    def test_one_detection_initially_goes_to_slot_a(self):
        t = FencerTracker()
        slots = t.select(
            np.array([5]),
            np.array([[ 50, 100, 150, 400]]),
            np.array([0.9]),
        )
        assert slots[0] is not None and slots[0][1] == 5
        assert slots[1] is None

    def test_swapped_detection_order_is_corrected(self):
        # initial frame assigns 5 -> slot0, 7 -> slot1
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[50, 100, 150, 400], [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        # subsequent frame: detections returned in opposite order, slight motion
        slots = t.select(
            np.array([7, 5]),
            np.array([[410, 110, 510, 410], [60, 110, 160, 410]]),
            np.array([0.85, 0.92]),
        )
        # slot 0 should still be the left fencer (now ID 5 box)
        assert slots[0][1] == 5
        assert slots[1][1] == 7

    def test_byte_track_id_change_is_tolerated(self):
        # initialise on IDs 5 and 7
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[50, 100, 150, 400], [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        # next frame: ByteTrack has assigned new IDs 12 and 14 to the same fencers
        slots = t.select(
            np.array([12, 14]),
            np.array([[55, 100, 155, 400], [405, 100, 505, 400]]),
            np.array([0.9, 0.8]),
        )
        # slot 0 stays leftmost (new ID 12), slot 1 stays rightmost (new ID 14)
        assert slots[0][1] == 12
        assert slots[1][1] == 14

    def test_single_detection_is_assigned_to_closer_slot(self):
        t = FencerTracker()
        # initialise
        t.select(
            np.array([5, 7]),
            np.array([[50, 100, 150, 400], [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        # only one detection this frame, near where slot 1 last was
        slots = t.select(
            np.array([5]),
            np.array([[410, 110, 510, 410]]),
            np.array([0.95]),
        )
        # closer to slot 1's last position -> assigned to slot 1
        assert slots[0] is None
        assert slots[1] is not None and slots[1][1] == 5

    def test_empty_detections(self):
        t = FencerTracker()
        slots = t.select(np.array([], dtype=int), np.empty((0, 4)), np.array([]))
        assert slots == [None, None]

    def test_far_away_bystander_is_rejected(self):
        """A new detection far from the slot's last position must not steal it."""
        t = FencerTracker()
        # lock both fencers
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],     # h=300, centre=(100,250)
                      [400, 100, 500, 400]]),   # h=300, centre=(450,250)
            np.array([0.9, 0.8]),
        )
        # next frame: fencer 1 disappears, a bystander appears very far away
        # bystander is well beyond GATE_DISTANCE_RATIO * box height of slot 0
        # -> should be rejected; slot 0 stays empty
        slots = t.select(
            np.array([9, 7]),
            np.array([[1500, 100, 1600, 400],   # bystander, far right
                      [ 405, 100,  505, 400]]), # real fencer 2, near last pos
            np.array([0.95, 0.85]),
        )
        assert slots[0] is None             # bystander rejected
        assert slots[1] is not None         # real fencer 2 still tracked
        assert slots[1][1] == 7

    def test_much_smaller_bystander_is_rejected(self):
        """A new detection much smaller than the last accepted box must not steal a slot."""
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],     # h=300
                      [400, 100, 500, 400]]),   # h=300
            np.array([0.9, 0.8]),
        )
        # slot 0 candidate is a small box (h=100) near slot 0's last position
        # 100/300 = 0.33, below MIN_SIZE_RATIO (0.5) -> rejected
        slots = t.select(
            np.array([9, 7]),
            np.array([[ 60, 300, 110, 400],     # small bystander near slot 0
                      [405, 100, 505, 400]]),   # real fencer 2
            np.array([0.95, 0.8]),
        )
        assert slots[0] is None
        assert slots[1] is not None

    def test_normal_motion_still_passes_gate(self):
        """Realistic frame-to-frame motion must not be rejected by the gates."""
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],
                      [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        # both fencers drift ~20 px (well inside gates), boxes same size
        slots = t.select(
            np.array([5, 7]),
            np.array([[ 70, 100, 170, 400],
                      [420, 100, 520, 400]]),
            np.array([0.9, 0.8]),
        )
        assert slots[0] is not None
        assert slots[1] is not None


# -------------------- PushPullTracker --------------------

class TestPushPullTracker:
    """
    The tracker smooths each fencer's x with a rolling median before
    computing per-frame movement. Tests use smooth_window=1 (no
    smoothing) when they want to assert on a precise frame-to-frame
    delta, and the default window when they want to verify smoothing.
    """

    def test_initial_state_is_zero(self):
        p = PushPullTracker()
        assert p.advance_m == [0.0, 0.0]
        assert p.retreat_m == [0.0, 0.0]

    def test_first_update_only_records_position(self):
        p = PushPullTracker(smooth_window=1)
        p.update(idx=0, fencer_x=100, opponent_x=500, scale_px_per_m=100)
        assert p.advance_m[0] == 0.0
        assert p.retreat_m[0] == 0.0
        assert p.prev_smooth[0] == 100

    def test_advance_when_moving_toward_opponent(self):
        p = PushPullTracker(smooth_window=1)   # no smoothing for clean assertion
        p.update(0, 100, 500, 100)
        # +10 px = +0.1 m advance (above 0.03 noise floor, below 0.15 clamp)
        p.update(0, 110, 500, 100)
        assert math.isclose(p.advance_m[0], 0.1, abs_tol=1e-9)
        assert p.retreat_m[0] == 0.0

    def test_retreat_when_moving_away(self):
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 100)
        # -10 px (away from opponent on right) = 0.1 m retreat
        p.update(0, 90, 500, 100)
        assert p.advance_m[0] == 0.0
        assert math.isclose(p.retreat_m[0], 0.1, abs_tol=1e-9)

    def test_advance_works_for_opponent_on_left(self):
        p = PushPullTracker(smooth_window=1)
        # this fencer is on the right; opponent on the left
        p.update(1, 500, 100, 100)
        # -10 px -> advancing toward opponent on the left
        p.update(1, 490, 100, 100)
        assert math.isclose(p.advance_m[1], 0.1, abs_tol=1e-9)
        assert p.retreat_m[1] == 0.0

    def test_camera_pan_jump_is_ignored(self):
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 100)
        # a jump > MAX_FRAME_MOVEMENT_M -> dropped entirely
        jump_px = (MAX_FRAME_MOVEMENT_M + 0.1) * 100
        p.update(0, 100 + jump_px, 500, 100)
        assert p.advance_m[0] == 0.0
        assert p.retreat_m[0] == 0.0

    def test_noise_floor_ignores_tiny_moves(self):
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 100)
        tiny_px = (PUSH_PULL_NOISE_FLOOR_M / 2) * 100
        p.update(0, 100 + tiny_px, 500, 100)
        assert p.advance_m[0] == 0.0
        assert p.retreat_m[0] == 0.0

    def test_accumulates_across_many_frames(self):
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 100)
        # +10px per step, each step crosses the noise floor (0.1m > 0.03)
        for x in (110, 120, 130, 140):
            p.update(0, x, 500, 100)
        assert math.isclose(p.advance_m[0], 0.4, abs_tol=1e-9)

    def test_smoothing_reduces_jitter_accumulation(self):
        """With smoothing on, noisy positions accumulate less motion."""
        sequence = [100, 105, 95, 102, 98, 103, 97, 101, 99, 100]

        no_smooth   = PushPullTracker(smooth_window=1)
        with_smooth = PushPullTracker(smooth_window=5)
        for x in sequence:
            no_smooth.update(0, x, 500, 100)
            with_smooth.update(0, x, 500, 100)

        total_no   = no_smooth.advance_m[0]   + no_smooth.retreat_m[0]
        total_with = with_smooth.advance_m[0] + with_smooth.retreat_m[0]
        assert total_with < total_no

    def test_zero_scale_is_safe(self):
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 0)
        p.update(0, 110, 500, 0)
        assert p.advance_m[0] == 0.0
        assert p.retreat_m[0] == 0.0


# -------------------- smoothing --------------------

class TestSmoothDistance:
    def test_single_value(self):
        buf = deque()
        assert smooth_distance(buf, 2.0, window=5) == 2.0

    def test_median_of_three(self):
        buf = deque()
        smooth_distance(buf, 1.0, window=5)
        smooth_distance(buf, 3.0, window=5)
        # median of [1, 3, 2] = 2
        assert smooth_distance(buf, 2.0, window=5) == 2.0

    def test_window_evicts_old_values(self):
        buf = deque()
        for v in (10.0, 10.0, 10.0):
            smooth_distance(buf, v, window=3)
        # next value pushes the first 10 out of the window
        result = smooth_distance(buf, 4.0, window=3)
        # buffer is now [10, 10, 4]; median = 10
        assert result == 10.0
        # one more push and it becomes [10, 4, 4]; median = 4
        result = smooth_distance(buf, 4.0, window=3)
        assert result == 4.0

    def test_robust_to_outlier(self):
        # one extreme value should not move the median much
        buf = deque()
        for v in (2.0, 2.1, 2.0, 1.9):
            smooth_distance(buf, v, window=5)
        result = smooth_distance(buf, 99.0, window=5)
        # median is much closer to 2 than the mean would be (~21.8)
        assert 1.5 < result < 2.5


# -------------------- PisteRegion --------------------

class TestPisteRegionContains:
    """Point-in-polygon acceptance for the piste filter."""

    def _rect(self):
        # 100..500 x 200..600
        return PisteRegion([[100, 200], [500, 200], [500, 600], [100, 600]])

    def test_inside_point_accepted(self):
        assert self._rect().contains((300, 400)) is True

    def test_outside_point_rejected(self):
        # far to the right of the polygon
        assert self._rect().contains((900, 400)) is False

    def test_above_polygon_rejected(self):
        # above the top edge
        assert self._rect().contains((300, 100)) is False

    def test_on_edge_is_inside(self):
        # cv2.pointPolygonTest returns 0 for on-edge; contains() treats that as inside
        assert self._rect().contains((100, 400)) is True

    def test_at_vertex_is_inside(self):
        assert self._rect().contains((500, 600)) is True

    def test_polygon_with_fewer_than_three_vertices_raises(self):
        with pytest.raises(ValueError):
            PisteRegion([[0, 0], [10, 10]])

    def test_none_polygon_raises(self):
        with pytest.raises(ValueError):
            PisteRegion(None)


class TestPisteRegionFromJsonFile:
    def test_none_path_returns_none(self):
        # calling from_json_file(None) is the "no piste config" path,
        # not an error
        assert PisteRegion.from_json_file(None) is None

    def test_loads_polygon_from_valid_file(self):
        polygon = [[10, 10], [200, 10], [200, 200], [10, 200]]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump({"polygon": polygon}, f)
            path = f.name
        try:
            region = PisteRegion.from_json_file(path)
            assert region.contains((100, 100)) is True
            assert region.contains((500, 500)) is False
        finally:
            os.unlink(path)


class TestPisteRegionDraw:
    """The diagnostic overlay must mark the frame without altering its shape."""

    def test_draw_modifies_frame_in_place(self):
        region = PisteRegion([[10, 10], [200, 10], [200, 200], [10, 200]])
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        out = region.draw(frame)
        assert out.shape == (300, 300, 3)
        assert out.any(), "draw() should have written some non-zero pixels"

    def test_draw_leaves_polygon_interior_mostly_untouched(self):
        # the outline is drawn, not a fill, so the centre should stay black
        region = PisteRegion([[10, 10], [200, 10], [200, 200], [10, 200]])
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        region.draw(frame)
        assert frame[105, 105].sum() == 0


class TestPisteRegionFilterDetections:
    """
    filter_detections() must drop detections whose feet fall outside the
    piste, and it must preserve the parallel-array shape (ids, xyxys,
    confs) so downstream code does not have to change.
    """

    def _rect(self):
        return PisteRegion([[100, 200], [500, 200], [500, 600], [100, 600]])

    def test_empty_input_returns_empty(self):
        region = self._rect()
        ids, xyxys, confs = region.filter_detections(
            np.array([], dtype=int), np.empty((0, 4)), np.array([]),
        )
        assert len(ids) == 0
        assert xyxys.shape == (0, 4)
        assert len(confs) == 0

    def test_keeps_inside_detection(self):
        region = self._rect()
        # bbox with feet at (200, 500) - inside the rectangle
        boxes = np.array([[150, 300, 250, 500]])
        ids, xyxys, confs = region.filter_detections(
            np.array([7]), boxes, np.array([0.9]),
        )
        assert len(ids) == 1
        assert ids[0] == 7

    def test_drops_outside_detection(self):
        region = self._rect()
        # bbox with feet at (800, 500) - outside the rectangle
        boxes = np.array([[750, 300, 850, 500]])
        ids, xyxys, confs = region.filter_detections(
            np.array([7]), boxes, np.array([0.9]),
        )
        assert len(ids) == 0
        assert xyxys.shape == (0, 4)
        assert len(confs) == 0

    def test_mixed_detections_keeps_only_inside(self):
        region = self._rect()
        # first detection is inside, second is way outside (a referee)
        boxes = np.array([
            [150, 300, 250, 500],   # feet at (200, 500), inside
            [800, 300, 900, 500],   # feet at (850, 500), outside
        ])
        ids, xyxys, confs = region.filter_detections(
            np.array([5, 9]), boxes, np.array([0.9, 0.8]),
        )
        assert list(ids) == [5]
        assert confs[0] == 0.9


# -------------------- Motion model (predicted_pos) --------------------

class TestFencerTrackerPredictedPos:
    """
    The tracker keeps a two-point position history per slot so that the
    matcher can gate against where each slot is HEADED, not just where
    it last was - the fix for the "bystander steps into the old position
    while the fencer keeps moving" case.
    """

    def test_predicted_pos_none_when_slot_uninitialised(self):
        t = FencerTracker()
        assert t.predicted_pos(0) is None
        assert t.predicted_pos(1) is None

    def test_predicted_pos_equals_last_after_first_commit(self):
        # only one commit -> no velocity information yet; predicted == last
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[50, 100, 150, 400], [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        assert t.predicted_pos(0) == t.last_pos[0]
        assert t.predicted_pos(1) == t.last_pos[1]

    def test_predicted_pos_extrapolates_velocity(self):
        # two commits with the left fencer drifting +20px per frame in x
        t = FencerTracker()
        t.select(
            np.array([5, 7]),
            np.array([[50, 100, 150, 400], [400, 100, 500, 400]]),
            np.array([0.9, 0.8]),
        )
        t.select(
            np.array([5, 7]),
            np.array([[70, 100, 170, 400], [420, 100, 520, 400]]),
            np.array([0.9, 0.8]),
        )
        # slot 0: last centre = (120, 250), prev centre = (100, 250);
        # predicted = (120 + 20, 250) = (140, 250)
        pred = t.predicted_pos(0)
        assert math.isclose(pred[0], 140.0)
        assert math.isclose(pred[1], 250.0)

    def test_bystander_stepping_into_old_position_is_rejected(self):
        """
        The whole point of the motion model: a bystander walks INTO where
        the fencer used to be, while the fencer keeps moving forward.
        Under last-known-position gating the bystander would look like a
        perfect match; under predicted-position gating it should not.
        """
        t = FencerTracker()
        # frame 1: fencers at x=100 and x=1000
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],     # slot 0 centre = (100, 250)
                      [950, 100, 1050, 400]]),  # slot 1 centre = (1000, 250)
            np.array([0.9, 0.8]),
        )
        # frame 2: slot 0 fencer has ADVANCED to x=400; slot 1 unchanged
        t.select(
            np.array([5, 7]),
            np.array([[350, 100, 450, 400],     # slot 0 centre = (400, 250)
                      [950, 100, 1050, 400]]),  # slot 1 centre = (1000, 250)
            np.array([0.9, 0.8]),
        )
        # frame 3: two detections. The FENCER is at x=700 (kept moving),
        # a BYSTANDER (identical size) is at x=100 - exactly where slot 0
        # started. Old code compared to last_pos=(400,250) and would have
        # picked the bystander (dist 300) over the real fencer (dist 300),
        # tied by original assignment; and the bystander was well inside
        # 3.5 * box height. The motion-model gate predicts slot 0 should
        # now be near x=700 and rejects the bystander at x=100.
        slots = t.select(
            np.array([5, 9]),
            np.array([[650, 100,  750, 400],   # real fencer 1 at (700, 250)
                      [ 50, 100,  150, 400]]),  # bystander at (100, 250)
            np.array([0.9, 0.85]),
        )
        # slot 0 must match the real fencer, not the bystander
        assert slots[0] is not None
        assert slots[0][1] == 5   # the real fencer's id
        assert 650 <= slots[0][0][0] <= 750


class TestMotionModelGapHandling:
    """
    Regression tests for a bug found on real footage: velocity was being
    differenced between two commits that were many frames apart, which
    measures total displacement over the gap rather than per-frame
    velocity. Extrapolating from it threw the predicted position far
    outside the frame, after which every real detection failed the gate
    and the slot could never commit again - and because nothing was
    committed, the bad history was never replaced. Coverage on the second
    test clip collapsed from 87% to 13% as a result.
    """

    def _init_both_slots(self, t):
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],
                      [900, 100, 1000, 400]]),
            np.array([0.9, 0.8]),
        )

    def test_velocity_not_extrapolated_across_a_long_gap(self):
        t = FencerTracker()
        self._init_both_slots(t)
        # slot 1 goes unmatched for many frames: only a slot-0 detection
        # arrives, far from slot 1, so slot 1 is never committed.
        for _ in range(10):
            t.select(
                np.array([5]),
                np.array([[50, 100, 150, 400]]),
                np.array([0.9]),
            )
        # slot 1 now commits again, but far from where it last was
        t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],
                      [300, 100, 400, 400]]),   # centre now (350, 250)
            np.array([0.9, 0.8]),
        )
        # gap between slot 1's two commits is > VELOCITY_MAX_GAP_FRAMES,
        # so the prediction must NOT extrapolate; it must equal last_pos.
        assert t.predicted_pos(1) == t.last_pos[1]

    def test_prediction_stays_on_screen_after_a_gap(self):
        """The concrete symptom: predicted position must not go negative."""
        t = FencerTracker()
        self._init_both_slots(t)
        for _ in range(10):
            t.select(
                np.array([5]),
                np.array([[1100, 100, 1200, 400]]),
                np.array([0.9]),
            )
        t.select(
            np.array([5, 7]),
            np.array([[1100, 100, 1200, 400],
                      [ 400, 100,  500, 400]]),
            np.array([0.9, 0.8]),
        )
        for slot in (0, 1):
            pred = t.predicted_pos(slot)
            if pred is not None:
                assert pred[0] >= 0, f"slot {slot} predicted x went negative"
                assert pred[1] >= 0, f"slot {slot} predicted y went negative"

    def test_slot_recovers_after_long_dropout(self):
        """
        A slot that has gone unmatched for longer than STALE_RESET_FRAMES
        must discard its history so it can re-acquire, rather than
        rejecting every candidate forever.
        """
        t = FencerTracker()
        self._init_both_slots(t)
        # slot 1 receives no detections for well over the stale threshold
        for _ in range(FencerTracker.STALE_RESET_FRAMES + 5):
            t.select(
                np.array([5]),
                np.array([[50, 100, 150, 400]]),
                np.array([0.9]),
            )
        assert t.last_pos[1] is None, "stale slot history should be cleared"

        # a fencer reappears somewhere new; the freed slot must accept it
        slots = t.select(
            np.array([5, 7]),
            np.array([[ 50, 100, 150, 400],
                      [600, 100, 700, 400]]),
            np.array([0.9, 0.8]),
        )
        assert slots[1] is not None, "slot should have re-acquired after dropout"

    def test_slot_history_survives_a_short_dropout(self):
        """
        The stale reset must not fire on brief dropouts - those are the
        cases spatial continuity is supposed to ride out.
        """
        t = FencerTracker()
        self._init_both_slots(t)
        for _ in range(3):
            t.select(
                np.array([5]),
                np.array([[50, 100, 150, 400]]),
                np.array([0.9]),
            )
        assert t.last_pos[1] is not None, "short dropout should keep history"
