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
    LM_LEFT_ANKLE,
    LM_RIGHT_ANKLE,
    LM_LEFT_HIP,
    LM_RIGHT_HIP,
    get_stance_features,
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

    def test_camera_pan_jump_is_bounded_not_dropped(self):
        """
        A jump beyond MAX_FRAME_MOVEMENT_M is capped at that limit rather than
        discarded.

        This test previously asserted the jump was dropped entirely, and that
        behaviour turned out to be a source of bias rather than robustness.
        Measured on the club clip, the threshold was exceeded on about one per
        cent of frames, and those frames carried roughly ten metres of net
        retreat, so discarding them injected ten metres of false advance.
        Capping bounds how much a single bad frame can contribute while keeping
        the direction of the movement underneath it.
        """
        p = PushPullTracker(smooth_window=1)
        p.update(0, 100, 500, 100)
        jump_px = (MAX_FRAME_MOVEMENT_M + 0.1) * 100
        p.update(0, 100 + jump_px, 500, 100)
        assert math.isclose(p.advance_m[0], MAX_FRAME_MOVEMENT_M, abs_tol=1e-9)
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


# -------------------- CameraMotionEstimator --------------------

class TestCameraMotionEstimator:
    """
    Push and pull are accumulated from image-space displacement, which conflates
    the fencer moving with the camera moving. The existing safeguards cannot
    catch a slow pan: the per-frame clamp only rejects biomechanically
    impossible jumps and the noise floor only rejects sub-jitter movement. On
    hand-held footage the unstabilised metric reported both fencers
    net-advancing a combined 32 m on a 14 m piste.
    """

    def _texture(self, h=400, w=600, seed=0):
        import cv2
        rng = np.random.default_rng(seed)
        img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (5, 5), 0)   # give LK a trackable gradient

    def test_first_frame_reports_no_motion(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        assert est.update(self._texture()) == 0.0

    def test_static_camera_reports_near_zero(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        img = self._texture()
        est.update(img)
        for _ in range(4):
            assert abs(est.update(img)) < 0.5

    def test_recovers_a_known_pan(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        base = self._texture()
        est.update(base)
        for i in range(1, 5):
            dx = est.update(np.roll(base, i * 5, axis=1))
            assert abs(dx - 5.0) < 1.0, f"expected ~5 px, got {dx}"

    def test_cumulative_offset_accumulates(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        base = self._texture()
        est.update(base)
        for i in range(1, 6):
            est.update(np.roll(base, i * 4, axis=1))
        assert abs(est.cumulative_dx - 20.0) < 2.0

    def test_stabilise_removes_the_offset(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        est.cumulative_dx = 30.0
        assert est.stabilise(100.0) == 70.0

    def test_stabilise_passes_none_through(self):
        from run_detection import CameraMotionEstimator
        assert CameraMotionEstimator().stabilise(None) is None

    def test_implausible_shift_is_rejected(self):
        """A cut or flash produces a huge apparent shift; better to report no
        motion than to inject a spurious one into the accumulator."""
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        est.update(self._texture(seed=1))
        # an unrelated frame gives incoherent flow, well beyond a real pan
        dx = est.update(self._texture(seed=99))
        limit = est.MAX_PAN_FRACTION * 600
        assert abs(dx) <= limit

    def test_featureless_frame_fails_safely(self):
        from run_detection import CameraMotionEstimator
        est = CameraMotionEstimator()
        blank = np.zeros((400, 600, 3), dtype=np.uint8)
        est.update(blank)
        assert est.update(blank) == 0.0
        assert est.frames_failed > 0

    def test_fencer_boxes_are_masked_out(self):
        """
        The fencers move too, so features on them would bias the estimate. They
        are excluded directly rather than relying on the median alone.
        """
        from run_detection import CameraMotionEstimator
        shape = (400, 600, 3)
        mask = CameraMotionEstimator._mask_excluding(shape, [[100, 100, 200, 300]])
        assert mask[200, 150] == 0        # inside the box
        assert mask[50, 500] == 255       # well outside it

    def test_masking_tolerates_missing_boxes(self):
        from run_detection import CameraMotionEstimator
        mask = CameraMotionEstimator._mask_excluding((400, 600, 3), [None])
        assert (mask == 255).all()


# -------------------- push/pull banking (directional bias) --------------------

class TestPushPullBanking:
    """
    The noise threshold exists to stop bounding-box jitter accumulating; an early
    version summed raw displacement and reported 216 m of push per fencer in a
    three-minute bout. Discarding sub-threshold movement fixed that number and
    introduced a directional bias, because advances and retreats in fencing do
    not share a speed: an attack is explosive and clears the threshold on every
    frame, while the recovery is slow and clears it on none.

    Banking sub-threshold movement instead of discarding it keeps the jitter
    rejection and removes the bias. These tests pin both halves of that, because
    a change to either constant could silently reintroduce the bug.
    """

    SCALE = 143.0

    def _cycle(self, adv_px, adv_frames, ret_frames, cycles=12):
        """
        Drive a fencer through advance/retreat cycles that return it to the exact
        starting position, so the correct net displacement is zero by construction.
        """
        p = PushPullTracker(n_fencers=2)
        x, opponent = 400.0, 900.0
        ret_px = (adv_px * adv_frames) / ret_frames
        for _ in range(cycles):
            for _ in range(adv_frames):
                x += adv_px
                p.update(0, x, opponent, self.SCALE)
            for _ in range(ret_frames):
                x -= ret_px
                p.update(0, x, opponent, self.SCALE)
        return p

    def test_symmetric_motion_nets_to_zero(self):
        p = self._cycle(adv_px=8.0, adv_frames=10, ret_frames=10)
        assert abs(p.advance_m[0] - p.retreat_m[0]) < 0.5

    def test_explosive_attack_slow_recovery_nets_to_zero(self):
        """
        The regression this fix addresses. With sub-threshold movement discarded
        this reported +5.25 m of net advance and pull of exactly 0.00 m, having
        thrown away every retreat frame.
        """
        p = self._cycle(adv_px=8.0, adv_frames=10, ret_frames=40)
        net = p.advance_m[0] - p.retreat_m[0]
        assert p.retreat_m[0] > 0, "slow retreats must not be discarded entirely"
        assert abs(net) < 0.5, f"expected net near zero, got {net:+.2f}"

    def test_very_slow_recovery_nets_to_zero(self):
        p = self._cycle(adv_px=8.0, adv_frames=10, ret_frames=80)
        assert abs(p.advance_m[0] - p.retreat_m[0]) < 0.5

    def test_jitter_still_does_not_accumulate(self):
        """
        The other half. A random walk with no true displacement must not produce
        large totals, or banking would have simply undone the original fix.
        """
        rng = np.random.default_rng(0)
        p = PushPullTracker(n_fencers=2)
        x = 400.0
        for _ in range(2000):
            x += rng.normal(0, 1.5)
            p.update(0, x, 900.0, self.SCALE)
        assert p.advance_m[0] < 10.0 and p.retreat_m[0] < 10.0

    def test_a_single_tiny_move_commits_nothing(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, 100.0)
        tiny = (PUSH_PULL_NOISE_FLOOR_M / 3) * 100.0
        p.update(0, 100.0 + tiny, 500.0, 100.0)
        assert p.advance_m[0] == 0.0
        assert p.pending_m[0] > 0, "it should be banked, not lost"

    def test_repeated_tiny_moves_eventually_commit(self):
        """Slow movement is delayed, never dropped."""
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, 100.0)
        step = (PUSH_PULL_NOISE_FLOOR_M / 3) * 100.0
        x = 100.0
        for _ in range(6):
            x += step
            p.update(0, x, 500.0, 100.0)
        assert p.advance_m[0] > 0, "banked movement should have committed by now"

    def test_implausible_jump_is_capped_not_discarded(self):
        """
        An implausible single-frame jump is capped, not deleted. Deleting it
        biased the result: on the club clip the threshold was exceeded on about
        one per cent of frames, but those frames carried roughly ten metres of
        net retreat, so discarding them injected ten metres of false advance.
        Capping bounds a glitch's influence while keeping its direction.
        """
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, 100.0)
        huge = (MAX_FRAME_MOVEMENT_M + 0.5) * 100.0
        p.update(0, 100.0 + huge, 500.0, 100.0)
        assert math.isclose(p.advance_m[0], MAX_FRAME_MOVEMENT_M, abs_tol=1e-9)
        assert p.retreat_m[0] == 0.0

    def test_a_capped_jump_keeps_its_direction(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 500.0, 900.0, 100.0)
        huge = (MAX_FRAME_MOVEMENT_M + 0.5) * 100.0
        p.update(0, 500.0 - huge, 900.0, 100.0)   # a big jump AWAY from the opponent
        assert math.isclose(p.retreat_m[0], MAX_FRAME_MOVEMENT_M, abs_tol=1e-9)
        assert p.advance_m[0] == 0.0


# -------------------- push/pull direction stability --------------------

class TestPushPullDirectionIsFixed:
    """
    "Toward the opponent" is established once per fencer and then held for the
    bout. The original implementation re-derived it every frame by comparing the
    opponent's current position against this fencer's previous one. On the club
    clip that returned the wrong side on 3 frames out of 5,109 and cost 10
    metres, because the comparison only fails when a position jumps, and a jump
    is exactly when the frame's displacement is large. A large movement given the
    wrong sign contributes twice its magnitude as error.
    """

    SCALE = 150.0

    def test_direction_is_learned_on_first_update(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, self.SCALE)     # opponent to the right
        p.update(1, 500.0, 100.0, self.SCALE)     # opponent to the left
        assert p.toward_opponent[0] == 1.0
        assert p.toward_opponent[1] == -1.0

    def test_direction_does_not_change_afterwards(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, self.SCALE)
        # a glitch frame reporting the opponent on the wrong side
        p.update(0, 110.0, 50.0, self.SCALE)
        assert p.toward_opponent[0] == 1.0

    def test_a_single_glitch_frame_cannot_invert_a_large_movement(self):
        """
        The regression this fixes. A large displacement arriving on a frame where
        the opponent appears on the wrong side must still be counted in the
        correct direction.
        """
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, 500.0, self.SCALE)     # establishes: right is forward
        # move toward the opponent while the opponent's reported x is corrupted
        p.update(0, 112.0, 20.0, self.SCALE)
        assert p.advance_m[0] > 0, "movement toward the opponent must count as advance"
        assert p.retreat_m[0] == 0.0

    def test_accumulated_net_matches_endpoint_displacement(self):
        """
        The arithmetic identity any correct accumulator must satisfy: summing
        signed per-frame movement equals the difference between first and last
        position. This is what the per-frame sign test broke.
        """
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        # steps chosen to stay inside MAX_FRAME_MOVEMENT_M so this measures the
        # sign handling rather than the cap, which has its own tests below
        xs = [100.0, 118.0, 111.0, 129.0, 122.0, 140.0]
        for x in xs:
            p.update(0, x, 900.0, self.SCALE)
        net = p.advance_m[0] - p.retreat_m[0]
        expected = (xs[-1] - xs[0]) / self.SCALE
        assert math.isclose(net, expected, abs_tol=0.02), f"{net} vs {expected}"

    def test_the_cap_breaks_the_identity_by_exactly_what_it_discards(self):
        """
        The counterpart to the test above, and the one that matters on real data.

        That test picks steps inside the cap so it measures sign handling. This one
        crosses the cap deliberately, because the cap is what breaks the identity
        on real footage: replaying clip 3's recorded positions through this logic,
        the cap fires on 63 and 65 frames of 5,246 and destroys -9.26 m and
        -23.94 m of signed movement, which is the whole of the divergence between
        the cumulative totals and the endpoint measurement.

        Pinning it here converts that from a surprise into a known property. The
        assertion is not that the identity holds, because it does not; it is that
        the shortfall equals the discarded amount and nothing else leaks.
        """
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        # A ONE-SIDED jump, i.e. the position steps and stays rather than snapping
        # back. That asymmetry is the point. A symmetric out-and-back glitch
        # crosses the cap twice in opposite directions and mostly cancels, so it
        # does little harm. What damages the total is a step that is never
        # returned, which is what re-acquiring a fencer at a new position after a
        # tracking gap produces, and it explains why clip 3's 65 capped frames
        # carried -23.94 m almost entirely in one direction instead of averaging
        # out.
        over = (MAX_FRAME_MOVEMENT_M + 0.40) * self.SCALE
        xs = [100.0, 110.0, 120.0, 120.0 - over]
        for x in xs:
            p.update(0, x, 900.0, self.SCALE)

        net = p.advance_m[0] - p.retreat_m[0]
        endpoint = (xs[-1] - xs[0]) / self.SCALE
        # the retreat of (cap + 0.40) was kept only as far as the cap, so the
        # accumulated net overstates advance by exactly the discarded remainder
        assert math.isclose(net - endpoint, 0.40, abs_tol=1e-6), f"{net} vs {endpoint}"

    def test_no_cap_crossing_means_the_identity_survives(self):
        """
        The same series without the glitch closes exactly, which isolates the cap
        as the cause rather than the smoothing or the banking buffer.
        """
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        xs = [100.0, 110.0, 120.0, 114.0, 130.0, 140.0]
        for x in xs:
            p.update(0, x, 900.0, self.SCALE)
        net = p.advance_m[0] - p.retreat_m[0]
        assert math.isclose(net, (xs[-1] - xs[0]) / self.SCALE, abs_tol=1e-6)

    def test_identity_holds_with_a_corrupted_opponent_position(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        xs = [100.0, 118.0, 111.0, 129.0, 122.0, 140.0]
        for i, x in enumerate(xs):
            # every third frame reports the opponent on the wrong side
            opp = 20.0 if i % 3 == 0 and i > 0 else 900.0
            p.update(0, x, opp, self.SCALE)
        net = p.advance_m[0] - p.retreat_m[0]
        expected = (xs[-1] - xs[0]) / self.SCALE
        assert math.isclose(net, expected, abs_tol=0.02)

    def test_no_opponent_position_means_no_accumulation(self):
        """Without an opponent the sign is undetermined, so nothing is counted."""
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        p.update(0, 100.0, None, self.SCALE)
        p.update(0, 140.0, None, self.SCALE)
        assert p.advance_m[0] == 0.0 and p.retreat_m[0] == 0.0


# -------------------- plot rendering --------------------

class TestSavePlot:
    """
    save_plot runs only at the very end of a pipeline run, after the CSV is
    written, so a NameError in it is silent: the data survives and the run
    appears to finish. Renaming the distance-band constants did exactly that and
    it went unnoticed for hours, because the only visible symptom was a missing
    plot and a missing printed summary. These tests exercise it directly.
    """

    def test_renders_a_plot_file(self, tmp_path):
        from run_detection import save_plot
        out = tmp_path / "p.png"
        save_plot([0.0, 0.5, 1.0, 1.5], [2.0, 2.4, 1.8, 3.1],
                  ["pose", "bbox", "pose", "bbox"], str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_handles_pose_only_samples(self, tmp_path):
        from run_detection import save_plot
        out = tmp_path / "p.png"
        save_plot([0.0, 0.5], [2.0, 2.4], ["pose", "pose"], str(out))
        assert out.exists()

    def test_handles_bbox_only_samples(self, tmp_path):
        from run_detection import save_plot
        out = tmp_path / "p.png"
        save_plot([0.0, 0.5], [2.0, 2.4], ["bbox", "bbox"], str(out))
        assert out.exists()

    def test_every_band_constant_it_draws_still_exists(self):
        """Guards against a constant rename silently breaking the plot again."""
        import run_detection as rd
        for name in ("DIST_CLOSE_M", "DIST_LUNGE_M", "DIST_ADVANCE_LUNGE_M"):
            assert hasattr(rd, name), f"save_plot draws {name}, which is missing"


# -------------------- closing share --------------------

class TestClosingShare:
    """
    The well-defined replacement for "how far did each fencer advance in total".

    That question has no answer as a distance in this data. Path length sums the
    magnitude of every frame's change, so measurement noise adds to it and never
    cancels: re-measuring clip 3 under median windows from 1 to 121 frames moved
    one fencer's path length from 161 m to 33 m with no asymptote, while net
    displacement stayed at exactly +3.43 m throughout. A quantity that changes
    fivefold with an arbitrary smoothing parameter measures the filter, not the
    fencer.

    Counting the sign of each movement instead of its magnitude avoids that,
    because noise contributes symmetrically to both directions. Under the same
    sweep the closing share moved only from 52.4 to 60.9 per cent.
    """

    SCALE = 150.0

    def test_none_before_any_movement(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        assert p.closing_share(0) is None

    def test_all_closing_gives_one(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        x = 100.0
        p.update(0, x, 900.0, self.SCALE)
        for _ in range(10):
            x += 8.0                      # steadily toward the opponent
            p.update(0, x, 900.0, self.SCALE)
        assert p.closing_share(0) == 1.0

    def test_all_opening_gives_zero(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        x = 500.0
        p.update(0, x, 900.0, self.SCALE)
        for _ in range(10):
            x -= 8.0                      # steadily away
            p.update(0, x, 900.0, self.SCALE)
        assert p.closing_share(0) == 0.0

    def test_alternating_gives_about_half(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        x = 500.0
        p.update(0, x, 900.0, self.SCALE)
        for i in range(20):
            x += 8.0 if i % 2 == 0 else -8.0
            p.update(0, x, 900.0, self.SCALE)
        assert 0.4 <= p.closing_share(0) <= 0.6

    def test_is_bounded(self):
        p = PushPullTracker(n_fencers=2, smooth_window=1)
        x = 100.0
        p.update(0, x, 900.0, self.SCALE)
        rng = np.random.default_rng(0)
        for _ in range(200):
            x += rng.normal(0, 10)
            p.update(0, x, 900.0, self.SCALE)
        s = p.closing_share(0)
        assert 0.0 <= s <= 1.0

    def test_magnitude_does_not_affect_it_above_the_noise_floor(self):
        """
        The property that makes it usable: scaling every movement leaves the
        share unchanged, whereas it would scale a path length proportionally.

        This holds only for movements that clear PUSH_PULL_NOISE_FLOOR_M. Below
        it the banking buffer decides when a movement commits, so magnitude does
        influence the count. The metric is therefore magnitude-invariant in the
        regime that matters and not universally, which is worth knowing before
        quoting it on footage where the fencers barely move.
        """
        floor_px = PUSH_PULL_NOISE_FLOOR_M * self.SCALE
        shares = []
        for step in (floor_px * 2, floor_px * 4, floor_px * 8):
            p = PushPullTracker(n_fencers=2, smooth_window=1)
            x = 100.0
            p.update(0, x, 900.0, self.SCALE)
            for i in range(30):
                x += step if i % 3 else -step
                p.update(0, x, 900.0, self.SCALE)
            shares.append(p.closing_share(0))
        assert max(shares) - min(shares) < 1e-9, shares

    def test_below_the_noise_floor_magnitude_does_matter(self):
        """Pins the limitation above, so it cannot be forgotten."""
        floor_px = PUSH_PULL_NOISE_FLOOR_M * self.SCALE
        shares = []
        for step in (floor_px * 0.3, floor_px * 4):
            p = PushPullTracker(n_fencers=2, smooth_window=1)
            x = 100.0
            p.update(0, x, 900.0, self.SCALE)
            for i in range(30):
                x += step if i % 3 else -step
                p.update(0, x, 900.0, self.SCALE)
            shares.append(p.closing_share(0))
        assert shares[0] != shares[1]


class TestStanceFeatures:
    """
    Stance geometry exists to answer one question: does pose carry a lunge signal
    worth keeping? B1c showed pose success can fall from 27 per cent of frames to
    5.7 per cent with no effect on touch-detection F1, so MediaPipe is currently
    not load-bearing. These tests pin the feature definitions; whether the signal
    is real is a measurement against ground truth, not a unit test.
    """

    SCALE = 100.0        # px per metre, so 100 px = 1 m

    def _lms(self, **overrides):
        """
        Landmarks for a fencer on guard: feet 60 cm apart, hips 1 m above them.

        Landmark keys are integers, so overrides are applied by dict merge at the
        call site rather than through keyword arguments.
        """
        return {
            LM_LEFT_ANKLE:  (100.0, 300.0),
            LM_RIGHT_ANKLE: (160.0, 300.0),
            LM_LEFT_HIP:    (125.0, 200.0),
            LM_RIGHT_HIP:   (135.0, 200.0),
        }

    def test_guard_stance_is_about_shoulder_width(self):
        f = get_stance_features(self._lms(), self.SCALE)
        assert math.isclose(f["stance_m"], 0.60, abs_tol=1e-9)
        assert math.isclose(f["hip_height_m"], 1.00, abs_tol=1e-9)

    def test_a_lunge_widens_the_stance_and_drops_the_hips(self):
        """
        The two features must move in OPPOSITE directions. That is what separates a
        lunge from the same fencer detected at a different scale, which would move
        both the same way and is the confound worth guarding against.
        """
        guard = get_stance_features(self._lms(), self.SCALE)
        # front foot thrown forward, hips dropped
        lunge = get_stance_features({
            LM_LEFT_ANKLE:  (100.0, 300.0),
            LM_RIGHT_ANKLE: (230.0, 300.0),
            LM_LEFT_HIP:    (160.0, 240.0),
            LM_RIGHT_HIP:   (170.0, 240.0),
        }, self.SCALE)
        assert lunge["stance_m"] > guard["stance_m"]
        assert lunge["hip_height_m"] < guard["hip_height_m"]

    def test_stance_is_horizontal_only(self):
        """
        A lunge extends along the piste, and the piste runs across the frame, so
        vertical ankle separation is camera geometry rather than stance. Raising one
        ankle must not change the measurement.
        """
        flat   = get_stance_features(self._lms(), self.SCALE)
        raised = get_stance_features({**self._lms(), LM_LEFT_ANKLE: (100.0, 240.0)},
                                     self.SCALE)
        assert math.isclose(flat["stance_m"], raised["stance_m"], abs_tol=1e-9)

    def test_one_missing_ankle_yields_no_stance_but_keeps_hip_height(self):
        """
        A partial skeleton is the normal case, not a failure. Reporting the feature
        that survives is what stops pose availability collapsing to all-or-nothing.
        """
        lms = {k: v for k, v in self._lms().items() if k != LM_RIGHT_ANKLE}
        f = get_stance_features(lms, self.SCALE)
        assert f["stance_m"] is None
        assert f["hip_height_m"] is not None

    def test_no_landmarks_or_no_scale_yields_none(self):
        assert get_stance_features(None, self.SCALE) is None
        assert get_stance_features({}, self.SCALE) is None
        assert get_stance_features(self._lms(), 0) is None

    def test_ankles_only_yields_stance_but_no_hip_height(self):
        lms = {LM_LEFT_ANKLE: (100.0, 300.0), LM_RIGHT_ANKLE: (160.0, 300.0)}
        f = get_stance_features(lms, self.SCALE)
        assert math.isclose(f["stance_m"], 0.60, abs_tol=1e-9)
        assert f["hip_height_m"] is None
