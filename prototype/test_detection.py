"""
Unit tests for the fencing detection prototype.
Tests the pure utility functions that don't require a video or model.

Run with:
    python3 -m pytest test_detection.py -v
"""

import math
import pytest
from run_detection import (
    get_box_centre,
    box_height_pixels,
    pixel_distance,
    normalise_distance,
    get_front_foot,
    LM_LEFT_ANKLE,
    LM_RIGHT_ANKLE,
)


class TestGetBoxCentre:
    def test_simple_box(self):
        assert get_box_centre([0, 0, 100, 100]) == (50.0, 50.0)

    def test_non_square_box(self):
        assert get_box_centre([10, 20, 50, 80]) == (30.0, 50.0)

    def test_single_pixel_box(self):
        assert get_box_centre([5, 5, 5, 5]) == (5.0, 5.0)


class TestBoxHeightPixels:
    def test_normal_box(self):
        assert box_height_pixels([0, 10, 100, 110]) == 100

    def test_zero_height(self):
        assert box_height_pixels([0, 50, 100, 50]) == 0

    def test_tall_box(self):
        assert box_height_pixels([0, 0, 50, 480]) == 480


class TestPixelDistance:
    def test_zero_distance(self):
        assert pixel_distance((10, 10), (10, 10)) == 0.0

    def test_horizontal(self):
        assert pixel_distance((0, 0), (100, 0)) == 100.0

    def test_vertical(self):
        assert pixel_distance((0, 0), (0, 50)) == 50.0

    def test_diagonal_345(self):
        # classic 3-4-5 right triangle
        assert math.isclose(pixel_distance((0, 0), (3, 4)), 5.0)

    def test_float_coords(self):
        assert math.isclose(pixel_distance((0.5, 0.5), (3.5, 4.5)), 5.0)


class TestNormaliseDistance:
    def test_equal_distance_and_height(self):
        # pixel distance == box height -> result should equal assumed real height
        assert math.isclose(normalise_distance(200, 200, real_height_m=1.75), 1.75)

    def test_zero_height_returns_none(self):
        assert normalise_distance(100, 0) is None

    def test_close_fencers(self):
        assert math.isclose(normalise_distance(100, 200, real_height_m=1.75), 0.875)

    def test_far_fencers(self):
        assert math.isclose(normalise_distance(400, 200, real_height_m=1.75), 3.5)

    def test_custom_real_height(self):
        assert math.isclose(normalise_distance(100, 100, real_height_m=2.0), 2.0)


class TestGetFrontFoot:
    """
    In a side-on view the front foot is the ankle closer to the opponent.
    Fencer on the LEFT (centre_x=100) facing RIGHT (opponent at x=500):
      front foot = ankle with the higher x value.
    Fencer on the RIGHT (centre_x=500) facing LEFT (opponent at x=100):
      front foot = ankle with the lower x value.
    """

    def _landmarks(self, left_ankle, right_ankle):
        return {LM_LEFT_ANKLE: left_ankle, LM_RIGHT_ANKLE: right_ankle}

    def test_left_fencer_front_foot_is_right_ankle(self):
        # left fencer; right ankle is more towards opponent (higher x)
        lms = self._landmarks(left_ankle=(80, 400), right_ankle=(120, 400))
        foot = get_front_foot(lms, fencer_centre_x=100, opponent_centre_x=500)
        assert foot == (120, 400)

    def test_right_fencer_front_foot_is_left_ankle(self):
        # right fencer; left ankle is more towards opponent (lower x)
        lms = self._landmarks(left_ankle=(480, 400), right_ankle=(520, 400))
        foot = get_front_foot(lms, fencer_centre_x=500, opponent_centre_x=100)
        assert foot == (480, 400)

    def test_missing_left_ankle_returns_right(self):
        lms = {LM_RIGHT_ANKLE: (120, 400)}
        foot = get_front_foot(lms, fencer_centre_x=100, opponent_centre_x=500)
        assert foot == (120, 400)

    def test_missing_right_ankle_returns_left(self):
        lms = {LM_LEFT_ANKLE: (80, 400)}
        foot = get_front_foot(lms, fencer_centre_x=100, opponent_centre_x=500)
        assert foot == (80, 400)

    def test_no_ankles_returns_none(self):
        assert get_front_foot({}, fencer_centre_x=100, opponent_centre_x=500) is None

    def test_empty_landmarks_returns_none(self):
        assert get_front_foot({}, 200, 600) is None
