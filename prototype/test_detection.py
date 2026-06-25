"""
Unit tests for the fencing detection prototype.
Tests the pure utility functions that don't require a video or model.

Run with:
    python3 -m pytest test_detection.py -v
"""

import math
import pytest
from run_detection import get_box_centre, box_height_pixels, pixel_distance, normalise_distance


class TestGetBoxCentre:
    def test_simple_box(self):
        # box [x1, y1, x2, y2]
        centre = get_box_centre([0, 0, 100, 100])
        assert centre == (50.0, 50.0)

    def test_non_square_box(self):
        centre = get_box_centre([10, 20, 50, 80])
        assert centre == (30.0, 50.0)

    def test_single_pixel_box(self):
        centre = get_box_centre([5, 5, 5, 5])
        assert centre == (5.0, 5.0)


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

    def test_horizontal_distance(self):
        assert pixel_distance((0, 0), (100, 0)) == 100.0

    def test_vertical_distance(self):
        assert pixel_distance((0, 0), (0, 50)) == 50.0

    def test_diagonal_distance(self):
        # 3-4-5 right triangle
        result = pixel_distance((0, 0), (3, 4))
        assert math.isclose(result, 5.0)

    def test_float_coords(self):
        result = pixel_distance((0.5, 0.5), (3.5, 4.5))
        assert math.isclose(result, 5.0)


class TestNormaliseDistance:
    def test_basic_normalisation(self):
        # if pixel distance equals fencer height, result should be 1.75 m
        result = normalise_distance(200, 200, real_height_m=1.75)
        assert math.isclose(result, 1.75)

    def test_zero_height_returns_none(self):
        # avoids division by zero
        assert normalise_distance(100, 0) is None

    def test_close_fencers(self):
        # fencers close together — distance less than one body height
        result = normalise_distance(100, 200, real_height_m=1.75)
        assert math.isclose(result, 0.875)

    def test_far_fencers(self):
        # fencers far apart — distance about 2x body height
        result = normalise_distance(400, 200, real_height_m=1.75)
        assert math.isclose(result, 3.5)

    def test_custom_real_height(self):
        # should work with any assumed height
        result = normalise_distance(100, 100, real_height_m=2.0)
        assert math.isclose(result, 2.0)
