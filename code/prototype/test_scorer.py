"""
Tests for lamp-based touch attribution.

The pure parts only: colour counting, threshold fitting and classification.
Whether the lamps actually carry the signal is not a question a unit test can
answer, and this project has been caught by that distinction four times, so it is
answered separately by running `detect_scorer.py --evaluate` against the
hand-labelled scorer column on all four clips.

Run with:
    cd code/prototype && python3 -m pytest test_scorer.py -v
"""

import numpy as np
import pytest

from detect_scorer import classify, fit_thresholds, lamp_counts


def patch(colour, size=40, shape=(120, 160)):
    """A frame that is dark grey except for one saturated patch."""
    frame = np.full((*shape, 3), 40, dtype=np.uint8)
    bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}[colour]
    frame[10:10 + size, 10:10 + size] = bgr
    return frame


class TestLampCounts:
    def test_finds_a_red_patch(self):
        red, green = lamp_counts(patch("red"))
        assert red == 40 * 40 and green == 0

    def test_finds_a_green_patch(self):
        red, green = lamp_counts(patch("green"))
        assert green == 40 * 40 and red == 0

    def test_ignores_colours_that_are_not_lamps(self):
        assert lamp_counts(patch("blue")) == (0, 0)

    def test_ignores_a_dull_patch(self):
        """
        A lamp is bright and saturated. Fencing whites, skin and pale wood are
        none of those, and counting them would swamp the signal.
        """
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        frame[10:50, 10:50] = (60, 60, 120)      # muted red, low saturation
        assert lamp_counts(frame) == (0, 0)


class TestFitThresholds:
    def _responses(self, pairs):
        return [{"time_s": float(i), "red": r, "green": g}
                for i, (r, g) in enumerate(pairs)]

    def test_splits_between_the_on_and_off_groups(self):
        r = self._responses([(100, 0), (0, 100), (100, 100)])
        th = fit_thresholds(r, ["left", "right", "double"], green_is="right")
        assert 0 < th["green"] < 100
        assert 0 < th["red"] < 100

    def test_a_colour_never_seen_off_still_gets_a_usable_threshold(self):
        """
        With three or four confirmed touches a lamp may only ever be seen firing.
        A threshold of zero would then call every later frame a firing, so it
        falls back to half the smallest observed value instead.
        """
        r = self._responses([(0, 80), (0, 120)])
        th = fit_thresholds(r, ["right", "right"], green_is="right")
        assert 0 < th["green"] <= 40

    def test_the_colour_to_side_mapping_is_respected(self):
        # Nothing in the image says which fencer the green lamp belongs to, so
        # the caller supplies it and the fit must follow.
        r = self._responses([(100, 0), (0, 100)])
        th = fit_thresholds(r, ["right", "left"], green_is="left")
        assert th["green_is"] == "left"
        assert 0 < th["green"] < 100


class TestClassify:
    TH = {"red": 50.0, "green": 50.0, "green_is": "right"}

    def test_green_alone_is_the_green_fencer(self):
        assert classify({"red": 0, "green": 200}, self.TH)["scorer"] == "right"

    def test_red_alone_is_the_other_fencer(self):
        assert classify({"red": 200, "green": 0}, self.TH)["scorer"] == "left"

    def test_both_lamps_is_a_double(self):
        # Legal and common in epee, and the reason attribution cannot be a
        # two-way choice.
        assert classify({"red": 200, "green": 200}, self.TH)["scorer"] == "double"

    def test_neither_lamp_is_reported_as_unknown_not_guessed(self):
        """
        A confident wrong answer costs the user more than a blank: they have to
        notice it is wrong before they can fix it, whereas a blank asks them the
        question they were going to answer anyway.
        """
        out = classify({"red": 0, "green": 0}, self.TH)
        assert out["scorer"] == "unknown" and out["confidence"] == 0.0

    def test_confidence_rises_with_the_margin(self):
        near = classify({"red": 0, "green": 55}, self.TH)["confidence"]
        far = classify({"red": 0, "green": 500}, self.TH)["confidence"]
        assert far > near

    def test_the_mapping_can_be_inverted(self):
        th = dict(self.TH, green_is="left")
        assert classify({"red": 0, "green": 200}, th)["scorer"] == "left"
        assert classify({"red": 200, "green": 0}, th)["scorer"] == "right"
