"""
Tests for per-bout lunge calibration.

The pure parts. Whether the stance ratio actually marks lunges is not something a
unit test can establish, and this project has been caught by that distinction
repeatedly, so it is answered separately by `detect_lunges.py --evaluate` against
the hand-labelled lunges on clips 2 and 3.

Run with:
    cd code/prototype && python3 -m pytest test_lunges.py -v
"""

import json

import numpy as np
import pytest

from detect_lunges import (
    MIN_CALIBRATION_LUNGES, calibrate, evaluate, load_confirmed,
    peak_in_window, propose, stance_ratio_series,
)


@pytest.fixture
def flat_series():
    """Ten seconds of ordinary play at ratio 1.0, with spikes at 2, 4, 6, 8."""
    t = np.arange(0, 10, 0.05)
    x = np.full_like(t, 1.0)
    for spike in (2.0, 4.0, 6.0, 8.0):
        x[np.abs(t - spike) < 0.1] = 2.5
    return t, x


class TestPeakInWindow:
    def test_finds_the_spike(self, flat_series):
        t, x = flat_series
        assert peak_in_window(t, x, 2.0) == pytest.approx(2.5)

    def test_returns_the_quiet_level_away_from_a_spike(self, flat_series):
        t, x = flat_series
        assert peak_in_window(t, x, 5.0) == pytest.approx(1.0)

    def test_returns_none_where_there_is_no_pose(self, flat_series):
        # Pose runs on a stride, so gaps are the sampling. A window containing no
        # samples has no answer, and inventing one would invent a posture.
        t, x = flat_series
        assert peak_in_window(t, x, 100.0) is None


class TestCalibrate:
    def test_fits_below_the_quietest_confirmed_lunge(self, flat_series):
        t, x = flat_series
        th, used = calibrate(t, x, [2.0, 4.0, 6.0, 8.0, 2.0])
        assert used == 5
        assert 1.0 < th <= 2.5

    def test_refuses_to_fit_on_too_few(self, flat_series):
        """
        A threshold fitted on two examples fires on a quarter of the bout, which
        is worse for the user than no proposals at all: they then have to reject
        every one.
        """
        t, x = flat_series
        th, used = calibrate(t, x, [2.0, 4.0])
        assert th is None and used == 2

    def test_the_threshold_is_specific_to_the_bout(self):
        """
        The whole point. Measured, clip 3 calibrates to 1.90 and clip 2 to 2.71,
        a 42 per cent difference, which is why a threshold carried between
        cameras reaches F1 0.22 where a per-bout one reaches 0.74.
        """
        t = np.arange(0, 10, 0.05)
        quiet = np.full_like(t, 1.0)
        loud = np.full_like(t, 1.0)
        for s in (2.0, 4.0, 6.0, 8.0, 9.0):
            quiet[np.abs(t - s) < 0.1] = 2.0
            loud[np.abs(t - s) < 0.1] = 4.0
        th_quiet, _ = calibrate(t, quiet, [2.0, 4.0, 6.0, 8.0, 9.0])
        th_loud, _ = calibrate(t, loud, [2.0, 4.0, 6.0, 8.0, 9.0])
        assert th_loud > th_quiet * 1.5


class TestPropose:
    def test_proposes_the_spikes_and_not_the_quiet(self, flat_series):
        t, x = flat_series
        got = propose(t, x, threshold=2.0)
        assert [round(p["time_s"]) for p in got] == [2, 4, 6, 8]

    def test_nothing_is_proposed_without_a_threshold(self, flat_series):
        # calibrate() returns None when it cannot fit, and that must propagate
        # as "no proposals" rather than as an exception or as everything.
        t, x = flat_series
        assert propose(t, x, threshold=None) == []

    def test_the_calibration_stretch_can_be_suppressed(self, flat_series):
        # The user should not be offered back the lunges they just confirmed.
        t, x = flat_series
        got = propose(t, x, threshold=2.0, skip_before=5.0)
        assert [round(p["time_s"]) for p in got] == [6, 8]

    def test_margin_says_how_far_above_the_bar(self, flat_series):
        t, x = flat_series
        got = propose(t, x, threshold=1.25)
        assert got[0]["margin"] == pytest.approx(2.0, abs=0.01)


class TestEvaluate:
    def test_calibration_lunges_never_score_themselves(self, flat_series):
        """
        The protocol that makes the number mean anything. An earlier version of
        this experiment fitted on all the lunges and scored the same ones,
        reporting F1 0.82 where the honest figure is 0.74.
        """
        t = np.arange(0, 60, 0.05)
        x = np.full_like(t, 1.0)
        spikes = [2.0 * i + 2 for i in range(12)]
        for s in spikes:
            x[np.abs(t - s) < 0.1] = 2.5
        result = evaluate(t, x, spikes, k=MIN_CALIBRATION_LUNGES)
        assert result is not None
        assert result["calibrated_on"] == MIN_CALIBRATION_LUNGES
        # only lunges after the calibration stretch are scored
        assert result["test_lunges"] < len(spikes) - MIN_CALIBRATION_LUNGES + 1

    def test_returns_none_when_there_is_not_enough_to_split(self, flat_series):
        t, x = flat_series
        assert evaluate(t, x, [2.0, 4.0]) is None


class TestStanceRatioSeries:
    def test_reads_the_ratio_and_skips_a_failed_pose(self, tmp_path):
        """
        Hip height near zero is a pose failure, not a measurement. Dividing by it
        yields an enormous ratio that would clear any threshold and be proposed
        as the most confident lunge in the bout.
        """
        p = tmp_path / "d.csv"
        p.write_text(
            "time_s,f2_stance_m,f2_hip_height_m\n"
            "0.0,1.0,1.0\n"
            "0.1,2.0,0.01\n"      # pose failure, must be dropped
            "0.2,1.5,1.0\n"
            "0.3,,\n")            # no pose at all
        t, x = stance_ratio_series(str(p), slot=1)
        assert list(t) == [0.0, 0.2]
        assert list(x) == [1.0, 1.5]


def test_load_confirmed_filters_by_fencer(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"lunges": [
        {"time_s": 3.0, "slot": 0}, {"time_s": 1.0, "slot": 1},
        {"time_s": 2.0, "slot": 1}]}))
    assert load_confirmed(str(p), 1) == [1.0, 2.0]
    assert load_confirmed(str(p), 0) == [3.0]
