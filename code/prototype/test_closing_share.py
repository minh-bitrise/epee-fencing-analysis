"""
Tests for the closing-share resolution analysis.

The point of this module is to make a null result readable, so most of these
check that it does not overclaim: that a balanced series is not reported as
biased, that a genuine bias is recovered, and that the block length tracks the
frame rate rather than being a fixed frame count.

Run with:
    cd code/prototype && python3 -m pytest test_closing_share.py -v
"""

import numpy as np
import pytest

from evaluate_closing_share import (
    BLOCK_S, block_frames_for, block_interval, closing_share,
)


def biased_series(p, n=5000, block=10, seed=0):
    """Autocorrelated movement with a known share of closing blocks."""
    rng = np.random.default_rng(seed)
    runs = rng.random(n // block) < p
    signed = np.repeat(np.where(runs, 1.0, -1.0), block)
    return signed * rng.uniform(0.001, 0.02, len(signed))


class TestClosingShare:
    def test_counts_direction_not_magnitude(self):
        # One large retreat and three small advances is 75 per cent closing.
        # Summing magnitudes would call it net retreat, which is the failure the
        # metric exists to avoid.
        d = np.array([0.01, 0.01, 0.01, -5.0])
        assert closing_share(d) == pytest.approx(0.75)

    def test_ignores_frames_with_no_movement(self):
        d = np.array([0.01, 0.0, 0.0, -0.01])
        assert closing_share(d) == pytest.approx(0.5)

    def test_returns_none_when_nothing_moved(self):
        # Distinct from 50 per cent: no data is not a balanced fencer.
        assert closing_share(np.zeros(100)) is None


class TestBlockLength:
    def test_the_block_tracks_the_frame_rate(self):
        """
        The block has to span the same amount of PLAY on every clip. A fixed
        frame count would preserve a third of a second at 30 fps and a sixth at
        60, which is the same units error found in the touch detector's window.
        """
        slow = [{"time_s": str(i / 30)} for i in range(100)]
        fast = [{"time_s": str(i / 60)} for i in range(100)]
        assert block_frames_for(fast) == pytest.approx(
            2 * block_frames_for(slow), rel=0.2)

    def test_the_block_is_about_a_third_of_a_second(self):
        rows = [{"time_s": str(i / 30)} for i in range(100)]
        assert block_frames_for(rows) == pytest.approx(BLOCK_S * 30, abs=1)

    def test_a_degenerate_series_gets_a_usable_default(self):
        assert block_frames_for([]) >= 3


class TestBlockInterval:
    def test_a_balanced_series_gives_an_interval_spanning_fifty(self):
        lo, hi = block_interval(biased_series(0.50, seed=7), 10, n=400)
        assert lo <= 0.5 <= hi

    def test_a_strong_bias_is_recovered(self):
        # If this failed, a null result on real footage would be unreadable:
        # it could mean balance or blindness.
        lo, hi = block_interval(biased_series(0.75, seed=3), 10, n=400)
        assert lo > 0.5

    def test_the_interval_is_wider_than_an_independent_frame_assumption(self):
        """
        The reason blocks are resampled at all. Treating frames as independent
        gives a standard error under one point on a three-minute clip, which
        would make every measured value look highly significant.
        """
        d = biased_series(0.52, seed=1)
        lo, hi = block_interval(d, 10, n=400)
        moving = d[np.abs(d) > 1e-9]
        naive_se = np.sqrt(0.25 / len(moving))
        assert (hi - lo) / 2 > 3 * naive_se

    def test_too_short_a_series_declines_to_answer(self):
        lo, hi = block_interval(np.array([0.01, -0.01, 0.01]), 10)
        assert lo is None and hi is None
