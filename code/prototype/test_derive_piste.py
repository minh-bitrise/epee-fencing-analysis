"""
Tests for piste-region derivation.

The two functions that decide which detections count as fencers are pure, so
they are tested directly against the geometry that broke them on real footage.
Both defects they encode were found by measurement rather than by reasoning, and
neither would have been caught by a test written before the measurement: the
rules were correct in principle and applied to the wrong population.

These cases therefore use the MEASURED numbers from the evaluation clips rather
than invented ones, so that a change which reintroduces either defect fails here
instead of being discovered on a full processing run.

Run with:
    cd code/prototype && python3 -m pytest test_derive_piste.py -v
"""

import numpy as np
import pytest

from derive_piste import (
    CLUSTER_GAP_HEIGHTS, cluster_1d, select_fencer_pair,
)


def box(x, feet_y, height, width=60):
    """A detection in the pipeline's xyxy form."""
    return np.array([x, feet_y - height, x + width, feet_y], dtype=float)


class TestSelectFencerPair:
    def test_picks_the_two_people_at_the_same_depth(self):
        fencers = [box(400, 440, 250), box(700, 447, 245)]
        assert select_fencer_pair(fencers) is not None

    def test_ignores_the_referee_even_though_he_is_the_tallest(self):
        """The defect that produced a useless polygon on the broadcast clip.

        Taking the two tallest boxes is what `calibrate_fixed_scale` does, and
        it is wrong here: the referee stands nearest the camera, so he is the
        tallest box in the frame.
        """
        referee = box(600, 716, 396)          # nearest the camera, tallest
        fencer_a = box(400, 440, 250)
        fencer_b = box(700, 447, 245)
        pair = select_fencer_pair([referee, fencer_a, fencer_b])
        feet = sorted(float(b[3]) for b in pair)
        assert feet == [440.0, 447.0], "the referee was selected as a fencer"

    def test_does_not_pair_a_fencer_with_a_distant_spectator(self):
        # The height guard. Two people can share a feet-y and still be a fencer
        # and someone standing far behind them.
        fencer = box(400, 440, 250)
        spectator = box(410, 441, 60)
        other_fencer = box(700, 455, 240)
        pair = select_fencer_pair([fencer, spectator, other_fencer])
        assert all(float(b[3] - b[1]) > 100 for b in pair)

    def test_returns_none_when_there_is_nobody_to_pair(self):
        assert select_fencer_pair([]) is None
        assert select_fencer_pair([box(400, 440, 250)]) is None

    def test_returns_none_when_only_one_box_is_tall_enough(self):
        # A lone fencer plus a distant bystander is not a pair, and inventing one
        # would feed a spurious position into the measured band.
        assert select_fencer_pair([box(400, 440, 250), box(100, 300, 30)]) is None


class TestCluster1d:
    def test_splits_on_a_wide_gap(self):
        # Clip 2's real shape: spectators around 210-255, fencers around 400-520.
        values = list(range(207, 256, 4)) + list(range(393, 528, 4))
        groups = cluster_1d(values, CLUSTER_GAP_HEIGHTS * 250)
        assert len(groups) == 2
        assert min(groups[0]) >= 393, "the fencer group should be the largest"

    def test_keeps_a_continuous_range_together(self):
        values = list(range(269, 361, 3))
        assert len(cluster_1d(values, CLUSTER_GAP_HEIGHTS * 102)) == 1

    def test_orders_groups_by_size(self):
        values = [10, 11, 12] + [500, 501, 502, 503, 504]
        groups = cluster_1d(values, 20)
        assert len(groups[0]) == 5 and len(groups[1]) == 3

    def test_handles_an_empty_input(self):
        assert cluster_1d([], 10) == []

    def test_a_percentile_of_everything_would_land_in_the_wrong_group(self):
        """Why clustering is needed at all, stated as an assertion.

        A percentile taken across a bimodal distribution describes neither
        mode.
        """
        spectators = list(range(207, 256, 4))
        fencers = list(range(393, 528, 4)) * 8      # roughly the real proportion
        everything = spectators + fencers

        naive_low = float(np.percentile(everything, 1))
        assert naive_low < 300, "this test no longer reproduces the failure"

        main = cluster_1d(everything, CLUSTER_GAP_HEIGHTS * 250)[0]
        clustered_low = float(np.percentile(main, 1))
        assert clustered_low > 380
