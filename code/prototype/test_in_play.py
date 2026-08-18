"""
Unit tests for in-play segment scoping.

Run with:
    python3 -m pytest test_in_play.py -v
"""

import math

import numpy as np
import pytest

from in_play import (
    DEFAULT_RESET_S,
    in_play_fraction,
    in_play_mask,
    load_touch_times,
    out_of_play_windows,
    scope_cumulative,
    scope_distance,
    scope_metrics,
)


# -------------------- window construction --------------------

class TestOutOfPlayWindows:
    def test_one_window_per_touch(self):
        w = out_of_play_windows([10.0, 30.0, 50.0], end_time=100.0, reset_s=6.0)
        assert len(w) == 3

    def test_window_starts_at_the_touch_and_runs_for_reset_s(self):
        w = out_of_play_windows([10.0], end_time=100.0, reset_s=6.0)
        assert w[0] == (10.0, 16.0)

    def test_window_is_truncated_by_the_next_touch(self):
        """Observed touch-to-touch gaps go as low as 5 s, below the reset."""
        w = out_of_play_windows([10.0, 13.0], end_time=100.0, reset_s=6.0)
        assert w[0] == (10.0, 13.0)      # cut short by the next touch
        assert w[1] == (13.0, 19.0)

    def test_window_is_truncated_by_the_end_of_the_recording(self):
        w = out_of_play_windows([98.0], end_time=100.0, reset_s=6.0)
        assert w[0] == (98.0, 100.0)

    def test_touch_at_the_very_end_yields_no_window(self):
        w = out_of_play_windows([100.0], end_time=100.0, reset_s=6.0)
        assert w == []

    def test_unsorted_input_is_handled(self):
        w = out_of_play_windows([50.0, 10.0, 30.0], end_time=100.0, reset_s=6.0)
        assert [start for start, _ in w] == [10.0, 30.0, 50.0]

    def test_no_touches_gives_no_windows(self):
        assert out_of_play_windows([], end_time=100.0) == []

    def test_default_reset_matches_the_measured_value(self):
        """
        DEFAULT_RESET_S is derived from the distance profile aligned on 14
        labelled touches, where separation peaks at +2 to +3 s and returns to
        the phrase baseline by about +7 s. Guard against it drifting back to a
        guessed value.
        """
        assert 4.0 <= DEFAULT_RESET_S <= 8.0


# -------------------- masking --------------------

class TestInPlayMask:
    def _t(self):
        return np.arange(0.0, 20.0, 0.5)

    def test_frames_inside_a_window_are_excluded(self):
        t = self._t()
        mask = in_play_mask(t, [(5.0, 11.0)])
        assert not mask[(t >= 5.0) & (t < 11.0)].any()

    def test_frames_outside_a_window_are_kept(self):
        t = self._t()
        mask = in_play_mask(t, [(5.0, 11.0)])
        assert mask[t < 5.0].all()
        assert mask[t >= 11.0].all()

    def test_window_start_is_inclusive_and_end_exclusive(self):
        t = np.array([4.9, 5.0, 10.9, 11.0])
        mask = in_play_mask(t, [(5.0, 11.0)])
        assert list(mask) == [True, False, False, True]

    def test_no_windows_keeps_everything(self):
        t = self._t()
        assert in_play_mask(t, []).all()

    def test_overlapping_windows_are_handled(self):
        t = self._t()
        mask = in_play_mask(t, [(5.0, 10.0), (8.0, 12.0)])
        assert not mask[(t >= 5.0) & (t < 12.0)].any()

    def test_fraction_is_a_share(self):
        t = np.arange(0.0, 10.0, 1.0)          # 10 frames
        f = in_play_fraction(t, [(0.0, 5.0)])  # excludes 5 of them
        assert math.isclose(f, 0.5)

    def test_fraction_of_empty_series(self):
        assert in_play_fraction(np.array([]), []) == 0.0


# -------------------- distance scoping --------------------

class TestScopeDistance:
    def test_selects_only_in_play_samples(self):
        d = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        assert list(scope_distance(d, mask)) == [1.0, 3.0]

    def test_drops_nan_gaps(self):
        d = np.array([1.0, np.nan, 3.0])
        mask = np.array([True, True, True])
        assert list(scope_distance(d, mask)) == [1.0, 3.0]

    def test_all_excluded_gives_empty(self):
        d = np.array([1.0, 2.0])
        assert len(scope_distance(d, np.array([False, False]))) == 0


# -------------------- cumulative rescoping --------------------

class TestScopeCumulative:
    """
    Push and pull are stored as running totals, so an in-play total requires
    differencing first and summing only the in-play deltas. Slicing the
    cumulative series would silently include reset movement.
    """

    def test_full_mask_recovers_the_total(self):
        series = [0.0, 1.0, 3.0, 6.0]
        t = np.arange(4.0)
        total = scope_cumulative(series, t, np.ones(4, dtype=bool))
        assert math.isclose(total, 6.0)

    def test_excluded_frames_drop_their_movement(self):
        # deltas are 0, 1, 2, 3; excluding the third frame drops its 2
        series = [0.0, 1.0, 3.0, 6.0]
        t = np.arange(4.0)
        mask = np.array([True, True, False, True])
        assert math.isclose(scope_cumulative(series, t, mask), 4.0)

    def test_a_flat_reset_contributes_nothing(self):
        # no movement during the excluded stretch, so the total is unchanged
        series = [0.0, 2.0, 2.0, 2.0, 5.0]
        t = np.arange(5.0)
        mask = np.array([True, True, False, False, True])
        assert math.isclose(scope_cumulative(series, t, mask), 5.0)

    def test_single_sample_is_zero(self):
        assert scope_cumulative([4.0], np.array([0.0]), np.array([True])) == 0.0

    def test_empty_series_is_zero(self):
        assert scope_cumulative([], np.array([]), np.array([], dtype=bool)) == 0.0


# -------------------- end to end over CSV rows --------------------

def make_rows(n=40, dt=0.5):
    """Synthetic bout: distance 2.0 throughout, each fencer moving 0.1 m/frame."""
    rows = []
    for i in range(n):
        rows.append({
            "frame": str(i), "time_s": f"{i * dt}",
            "distance_raw_m": "2.0", "distance_smooth_m": "2.0", "method": "pose",
            "f1_advance_m": f"{i * 0.1:.4f}", "f1_retreat_m": "0.0",
            "f2_advance_m": f"{i * 0.2:.4f}", "f2_retreat_m": "0.0",
        })
    return rows


class TestScopeMetrics:
    def test_no_touches_leaves_metrics_unchanged(self):
        rows = make_rows()
        r = scope_metrics(rows, [], reset_s=6.0)
        assert r["in_play_fraction"] == 1.0
        assert r["in_play"]["f1_push_m"] == r["whole_recording"]["f1_push_m"]

    def test_a_touch_reduces_the_in_play_share(self):
        rows = make_rows()
        r = scope_metrics(rows, [5.0], reset_s=6.0)
        assert r["in_play_fraction"] < 1.0

    def test_scoped_push_is_lower_than_whole_recording(self):
        rows = make_rows()
        r = scope_metrics(rows, [5.0], reset_s=6.0)
        assert r["in_play"]["f1_push_m"] < r["whole_recording"]["f1_push_m"]

    def test_distance_unchanged_when_it_is_constant(self):
        """Scoping changes which frames count, not the distance values."""
        rows = make_rows()
        r = scope_metrics(rows, [5.0], reset_s=6.0)
        assert math.isclose(r["in_play"]["mean_distance_m"], 2.0)

    def test_reports_window_count(self):
        rows = make_rows()
        r = scope_metrics(rows, [3.0, 12.0], reset_s=6.0)
        assert r["out_of_play_windows"] == 2
        assert r["touches"] == 2


# -------------------- touch file loading --------------------

class TestLoadTouchTimes:
    def test_reads_ground_truth_with_comments(self, tmp_path):
        f = tmp_path / "gt.csv"
        f.write_text(
            "# a comment line that must be skipped\n"
            "time_s,scorer,annulled,notes\n"
            "19,right,0,\"clanky\"\n"
            "35,left,0,\"\"\n"
        )
        assert load_touch_times(str(f)) == [19.0, 35.0]

    def test_reads_detector_output(self, tmp_path):
        f = tmp_path / "cand.csv"
        f.write_text(
            "time_s,duration_s,confidence,min_distance_m,separation_m,"
            "audio_strength,sustained,in_distance,closing,separated,isolated,signals\n"
            "18.9,0.2,0.95,0.63,1.47,0.9,1,1,1,1,1,sustained\n"
        )
        assert load_touch_times(str(f)) == [18.9]


# -------------------- provenance --------------------

class TestTouchProvenance:
    """
    Provenance is inferred from columns, not filenames, because the summary
    stage must tell the reader whether a touch count is confirmed or
    approximate. An earlier version passed only the file's basename, which lost
    the directory and led the model to describe hand-labelled touches as
    detector output.
    """

    def test_ground_truth_is_human_confirmed(self, tmp_path):
        from in_play import touch_provenance
        f = tmp_path / "gt.csv"
        f.write_text("# note\ntime_s,scorer,annulled,notes\n19,right,0,\"\"\n")
        assert touch_provenance(str(f)) == "human_confirmed"

    def test_detector_output_is_automatic(self, tmp_path):
        from in_play import touch_provenance
        f = tmp_path / "cand.csv"
        f.write_text("time_s,confidence,signals\n18.9,0.95,sustained\n")
        assert touch_provenance(str(f)) == "automatic_detector"

    def test_unrecognised_schema_is_unknown(self, tmp_path):
        from in_play import touch_provenance
        f = tmp_path / "other.csv"
        f.write_text("time_s,whatever\n1.0,x\n")
        assert touch_provenance(str(f)) == "unknown"


# -------------------- data-quality warning --------------------

class TestNetDisplacementWarning:
    """
    Net forward displacement over a bout should be near zero, since fencers
    reset between touches. A large value means uncorrected camera motion, and
    the payload must flag it: supplying it unflagged led the model to report an
    impossible +21.88 m as decisive aggression.
    """

    def _rows(self, f2_push, f2_pull, n=40):
        rows = []
        for i in range(n):
            frac = i / (n - 1)
            rows.append({
                "frame": str(i), "time_s": f"{i * 0.5}",
                "distance_raw_m": "2.0", "distance_smooth_m": "2.0", "method": "pose",
                "f1_advance_m": f"{frac * 10:.4f}", "f1_retreat_m": f"{frac * 10:.4f}",
                "f2_advance_m": f"{frac * f2_push:.4f}",
                "f2_retreat_m": f"{frac * f2_pull:.4f}",
            })
        return rows

    def test_large_net_displacement_is_flagged(self):
        from generate_summary import add_in_play_scope, compute_stats
        import tempfile, os
        rows = self._rows(f2_push=40.0, f2_pull=10.0)     # net +30 m, impossible
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("time_s,scorer,annulled,notes\n5,right,0,\"\"\n")
            path = f.name
        try:
            stats = add_in_play_scope(compute_stats(rows), rows, path)
            assert "data_quality_warnings" in stats
            assert "net_displacement_m" in stats["data_quality_warnings"][0]
        finally:
            os.unlink(path)

    def test_plausible_net_displacement_is_not_flagged(self):
        from generate_summary import add_in_play_scope, compute_stats
        import tempfile, os
        rows = self._rows(f2_push=12.0, f2_pull=11.0)     # net +1 m, plausible
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("time_s,scorer,annulled,notes\n5,right,0,\"\"\n")
            path = f.name
        try:
            stats = add_in_play_scope(compute_stats(rows), rows, path)
            assert "data_quality_warnings" not in stats
        finally:
            os.unlink(path)

    def test_in_play_block_carries_derived_figures(self):
        """
        in_play_only must include the reliable movement figures, or the model
        falls back to the whole-recording values, which is the bug that produced
        the impossible reading.
        """
        from generate_summary import add_in_play_scope, compute_stats
        import tempfile, os
        rows = self._rows(f2_push=12.0, f2_pull=11.0)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("time_s,scorer,annulled,notes\n5,right,0,\"\"\n")
            path = f.name
        try:
            stats = add_in_play_scope(compute_stats(rows), rows, path)
            for fencer in ("fencer_1", "fencer_2"):
                block = stats["in_play_only"][fencer]
                assert set(block) == {"net_forward_movement_m", "closing_share_pct"}
                # the scoped figure is not an endpoint measurement, so it must
                # not be named as a displacement anywhere in the payload
                assert "net_displacement_m" not in block
        finally:
            os.unlink(path)
