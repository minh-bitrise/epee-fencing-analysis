"""Tests for measure_framing.py.

The module exists to stop a number in the report going stale, so the thing worth
testing is the one piece of logic the report's claim rests on: splitting missing
frames by gap length. Coverage alone said clip 8 and clip 4 were the same kind of
problem. They are not, and the chapter's argument turns on the split being right.

The YOLO pass is not tested here. It is a wrapper over a model this project does
not own, and a test of it would assert that ultralytics works.
"""

import csv
import os

import pytest

import measure_framing as mf


def rows(pattern):
    """'..x.' -> four frames, the third untracked."""
    return [{"distance_smooth_m": "" if ch == "x" else "1.5"} for ch in pattern]


def test_gap_runs_counts_each_contiguous_run():
    assert mf.gap_runs(rows("..xx...x.")) == [2, 1]


def test_gap_runs_closes_a_run_that_reaches_the_end():
    """The loop appends on the transition back to tracked, so a clip ending in a
    gap has no transition to append on. Without the tail flush the longest gap of
    clip 8, which runs to the final frame, disappeared entirely."""
    assert mf.gap_runs(rows("..xxx")) == [3]


def test_gap_runs_handles_a_clip_that_is_entirely_gap():
    assert mf.gap_runs(rows("xxxx")) == [4]


def test_gap_runs_is_empty_when_nothing_is_missing():
    assert mf.gap_runs(rows("....")) == []


def test_gap_runs_counts_a_leading_gap():
    assert mf.gap_runs(rows("xx..")) == [2]


def test_short_and_long_gaps_are_distinguished_at_the_boundary():
    """One second at 25 fps is 25 frames, and the threshold is inclusive: a gap of
    exactly a second counts as the detector faltering, not as a break in play."""
    fps, thresh = 25.0, mf.SHORT_GAP_S
    runs = [25, 26]
    short = [r for r in runs if r <= thresh * fps]
    assert short == [25]


def test_the_two_failure_modes_separate():
    """The report's claim in one test.

    Many short gaps and few long ones are different defects with the same
    coverage. Built so both patterns lose the same number of frames: only the
    split tells them apart.
    """
    fps = 25.0
    flicker = mf.gap_runs(rows(("....x" * 20)))          # 20 gaps of 1 frame
    absence = mf.gap_runs(rows("." * 80 + "x" * 20))     # 1 gap of 20 frames
    assert sum(flicker) == sum(absence) == 20            # identical coverage

    def short(runs):
        return sum(r for r in runs if r <= mf.SHORT_GAP_S * fps)

    assert short(flicker) == 20
    assert short(absence) == 20, "a 20-frame gap is under a second at 25 fps"

    # At a lower frame rate the same 20-frame gap is a break, not flicker.
    assert sum(r for r in absence if r <= mf.SHORT_GAP_S * 10) == 0


def test_output_columns_are_the_ones_the_figures_read(tmp_path):
    """make_report_figures.py reads framing.csv by column name. Renaming a column
    here would leave the figures reading a KeyError, or worse a stale file."""
    needed = {"clip", "fencer_px", "coverage_pct", "short_gap_pct"}
    path = os.path.join(os.path.dirname(os.path.abspath(mf.__file__)),
                        "results_current", "framing.csv")
    if not os.path.exists(path):
        pytest.skip("framing.csv not present; run measure_framing.py")
    with open(path) as f:
        got = set(next(csv.reader(f)))
    assert needed <= got, f"framing.csv is missing {needed - got}"
