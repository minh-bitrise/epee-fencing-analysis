"""
Unit tests for tempo and exchange metrics.

Run with:
    python3 -m pytest test_tempo.py -v
"""

import math

import pytest

from tempo import compute_tempo, exchanges, load_touches_with_scorer


def touches(*specs):
    """specs are (time_s, scorer) or bare times."""
    out = []
    for s in specs:
        if isinstance(s, tuple):
            out.append({"time_s": float(s[0]), "scorer": s[1]})
        else:
            out.append({"time_s": float(s), "scorer": None})
    return out


# -------------------- exchanges --------------------

class TestExchanges:
    def test_one_exchange_per_touch(self):
        ex = exchanges(touches(20, 50, 90), duration_s=120, reset_s=6.0)
        assert len(ex) == 3

    def test_first_exchange_starts_at_the_recording_start(self):
        ex = exchanges(touches(20), duration_s=120, reset_s=6.0)
        assert ex[0]["start_s"] == 0.0
        assert ex[0]["end_s"] == 20.0
        assert ex[0]["duration_s"] == 20.0

    def test_later_exchanges_start_after_the_reset(self):
        """Play resumes reset_s after the previous touch, not at it."""
        ex = exchanges(touches(20, 50), duration_s=120, reset_s=6.0)
        assert ex[1]["start_s"] == 26.0
        assert ex[1]["duration_s"] == 24.0

    def test_a_touch_inside_the_previous_reset_is_flagged_not_negative(self):
        """
        Observed gaps go as low as 5 s, below the reset, so a touch can fall
        inside the previous reset window. That must not produce a negative
        duration, which would corrupt every aggregate silently.
        """
        ex = exchanges(touches(20, 23), duration_s=120, reset_s=6.0)
        assert ex[1]["duration_s"] == 0.0
        assert ex[1]["overlapping"] is True

    def test_reset_is_clamped_by_the_end_of_the_recording(self):
        ex = exchanges(touches(118), duration_s=120, reset_s=6.0)
        assert ex[0]["end_s"] == 118.0

    def test_scorer_is_carried_through(self):
        ex = exchanges(touches((20, "left")), duration_s=120)
        assert ex[0]["scorer"] == "left"

    def test_no_touches_gives_no_exchanges(self):
        assert exchanges([], duration_s=120) == []


# -------------------- tempo --------------------

class TestComputeTempo:
    def test_no_touches_is_reported_not_crashed(self):
        r = compute_tempo([], duration_s=180)
        assert r["touches"] == 0 and "note" in r

    def test_touch_rate(self):
        r = compute_tempo(touches(30, 60, 90), duration_s=180)
        assert math.isclose(r["touches_per_minute"], 1.0)

    def test_gaps_between_touches(self):
        r = compute_tempo(touches(10, 20, 40), duration_s=180)
        g = r["time_between_touches_s"]
        assert g["min"] == 10.0 and g["max"] == 20.0 and g["mean"] == 15.0

    def test_single_touch_has_no_gaps(self):
        r = compute_tempo(touches(30), duration_s=180)
        assert r["time_between_touches_s"] == {}

    def test_std_of_one_gap_is_zero_not_an_error(self):
        r = compute_tempo(touches(10, 20), duration_s=180)
        assert r["time_between_touches_s"]["std"] == 0.0

    def test_detects_a_slowing_bout(self):
        # gaps: 5, 5, 5, 30, 30  -> second half much longer
        r = compute_tempo(touches(10, 15, 20, 25, 55, 85), duration_s=180)
        assert r["tempo_trend"]["direction"] == "slowing"

    def test_detects_a_quickening_bout(self):
        r = compute_tempo(touches(10, 40, 70, 75, 80, 85), duration_s=180)
        assert r["tempo_trend"]["direction"] == "quickening"

    def test_detects_a_steady_bout(self):
        r = compute_tempo(touches(10, 20, 30, 40, 50, 60), duration_s=180)
        assert r["tempo_trend"]["direction"] == "steady"

    def test_exchange_durations_exclude_overlapping_ones(self):
        """An overlapping exchange has no meaningful duration, so it must not be
        averaged in as a zero."""
        r = compute_tempo(touches(20, 23, 60), duration_s=180, reset_s=6.0)
        assert r["exchange_duration_s"]["min"] > 0


# -------------------- per-scorer metrics --------------------

class TestScorerMetrics:
    def test_counts_scorers_when_supplied(self):
        r = compute_tempo(touches((10, "left"), (20, "right"), (30, "left")),
                          duration_s=180)
        assert r["scorer_counts"] == {"left": 2, "right": 1}

    def test_longest_streak(self):
        r = compute_tempo(touches((10, "left"), (20, "left"), (30, "left"),
                                  (40, "right")), duration_s=180)
        assert r["longest_streak"] == {"scorer": "left", "touches": 3}

    def test_doubles_count_as_their_own_scorer(self):
        r = compute_tempo(touches((10, "double"), (20, "double")), duration_s=180)
        assert r["scorer_counts"] == {"double": 2}
        assert r["longest_streak"]["scorer"] == "double"

    def test_omitted_rather_than_inferred_when_absent(self):
        """
        The detector cannot supply scorers. Testing showed geometry cannot
        either: in epee there is no right-of-way, so a counter-attack scores as
        readily as the attack, and "whichever fencer advanced more" predicts the
        scorer on 53 per cent of single-scorer touches against a 50 per cent
        chance baseline. Guessing would be worse than declining.
        """
        r = compute_tempo(touches(10, 20, 30), duration_s=180)
        assert r["scorer_counts"] is None
        assert "omitted rather than inferred" in r["note"]


# -------------------- file loading --------------------

class TestLoadTouchesWithScorer:
    def test_reads_ground_truth_with_scorers(self, tmp_path):
        f = tmp_path / "gt.csv"
        f.write_text("# comment\ntime_s,scorer,annulled,notes\n"
                     "19,right,0,\"\"\n35,left,0,\"\"\n")
        t = load_touches_with_scorer(str(f))
        assert [x["time_s"] for x in t] == [19.0, 35.0]
        assert [x["scorer"] for x in t] == ["right", "left"]

    def test_detector_output_has_no_scorer(self, tmp_path):
        f = tmp_path / "cand.csv"
        f.write_text("time_s,confidence,signals\n18.9,0.95,approach+separated\n")
        t = load_touches_with_scorer(str(f))
        assert t[0]["scorer"] is None

    def test_sorts_by_time(self, tmp_path):
        f = tmp_path / "gt.csv"
        f.write_text("time_s,scorer,annulled,notes\n50,left,0,\"\"\n10,right,0,\"\"\n")
        t = load_touches_with_scorer(str(f))
        assert [x["time_s"] for x in t] == [10.0, 50.0]
