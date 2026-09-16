"""
Tests for the fencer profile.

The profile's risk is not that it crashes. It is that it draws a confident,
professional-looking shape out of data that does not support one, which is the
failure mode this project has already hit four times: a metric that passed its
tests while measuring something other than what it claimed. So most of these
tests are about the profile REFUSING, or reporting absence as absence.
"""
import csv
import math

import pytest

import fencer_profile as fp


def write_csv(tmp_path, rows, name="m.csv"):
    path = tmp_path / name
    cols = ["frame", "time_s", "distance_raw_m", "distance_smooth_m", "method",
            "f1_pos_m", "f2_pos_m"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return str(path)


# Footwork by default. A bout where neither fencer moves is refused outright, and
# it should be: every axis assumes fencing done on the feet. The earlier fixtures
# held both positions constant, which is the shape of a wheelchair bout rather
# than of the clean foot-fencing bout they were standing in for.
def _walk(base, amp=0.6):
    return lambda i: base + amp * math.sin(i / 30.0)


def bout(n=400, f1=_walk(2.0), f2=_walk(6.0), dist=lambda i: 4.0):
    """A clean bout: fencer 1 on the left throughout, fencer 2 on the right."""
    return [{"frame": i, "time_s": round(i * 0.1, 2),
             "distance_raw_m": dist(i), "distance_smooth_m": dist(i),
             "method": "pose", "f1_pos_m": f1(i), "f2_pos_m": f2(i)}
            for i in range(n)]


class TestRefusal:
    def test_refuses_when_the_tracker_swapped_the_fencers(self, tmp_path):
        """
        The test that matters most. Clip 4 swapped 14 times, and a profile there
        would describe the tracker while looking exactly as plausible as a real
        one. A radar with a caveat still gets read as a radar, so it is withheld.
        """
        rows = bout(f1=lambda i: 2.0 if i < 200 else 6.0,
                    f2=lambda i: 6.0 if i < 200 else 2.0)
        out = fp.build(write_csv(tmp_path, rows), [])
        assert out["available"] is False
        assert out["swaps"] >= 1
        assert "tracking" in out["reason"]

    def test_refuses_when_almost_nothing_was_tracked(self, tmp_path):
        rows = bout(n=400)
        for r in rows[5:]:
            r["f1_pos_m"] = r["f2_pos_m"] = ""
        out = fp.build(write_csv(tmp_path, rows), [])
        assert out["available"] is False
        assert "too few frames" in out["reason"]

    def test_refuses_on_an_empty_bout(self, tmp_path):
        out = fp.build(write_csv(tmp_path, []), [])
        assert out["available"] is False

    def test_a_clean_bout_is_profiled(self, tmp_path):
        # The counterpart: the refusals above must not be firing on good footage.
        rows = bout(f1=lambda i: 2.0 + 0.4 * math.sin(i / 20.0),
                    f2=lambda i: 6.0 + 0.4 * math.cos(i / 20.0))
        out = fp.build(write_csv(tmp_path, rows), [])
        assert out["available"] is True
        assert len(out["axes"]) == 6


class TestAbsentVersusZero:
    def test_an_axis_with_no_data_scores_none_rather_than_zero(self, tmp_path):
        """
        A fencer with no confirmed lunges has not been measured lunging rarely;
        they have not been measured at all. Scoring that as 0 would draw a
        collapsed spoke, which reads as a finding.
        """
        out = fp.build(write_csv(tmp_path, bout()), [])
        lunge = next(a for a in out["axes"] if a["key"] == "lunge_rate_per_min")
        assert lunge["fencer_1"]["value"] is None
        assert lunge["fencer_1"]["score"] is None

    def test_no_touches_means_no_best_run_rather_than_a_run_of_zero(self, tmp_path):
        """
        Found by looking at the panel on a real bout rather than by any test
        here. Every axis was individually correct and the picture still said
        something false: two fencers drawn tied on "best run" reads as a finding
        about them, not as a statement that nothing has been reviewed yet.
        """
        out = fp.build(write_csv(tmp_path, bout()), [])
        run = next(a for a in out["axes"] if a["key"] == "longest_streak")
        assert run["fencer_1"]["value"] is None
        assert run["fencer_1"]["score"] is None

    def test_lunges_are_scored_once_supplied(self, tmp_path):
        out = fp.build(write_csv(tmp_path, bout()), [],
                       lunges={1: [1.0, 2.0, 3.0], 2: [4.0]})
        lunge = next(a for a in out["axes"] if a["key"] == "lunge_rate_per_min")
        assert lunge["fencer_1"]["score"] == 75.0
        assert lunge["fencer_2"]["score"] == 25.0


class TestTerritory:
    def test_measured_from_each_fencers_own_end(self, tmp_path):
        """
        Both fencers pressing forward should both score as having taken ground,
        rather than the one with the larger raw coordinate always winning. A raw
        mean position would make the right-hand fencer look dominant in every
        bout ever recorded.
        """
        # Fencer 1 sits 3 m up from the left end; fencer 2 sits 1 m down from the
        # right end. Fencer 1 has taken more ground despite the smaller position.
        # Frame 0 establishes the observed extent of the strip; both fencers walk
        # thereafter, since a bout with no footwork is refused before the axes
        # are computed at all.
        rows = bout(f1=lambda i: 0.0 if i == 0 else 3.0 + 0.4 * math.sin(i / 30.0),
                    f2=lambda i: 8.0 if i == 0 else 7.0 + 0.4 * math.sin(i / 30.0))
        out = fp.build(write_csv(tmp_path, rows), [])
        terr = next(a for a in out["axes"] if a["key"] == "territory_m")
        assert terr["fencer_1"]["score"] > terr["fencer_2"]["score"]


class TestGroundUsed:
    def test_a_single_dropout_does_not_define_the_range(self, tmp_path):
        """
        Why the interquartile range and not the full range. One frame where
        tracking jumped to the far end of the piste would set a full-range figure
        entirely by itself, and that frame is the least trustworthy in the bout.

        The spike stays on fencer 1's own side of the piste on purpose: a jump
        past the opponent is caught earlier and more bluntly, by the side-swap
        refusal. This is the subtler case that no other guard sees.
        """
        steady = bout(f1=lambda i: 2.0 + (i % 4) * 0.1)
        spiked = [dict(r) for r in steady]
        spiked[100]["f1_pos_m"] = 5.5

        a = fp.build(write_csv(tmp_path, steady, "a.csv"), [])
        b = fp.build(write_csv(tmp_path, spiked, "b.csv"), [])
        used = lambda o: next(x for x in o["axes"]
                              if x["key"] == "range_used_m")["fencer_1"]["value"]
        assert used(a) == used(b)


class TestTouchAxes:
    def test_a_double_counts_half_to_each_fencer(self, tmp_path):
        touches = [{"time_s": 5.0, "scorer": "left"},
                   {"time_s": 10.0, "scorer": "double"}]
        out = fp.build(write_csv(tmp_path, bout()), touches)
        sc = next(a for a in out["axes"] if a["key"] == "scoring_share_pct")
        assert sc["fencer_1"]["value"] == 75.0
        assert sc["fencer_2"]["value"] == 25.0

    def test_no_scored_touches_is_absence_not_a_streak_of_zero(self):
        assert fp.longest_streak([], "left") is None

    def test_a_fencer_who_scored_none_has_a_streak_of_zero(self):
        # Distinct from the case above: touches exist, this fencer scored none.
        # That IS a measurement, and it should read as one.
        assert fp.longest_streak(["right", "right"], "left") == 0

    def test_a_double_breaks_a_streak(self):
        # Two touches either side of a double is not a run of four.
        assert fp.longest_streak(
            ["left", "left", "double", "left", "left"], "left") == 2

    def test_streak_counts_consecutive_touches(self):
        assert fp.longest_streak(["left", "right", "left", "left", "left"],
                                 "left") == 3

    def test_scoring_range_uses_the_distance_when_they_scored(self, tmp_path):
        # Fencer 1's two touches land where the distance reads 1.0 m; the bout's
        # mean distance is far higher, so the axis must not be reading that.
        rows = bout(dist=lambda i: 1.0 if i in (50, 100) else 5.0)
        touches = [{"time_s": 5.0, "scorer": "left"},
                   {"time_s": 10.0, "scorer": "left"}]
        out = fp.build(write_csv(tmp_path, rows), touches)
        rng = next(a for a in out["axes"] if a["key"] == "scoring_range_m")
        assert rng["fencer_1"]["value"] == pytest.approx(1.0)

    def test_a_touch_with_no_nearby_frame_is_skipped(self, tmp_path):
        # A touch labelled beyond the tracked footage must not silently attach
        # itself to the last frame that happens to exist.
        assert fp.distance_at(bout(n=100), 900.0) is None


class TestScoping:
    def test_confirmed_touches_scope_out_the_resets(self, tmp_path):
        out = fp.build(write_csv(tmp_path, bout()),
                       [{"time_s": 20.0, "scorer": "left"}])
        assert "resets" in out["scoping"]

    def test_without_touches_it_says_it_used_everything(self, tmp_path):
        out = fp.build(write_csv(tmp_path, bout()), [])
        assert "whole recording" in out["scoping"]


class TestShare:
    def test_parity_scores_fifty(self):
        assert fp._share(4.0, 4.0) == 50.0

    def test_a_missing_half_yields_no_score(self):
        assert fp._share(4.0, None) is None

    def test_two_zeroes_are_parity_not_a_division_error(self):
        assert fp._share(0.0, 0.0) == 50.0


class TestScoreProgression:
    """
    The scoreline. Derived entirely from confirmed touches, so it is the one part
    of the profile that survives a tracking failure.
    """

    def t(self, time_s, scorer):
        return {"time_s": time_s, "scorer": scorer}

    def test_a_lead_changing_through_level_counts_as_a_change(self):
        """
        The defect this replaced. A lead almost always changes hands by passing
        through level, so comparing only against the CURRENT leader counts no
        change at all: the sequence is 1, None, 2 and neither step is a swap
        between two fencers. On clip 3 this reported zero lead changes for a bout
        where one fencer led 87 seconds and the other 38.
        """
        r = fp.score_progression(
            [self.t(10, "left"), self.t(20, "right"), self.t(30, "right")], 60)
        assert r["lead_changes"] == 1

    def test_no_change_when_one_fencer_leads_throughout(self):
        r = fp.score_progression(
            [self.t(10, "left"), self.t(20, "left"), self.t(30, "right")], 60)
        assert r["lead_changes"] == 0

    def test_a_double_advances_both_scores(self):
        # Correct epee behaviour, and not a special case: at 4-4 a double is 5-5.
        r = fp.score_progression([self.t(10, "double")], 60)
        assert r["final"] == {"fencer_1": 1, "fencer_2": 1}

    def test_a_double_can_end_a_lead_without_anyone_scoring_past(self):
        r = fp.score_progression([self.t(10, "left"), self.t(20, "double")], 60)
        assert r["final"] == {"fencer_1": 2, "fencer_2": 1}
        assert r["lead_changes"] == 0

    def test_time_leading_runs_to_the_end_of_the_bout(self):
        # The stretch after the last touch belongs to whoever finished ahead.
        r = fp.score_progression([self.t(10, "left")], 60)
        assert r["time_leading_s"][1] == 50.0

    def test_time_before_the_first_touch_is_level_not_led(self):
        r = fp.score_progression([self.t(10, "left")], 60)
        assert r["level_s"] == 10.0

    def test_an_unattributed_touch_advances_neither_score(self):
        # It is an unknown event, not a nil-nil one, and inventing a scorer would
        # put a fabricated scoreline in front of the user.
        r = fp.score_progression([self.t(10, "unknown"), self.t(20, "left")], 60)
        assert r["final"] == {"fencer_1": 1, "fencer_2": 0}
        assert r["unattributed"] == 1

    def test_touches_out_of_order_are_sorted_first(self):
        r = fp.score_progression([self.t(30, "right"), self.t(10, "left")], 60)
        assert [x["time_s"] for x in r["timeline"]] == [10, 30]

    def test_no_touches_means_no_scoreline(self):
        assert fp.score_progression([], 60)["available"] is False


class TestPisteZones:
    def test_both_fencers_are_measured_from_their_own_end(self, tmp_path):
        """
        Otherwise "the far third" means opposite ends of the piste for the two
        fencers, and the two rows cannot be compared at all.
        """
        rows = bout(n=100, f1=lambda i: 1.0, f2=lambda i: 9.0)
        out = fp.piste_zones(rows, [{"time_s": 1.0, "scorer": "left"},
                                    {"time_s": 1.0, "scorer": "right"}])
        # Each scored while standing at their own end, so each lands in bucket 0.
        assert out["fencer_1"][0] == 1
        assert out["fencer_2"][0] == 1

    def test_an_unattributed_touch_is_not_placed(self, tmp_path):
        rows = bout(n=100)
        out = fp.piste_zones(rows, [{"time_s": 1.0, "scorer": "unknown"}])
        assert sum(out["fencer_1"]) + sum(out["fencer_2"]) == 0

    def test_says_so_when_there_is_nothing_to_place(self):
        assert fp.piste_zones([], [])["available"] is False


class TestScoreSurvivesTheRefusal:
    def test_a_swapped_bout_still_reports_its_scoreline(self, tmp_path):
        """
        The swap check withholds the per-fencer axes because they describe the
        tracker. The score does not come from tracking at all, and withholding it
        would mean suppressing something the system got right.
        """
        rows = bout(f1=lambda i: 2.0 if i < 200 else 6.0,
                    f2=lambda i: 6.0 if i < 200 else 2.0)
        out = fp.build(write_csv(tmp_path, rows),
                       [{"time_s": 10.0, "scorer": "left"}])
        assert out["available"] is False
        assert out["score"]["available"] is True
        assert out["score"]["final"] == {"fencer_1": 1, "fencer_2": 0}


class TestFootworkRefusal:
    """
    The check that stops the system answering for a sport it cannot measure.

    Wheelchair fencing is fenced from frames bolted to the floor. Every axis
    here, and the touch and lunge detectors upstream, assume distance opens and
    closes because the fencers move their feet. Pointed at such a bout the system
    does not fail visibly, which is worse: it returns a flat distance series, no
    proposed touches, and a profile at parity, all of which read as a cagey bout.
    """

    def test_refuses_when_neither_fencer_moves(self, tmp_path):
        rows = bout(f1=lambda i: 2.0, f2=lambda i: 6.0)
        out = fp.build(write_csv(tmp_path, rows), [])
        assert out["available"] is False
        assert "wheelchair" in out["reason"]
        assert out["footwork_iqr_m"] < fp.MIN_FOOTWORK_IQR_M

    def test_one_fencer_holding_still_is_ordinary_fencing(self, tmp_path):
        # A fencer who holds the line while the other works is normal. Only
        # NEITHER moving is the case this exists to catch, which is why the
        # larger of the two ranges is tested rather than the mean.
        rows = bout(f1=lambda i: 2.0, f2=_walk(6.0))
        out = fp.build(write_csv(tmp_path, rows), [])
        assert out["available"] is True

    def test_the_scoreline_survives_the_refusal(self, tmp_path):
        # The touches came from the user, not from footwork. Withholding them
        # would suppress something the system did not get wrong.
        rows = bout(f1=lambda i: 2.0, f2=lambda i: 6.0)
        out = fp.build(write_csv(tmp_path, rows),
                       [{"time_s": 10.0, "scorer": "left"}])
        assert out["available"] is False
        assert out["score"]["available"] is True
        assert out["score"]["final"] == {"fencer_1": 1, "fencer_2": 0}

    def test_every_evaluation_clip_passes_the_check(self):
        """
        The threshold must not reject real foot fencing. The smallest own-position
        spread measured across the four clips is 0.66 m, against a threshold of
        0.25, so there is a wide margin. This test fails if a future change to
        the threshold or to the position pipeline closes it.
        """
        import glob
        import os
        seen = 0
        for path in sorted(glob.glob(os.path.join(
                os.path.dirname(__file__), "results_current",
                "fencing_clip*_distance.csv"))):
            rows = fp.load_rows(path)
            _, p1, p2 = fp._positions(rows)
            if len(p1) < 10:
                continue
            seen += 1
            assert fp.footwork_range(p1, p2) > fp.MIN_FOOTWORK_IQR_M * 2, path
        if seen == 0:
            pytest.skip("evaluation footage not present in this checkout")


# --- pace ---------------------------------------------------------------

def test_pace_counts_doubles_half_to_each():
    t = [{"time_s": 10, "scorer": "left"},
         {"time_s": 20, "scorer": "right"},
         {"time_s": 30, "scorer": "double"}]
    p = fp.pace(t, 60.0)
    assert p["per_min"] == 3.0
    assert p["fencer_1_per_min"] == 1.5
    assert p["fencer_2_per_min"] == 1.5


def test_pace_does_not_attribute_a_touch_with_no_scorer():
    """The per-fencer rates need not sum to the bout rate, and must not be made
    to: splitting an unconfirmed touch would invent an attribution."""
    t = [{"time_s": 10, "scorer": "left"}, {"time_s": 20}]
    p = fp.pace(t, 60.0)
    assert p["per_min"] == 2.0
    assert p["fencer_1_per_min"] + p["fencer_2_per_min"] == 1.0
    assert p["unattributed"] == 1


def test_pace_is_absent_rather_than_zero_without_touches():
    assert fp.pace([], 120.0)["available"] is False
    assert fp.pace([{"time_s": 1, "scorer": "left"}], 0)["available"] is False


def test_pace_survives_a_tracking_refusal(tmp_path):
    """Pace comes from confirmed touches and the clock, not from tracking, so a
    bout whose per-fencer axes are withheld still has a pace."""
    rows = bout(n=400, f1=lambda i: 2.0 + 4.0 * (i > 200),
                f2=lambda i: 6.0 - 4.0 * (i > 200))
    csv_path = write_csv(tmp_path, rows)
    touches = [{"time_s": 10, "scorer": "left"}, {"time_s": 12, "scorer": "right"}]
    res = fp.build(csv_path, touches)
    assert res["available"] is False
    assert res["pace"]["available"] is True
    assert res["pace"]["touches"] == 2


def test_pace_would_duplicate_the_scoring_share_axis():
    """The reason pace is not a seventh axis, asserted rather than left in a
    comment: within one bout both fencers are divided by the same duration, so
    scoring a rate as a share of the pair reproduces the existing axis exactly."""
    t = [{"time_s": 5, "scorer": "left"}, {"time_s": 9, "scorer": "left"},
         {"time_s": 14, "scorer": "right"}, {"time_s": 20, "scorer": "double"}]
    p = fp.pace(t, 120.0)
    as_axis = fp._share(p["fencer_1_per_min"], p["fencer_2_per_min"])
    rows = [{"time_s": str(t_), "distance_smooth_m": "2.0"} for t_ in range(30)]
    share = fp.touch_axes(rows, t)[1]["scoring_share_pct"]
    assert as_axis == share
