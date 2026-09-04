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


def bout(n=400, f1=lambda i: 2.0, f2=lambda i: 6.0, dist=lambda i: 4.0):
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
        rows = bout(f1=lambda i: 3.0 + (0.0 if i else -3.0),
                    f2=lambda i: 7.0 + (0.0 if i else 1.0))
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
