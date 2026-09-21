"""
Unit tests for the LLM tactical-summary generator.
Tests the pure data path (stats, prompt, cache) and the caching
behaviour of generate() with the API call mocked out - no network
and no API key needed.

Run with:
    python3 -m pytest test_summary.py -v
"""

import json
import math

import pytest

import generate_summary
from generate_summary import (
    SYSTEM_PROMPT,
    build_prompt,
    cache_key,
    compute_stats,
    generate,
)


def make_rows():
    """A tiny synthetic bout: 4 frames, one frame without a distance."""
    return [
        {"frame": "0", "time_s": "0.0", "distance_raw_m": "2.0",
         "distance_smooth_m": "2.0", "method": "pose",
         "f1_advance_m": "0.0", "f1_retreat_m": "0.0",
         "f2_advance_m": "0.0", "f2_retreat_m": "0.0"},
        {"frame": "1", "time_s": "0.5", "distance_raw_m": "0.8",
         "distance_smooth_m": "1.0", "method": "bbox",
         "f1_advance_m": "0.5", "f1_retreat_m": "0.0",
         "f2_advance_m": "0.0", "f2_retreat_m": "0.4"},
        {"frame": "2", "time_s": "1.0", "distance_raw_m": "",
         "distance_smooth_m": "", "method": "",
         "f1_advance_m": "0.5", "f1_retreat_m": "0.0",
         "f2_advance_m": "0.0", "f2_retreat_m": "0.4"},
        {"frame": "3", "time_s": "1.5", "distance_raw_m": "1.5",
         "distance_smooth_m": "1.4", "method": "pose",
         "f1_advance_m": "1.5", "f1_retreat_m": "0.5",
         "f2_advance_m": "1.0", "f2_retreat_m": "1.0"},
    ]


class TestComputeStats:
    def test_basic_shape(self):
        stats = compute_stats(make_rows())
        assert stats["frames_total"] == 4
        assert stats["duration_s"] == 1.5

    def test_coverage_excludes_missing_distance(self):
        # 3 of 4 frames have a distance value
        stats = compute_stats(make_rows())
        assert stats["coverage_pct"] == 75.0

    def test_pose_share_is_relative_to_covered_frames(self):
        # 2 pose frames out of 3 frames with distance
        stats = compute_stats(make_rows())
        assert math.isclose(stats["pose_method_pct"], 66.7, abs_tol=0.1)

    def test_distance_aggregates(self):
        stats = compute_stats(make_rows())
        assert stats["distance_m"]["min"] == 0.8
        assert stats["distance_m"]["max"] == 2.0
        assert math.isclose(stats["distance_m"]["mean"], 1.43, abs_tol=0.01)

    def test_zone_split(self):
        # distances present are 0.8 and 1.5 (both close/infighting, since the
        # close band runs to 1.5 m inclusive) and 2.0 (lunge distance)
        stats = compute_stats(make_rows())
        zones = stats["time_in_zone_pct"]
        assert math.isclose(zones["close_infighting_under_1.5m"], 66.7, abs_tol=0.1)
        assert math.isclose(zones["lunge_distance_1.5_to_2.6m"], 33.3, abs_tol=0.1)
        assert zones["advance_lunge_2.6_to_3.5m"] == 0.0
        assert zones["out_of_distance_over_3.5m"] == 0.0

    def test_zone_split_spans_all_four_bands(self):
        """One sample per band, so each band is exercised and sums to 100."""
        def row(i, d):
            return {"frame": str(i), "time_s": f"{i*0.5}", "distance_raw_m": str(d),
                    "distance_smooth_m": str(d), "method": "pose",
                    "f1_advance_m": "0.0", "f1_retreat_m": "0.0",
                    "f2_advance_m": "0.0", "f2_retreat_m": "0.0"}
        rows = [row(0, 1.0), row(1, 2.2), row(2, 3.0), row(3, 6.0)]
        zones = compute_stats(rows)["time_in_zone_pct"]
        assert math.isclose(zones["close_infighting_under_1.5m"], 25.0, abs_tol=0.1)
        assert math.isclose(zones["lunge_distance_1.5_to_2.6m"], 25.0, abs_tol=0.1)
        assert math.isclose(zones["advance_lunge_2.6_to_3.5m"], 25.0, abs_tol=0.1)
        assert math.isclose(zones["out_of_distance_over_3.5m"], 25.0, abs_tol=0.1)
        assert math.isclose(sum(zones.values()), 100.0, abs_tol=0.1)

    def test_net_displacement_is_the_headline_movement_figure(self):
        """
        Fencer 1 advances 1.5 m and retreats 0.5 m, so finishes 1.0 m forward.
        Fencer 2 advances and retreats 1.0 m each, so finishes where they began.
        """
        stats = compute_stats(make_rows())
        assert stats["fencer_1"]["net_displacement_m"] == 1.0
        assert stats["fencer_2"]["net_displacement_m"] == 0.0

    def test_closing_share_counts_directions_not_distances(self):
        """
        Both of Fencer 1's moving frames go forward, so 100 per cent. Fencer 2
        has one frame each way, so 50 per cent, even though the two movements
        differ in size. That magnitude-independence is the whole point of the
        metric: it is why noise contributes symmetrically instead of accumulating.
        """
        stats = compute_stats(make_rows())
        assert stats["fencer_1"]["closing_share_pct"] == 100.0
        assert stats["fencer_2"]["closing_share_pct"] == 50.0

    def test_cumulative_totals_reach_the_payload_in_no_form(self):
        """
        The totals are wrong rather than approximate. B1g traced their error to the
        per-frame movement cap and measured it at 24 m on a 14 m piste, so no
        caveat makes them usable, and every key ending "_indicative" has gone with
        them. Including them cost three sentences of prompt spent talking the model
        out of a number worth nothing. They stay in the CSV, which is evidence.
        """
        f1 = compute_stats(make_rows())["fencer_1"]
        assert set(f1) == {"net_displacement_m", "closing_share_pct"}
        assert not any("indicative" in k for k in f1)

    def test_zero_movement_has_no_closing_share(self):
        rows = make_rows()[:1]  # single frame, all zeros
        stats = compute_stats(rows)
        assert stats["fencer_1"]["closing_share_pct"] is None
        assert stats["fencer_1"]["net_displacement_m"] == 0.0

    def test_movement_basis_records_the_derivation_route(self):
        """
        These rows carry no position columns, so the figures come from the
        cumulative fallback and the payload must say so. The two routes disagreed
        by 3.5 m on clip 3, so which one was used is not a detail.
        """
        basis = compute_stats(make_rows())["movement_basis"]
        assert "cumulative" in basis["source"]
        assert basis["closing_share_is_window_dependent"] is True

    def test_movement_basis_reports_positions_when_available(self):
        rows = [dict(r, f1_pos_m=str(i * 0.5), f2_pos_m=str(4.0 - i * 0.5))
                for i, r in enumerate(make_rows())]
        basis = compute_stats(rows)["movement_basis"]
        assert basis["source"] == "raw per-frame positions"

    def test_implausible_net_displacement_is_flagged(self):
        """
        Play resets to the guard lines after every touch, so a fencer finishing
        far up the piste means the position measurement drifted. That physical
        constraint is a correctness check needing no ground truth.
        """
        rows = make_rows()
        rows[-1]["f1_advance_m"] = "30.0"
        stats = compute_stats(rows)
        assert "data_quality_warnings" in stats
        warning = stats["data_quality_warnings"][0]
        assert "net_displacement_m" in warning
        # panning was tested and ruled out, so the warning must not blame it
        assert "closing_share_pct" in warning

    def test_plausible_net_displacement_is_not_flagged(self):
        assert "data_quality_warnings" not in compute_stats(make_rows())

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError):
            compute_stats([])

    def test_no_distances_raise(self):
        rows = [dict(make_rows()[2])]  # only the empty-distance frame
        with pytest.raises(ValueError):
            compute_stats(rows)


class TestPrompt:
    def test_prompt_contains_stats_json(self):
        stats = compute_stats(make_rows())
        prompt = build_prompt(stats)
        # the stats payload must be embedded verbatim as JSON
        assert json.dumps(stats, indent=2) in prompt

    def test_prompt_requests_required_sections(self):
        prompt = build_prompt(compute_stats(make_rows()))
        for heading in ("## Bout summary", "## Observed tendencies",
                        "## Suggestions to explore", "## Data caveats"):
            assert heading in prompt

    def test_system_prompt_has_honesty_constraints(self):
        assert "Never invent touches" in SYSTEM_PROMPT
        assert "estimates" in SYSTEM_PROMPT

    def test_prompt_forbids_claims_from_indicative_figures(self):
        """
        Both prompt variants must carry the movement-reading rules. The payload
        cannot convey them: every number in it looks equally authoritative, and
        this pipeline has already produced a summary that faithfully reported a
        mis-specified input as fact.
        """
        for has_touches in (False, True):
            prompt = generate_summary.build_system_prompt(has_touches=has_touches)
            assert "net_displacement_m" in prompt
            assert "closing_share_pct" in prompt
            # the totals are absent from the payload, so the prompt must forbid
            # reconstructing them rather than caveat a figure that is not there
            assert "total distance covered" in prompt
            assert "do not describe a fencer as having covered any distance" in prompt

    def test_prompt_separates_displacement_from_scoped_movement(self):
        """
        The scoped figure can exceed the whole-recording displacement, so the
        prompt has to say it is not an endpoint measurement or the model will
        report a fencer as finishing 7.6 m up a 14 m piste.
        """
        prompt = generate_summary.build_system_prompt(has_touches=True)
        assert "net_forward_movement_m" in prompt
        assert "NOT a displacement" in prompt


class TestCacheKey:
    def test_stable_for_same_inputs(self):
        assert cache_key("m", "s", "u") == cache_key("m", "s", "u")

    def test_changes_with_any_input(self):
        base = cache_key("m", "s", "u")
        assert cache_key("m2", "s", "u") != base
        assert cache_key("m", "s2", "u") != base
        assert cache_key("m", "s", "u2") != base


class TestGenerateCaching:
    def _write_csv(self, tmp_path):
        csv_path = tmp_path / "bout_distance.csv"
        rows = make_rows()
        header = ",".join(rows[0].keys())
        lines = [header] + [",".join(r.values()) for r in rows]
        csv_path.write_text("\n".join(lines) + "\n")
        return str(csv_path)

    def test_generates_and_then_hits_cache(self, tmp_path, monkeypatch):
        csv_path = self._write_csv(tmp_path)
        calls = []

        def fake_llm(model, system_prompt, user_prompt):
            calls.append(model)
            return "## Bout summary\nfake"

        monkeypatch.setattr(generate_summary, "call_llm", fake_llm)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        out1 = generate(csv_path)          # first run calls the (fake) API
        out2 = generate(csv_path)          # second run must hit the cache
        assert out1 == out2
        assert len(calls) == 1

        with open(out1) as f:
            assert "fake" in f.read()

    def test_force_regenerates(self, tmp_path, monkeypatch):
        csv_path = self._write_csv(tmp_path)
        calls = []
        monkeypatch.setattr(
            generate_summary, "call_llm",
            lambda m, s, u: calls.append(m) or "summary")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        generate(csv_path)
        generate(csv_path, force=True)
        assert len(calls) == 2

    def test_missing_api_key_exits_with_help(self, tmp_path, monkeypatch):
        csv_path = self._write_csv(tmp_path)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            generate(csv_path)


class TestSideSwapDetection:
    """Whether slot identity held, which is the check this project was missing.

    Clip 4 reported +7.08 m of net displacement for one fencer and the report
    recorded the cause as not yet identified.
    """

    def rows(self, pairs):
        return [{"time_s": str(i * 0.1), "f1_pos_m": str(a), "f2_pos_m": str(b)}
                for i, (a, b) in enumerate(pairs)]

    def test_a_stable_pair_records_no_swaps(self):
        from generate_summary import count_side_swaps
        # one fencer stays left, the other right, which is what fencing is
        swaps, left, _ = count_side_swaps(
            self.rows([(2.0, 6.0), (2.5, 5.5), (3.0, 5.0), (2.2, 6.1)]))
        assert swaps == 0 and left == 1.0

    def test_a_crossing_pair_records_a_swap(self):
        from generate_summary import count_side_swaps
        swaps, left, _ = count_side_swaps(
            self.rows([(2.0, 6.0), (3.0, 5.0), (6.0, 2.0), (6.5, 1.5)]))
        assert swaps == 1
        assert left == 0.5

    def test_repeated_swapping_is_counted_each_time(self):
        from generate_summary import count_side_swaps
        swaps, _, _ = count_side_swaps(
            self.rows([(2, 6), (6, 2), (2, 6), (6, 2), (2, 6)]))
        assert swaps == 4

    def test_reports_how_close_they_came(self):
        # The separation at which the matcher cannot tell them apart is the
        # mechanism, so it is reported rather than merely the count.
        from generate_summary import count_side_swaps
        _, _, min_sep = count_side_swaps(
            self.rows([(2.0, 6.0), (3.9, 4.0), (2.0, 6.0)]))
        assert min_sep == pytest.approx(0.1, abs=0.01)

    def test_too_few_frames_is_not_a_swap(self):
        from generate_summary import count_side_swaps
        assert count_side_swaps([]) == (0, None, None)

    def test_a_swap_warns_that_per_slot_figures_are_unattributable(self):
        """
        The warning has to say what is unusable, not merely that something is
        wrong. A per-slot figure on a bout with swaps cannot be attributed to a
        fencer at all, which is a stronger statement than "this looks large".
        """
        from generate_summary import movement_quality_warnings
        stats = {"fencer_1": {"net_displacement_m": 0.05},
                 "fencer_2": {"net_displacement_m": 7.08}}
        rows = self.rows([(2, 6), (6, 2), (2, 6)])
        warnings = movement_quality_warnings(stats, rows)
        assert any("NOT attributable" in w.lower() or
                   "not attributable" in w.lower() for w in warnings)

    def test_a_clean_bout_with_a_small_net_warns_about_nothing(self):
        from generate_summary import movement_quality_warnings
        stats = {"fencer_1": {"net_displacement_m": 0.3},
                 "fencer_2": {"net_displacement_m": -0.2}}
        rows = self.rows([(2.0, 6.0), (2.5, 5.5), (2.1, 6.0)])
        assert movement_quality_warnings(stats, rows) == []
