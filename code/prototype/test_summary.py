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
        # distances: 2.0 (far), 0.8 (close), 1.5 (engagement)
        stats = compute_stats(make_rows())
        zones = stats["time_in_zone_pct"]
        assert math.isclose(zones["close_under_1m"], 33.3, abs_tol=0.1)
        assert math.isclose(zones["engagement_1_to_1.8m"], 33.3, abs_tol=0.1)
        assert math.isclose(zones["far_over_1.8m"], 33.3, abs_tol=0.1)

    def test_push_pull_from_last_row(self):
        stats = compute_stats(make_rows())
        assert stats["fencer_1"]["push_m"] == 1.5
        assert stats["fencer_1"]["pull_m"] == 0.5
        assert stats["fencer_1"]["net_forward_m"] == 1.0
        assert stats["fencer_1"]["push_share_pct"] == 75.0
        assert stats["fencer_2"]["net_forward_m"] == 0.0

    def test_zero_movement_has_no_push_share(self):
        rows = make_rows()[:1]  # single frame, all zeros
        stats = compute_stats(rows)
        assert stats["fencer_1"]["push_share_pct"] is None

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
