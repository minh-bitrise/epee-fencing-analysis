"""
Tests for the language model faithfulness audit.

The audit's own failure mode is being reassuring. A checker that returns "no
unsupported figures" because it cannot detect an unsupported figure is worse
than no checker, because the clean result gets quoted. Most of these tests
therefore attack the audit rather than exercise it.
"""
import pytest

import audit_summary as a

PAYLOAD = {
    "duration_s": 180.0,
    "coverage_pct": 97.4,
    "distance_m": {"mean": 2.53, "max": 5.02},
    "time_in_zone_pct": {"lunge_distance_1.5_to_2.6m": 45.2},
    "touches": {"count": 14, "times_s": [19.0, 35.0]},
    "fencer_1": {"closing_share_pct": 52.4},
    "movement_basis": {"source": "raw per-frame positions"},
}


class TestCatchingFabrication:
    def test_catches_a_number_that_is_not_in_the_payload(self):
        r = a.audit("Reaction time averaged 0.31 seconds.", PAYLOAD)
        assert [t for t, _ in r["unsupported"]] == ["0.31"]

    def test_accepts_a_number_that_is(self):
        r = a.audit("Coverage was 97.4 per cent.", PAYLOAD)
        assert r["unsupported"] == []

    def test_catches_a_plausible_near_miss(self):
        # The dangerous case: a figure close enough to a real one to read as a
        # rounding, far enough to be a different claim.
        r = a.audit("Coverage was 87.4 per cent.", PAYLOAD)
        assert [t for t, _ in r["unsupported"]] == ["87.4"]

    def test_reads_numbers_written_as_words(self):
        # A model writes "fourteen touches" as readily as "14", and the written
        # form carries the most quotable claims.
        assert a.audit("There were fourteen touches.", PAYLOAD)["unsupported"] == []
        assert a.audit("There were thirty touches.", PAYLOAD)["unsupported"]


class TestTolerance:
    def test_precision_written_sets_the_tolerance(self):
        # "52" claims no decimals and should accept 52.4 as a rounding; "52.0"
        # claims one and should not.
        assert a.audit("about 52 per cent", PAYLOAD)["unsupported"] == []
        assert a.audit("exactly 52.0 per cent", PAYLOAD)["unsupported"]

    def test_tolerance_follows_the_decimals(self):
        assert a.tolerance_for("45") == 0.5
        assert a.tolerance_for("45.2") == 0.05
        assert a.tolerance_for("45.25") == 0.005


class TestPayloadCoverage:
    def test_numbers_in_key_names_count_as_grounded(self):
        # Zone boundaries are supplied by the payload as part of the key
        # `lunge_distance_1.5_to_2.6m`, not as values.
        r = a.audit("Between 1.5 and 2.6 metres.", PAYLOAD)
        assert r["unsupported"] == []

    def test_values_inside_lists_are_found(self):
        assert a.audit("A touch at 19.0 seconds.", PAYLOAD)["unsupported"] == []

    def test_strings_in_the_payload_are_not_treated_as_numbers(self):
        assert "movement_basis.source" not in a.flatten(PAYLOAD)


class TestSensitivityIsMeasured:
    def test_the_audit_reports_what_it_can_detect(self):
        """
        A clean audit is only readable alongside what it could have caught. This
        is the same reasoning the project applies to its null movement results.
        """
        s = a.sensitivity(PAYLOAD, trials=100)
        assert set(s) == {"small integers (0-20)", "larger integers (21-500)",
                          "one decimal", "two decimals"}
        assert all(0 <= v <= 100 for v in s.values())

    def test_decimals_are_detected_far_better_than_small_integers(self):
        # The structural weakness, asserted so it cannot be quietly lost: a
        # payload of this size covers the small integers densely.
        s = a.sensitivity(PAYLOAD, trials=200)
        assert s["two decimals"] > s["small integers (0-20)"]

    def test_pair_derivation_destroys_sensitivity(self):
        """
        The measurement that changed the design. Searching differences and sums
        over every pair looks conservative and makes the audit unable to detect
        anything, because the combinations cover the number line densely.
        """
        strict = a.sensitivity(PAYLOAD, trials=200, pairs=False)
        lenient = a.sensitivity(PAYLOAD, trials=200, pairs=True)
        assert strict["one decimal"] > lenient["one decimal"]


class TestReporting:
    def test_a_summary_with_no_numbers_does_not_divide_by_zero(self):
        r = a.audit("The bout was closely fought throughout.", PAYLOAD)
        assert r["figures"] == 0
        assert r["grounded_pct"] is None

    def test_the_rate_counts_every_figure_found(self):
        r = a.audit("Coverage 97.4 per cent and reaction 0.31 seconds.", PAYLOAD)
        assert r["figures"] == len(r["grounded"]) + len(r["derived"]) \
            + len(r["unsupported"])
