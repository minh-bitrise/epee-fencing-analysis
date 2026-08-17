"""
Tests for the annotation store and API.

These exercise the four annotation actions the design specifies, against a
synthetic bout written to a temporary directory, so no real footage or pipeline
run is required.

Run with:
    cd code/backend && python3 -m pytest test_api.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prototype")))

from store import (  # noqa: E402
    AnnotationStore, CONFIRMED, PENDING, REJECTED,
    discover_bouts, load_proposed_touches,
)


# -------------------- fixtures --------------------

@pytest.fixture
def results_dir(tmp_path):
    """A directory shaped like the pipeline's output."""
    d = tmp_path / "results_test"
    d.mkdir()
    (d / "mybout_distance.csv").write_text(
        "frame,time_s,distance_raw_m,distance_smooth_m,method,"
        "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m\n"
        + "".join(
            f"{i},{i*0.5},2.0,2.0,pose,{i*0.1:.2f},0.0,{i*0.2:.2f},0.0\n"
            for i in range(60))
    )
    (d / "mybout_distance_touches.csv").write_text(
        "time_s,confidence,min_distance_m,separation_m,audio_support,signals\n"
        "10.0,0.90,1.20,2.10,0,approach+separated\n"
        "20.0,0.60,1.80,1.10,0,approach+separated\n"
    )
    return d


@pytest.fixture
def store(tmp_path):
    return AnnotationStore(str(tmp_path / "annotations"))


# -------------------- discovery --------------------

class TestDiscoverBouts:
    def test_finds_a_processed_bout(self, results_dir):
        bouts = discover_bouts([str(results_dir)])
        assert len(bouts) == 1
        b = list(bouts.values())[0]
        assert b.exists() and b.touches_csv

    def test_bout_id_includes_the_results_directory(self, results_dir):
        """
        The same clip processed into different output directories must stay
        distinguishable, because the evaluation depends on before/after
        comparisons living side by side.
        """
        bouts = discover_bouts([str(results_dir)])
        assert list(bouts)[0].startswith("results_test:")

    def test_missing_directory_is_ignored(self):
        assert discover_bouts(["/no/such/place"]) == {}

    def test_absent_touches_file_is_tolerated(self, results_dir):
        os.unlink(results_dir / "mybout_distance_touches.csv")
        b = list(discover_bouts([str(results_dir)]).values())[0]
        assert b.touches_csv == ""


class TestLoadProposedTouches:
    def test_reads_detector_output(self, results_dir):
        t = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        assert [x["time_s"] for x in t] == [10.0, 20.0]
        assert t[0]["confidence"] == 0.90
        assert all(x["origin"] == "detector" for x in t)

    def test_ids_are_stable_and_distinct(self, results_dir):
        t = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        assert [x["id"] for x in t] == ["p0", "p1"]

    def test_missing_file_gives_empty_list(self):
        assert load_proposed_touches("/no/such/file.csv") == []
        assert load_proposed_touches("") == []


# -------------------- action 1: confirm / correct / reject --------------------

class TestConfirmCorrectReject:
    def test_unreviewed_touches_are_pending(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        p = store.review_progress("b", proposed)
        assert p["reviewed"] == 0 and not p["complete"]

    def test_confirming_records_the_state(self, store):
        store.set_touch_state("b", "p0", CONFIRMED)
        assert store.load("b")["touch_states"]["p0"]["state"] == CONFIRMED

    def test_rejecting_records_the_state(self, store):
        store.set_touch_state("b", "p0", REJECTED)
        assert store.load("b")["touch_states"]["p0"]["state"] == REJECTED

    def test_correction_and_confirmation_are_one_action(self, store):
        """
        Adjusting the timestamp and assigning a scorer happen in the same call as
        confirming, because requiring two interactions for one decision would work
        against the design's aim of resolving each error in a single step.
        """
        store.set_touch_state("b", "p0", CONFIRMED, scorer="left", time_s=10.4)
        e = store.load("b")["touch_states"]["p0"]
        assert e["state"] == CONFIRMED and e["scorer"] == "left" and e["time_s"] == 10.4

    def test_invalid_state_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.set_touch_state("b", "p0", "maybe")

    def test_invalid_scorer_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.set_touch_state("b", "p0", CONFIRMED, scorer="nobody")

    def test_progress_counts_reviewed_touches(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        store.set_touch_state("b", "p0", CONFIRMED)
        p = store.review_progress("b", proposed)
        assert p["reviewed"] == 1 and p["confirmed"] == 1 and not p["complete"]
        store.set_touch_state("b", "p1", REJECTED)
        p = store.review_progress("b", proposed)
        assert p["complete"] and p["rejected"] == 1


# -------------------- action 2: add a missed touch --------------------

class TestAddTouch:
    def test_adds_and_returns_an_id(self, store):
        tid = store.add_touch("b", 42.0, "right")
        added = store.load("b")["added_touches"]
        assert len(added) == 1 and added[0]["id"] == tid
        assert added[0]["origin"] == "user"

    def test_rejects_an_unknown_scorer(self, store):
        with pytest.raises(ValueError):
            store.add_touch("b", 42.0, "sideways")

    def test_can_be_removed(self, store):
        tid = store.add_touch("b", 42.0)
        assert store.remove_added_touch("b", tid)
        assert store.load("b")["added_touches"] == []

    def test_removing_an_unknown_id_reports_failure(self, store):
        assert not store.remove_added_touch("b", "nope")


# -------------------- action 3: mark a segment unreliable --------------------

class TestUnreliableSegments:
    def test_records_a_segment(self, store):
        store.add_unreliable_segment("b", 30.0, 35.0, "clinch")
        segs = store.load("b")["unreliable_segments"]
        assert len(segs) == 1 and segs[0]["start_s"] == 30.0

    def test_rejects_a_reversed_range(self, store):
        with pytest.raises(ValueError):
            store.add_unreliable_segment("b", 35.0, 30.0)

    def test_rejects_an_empty_range(self, store):
        with pytest.raises(ValueError):
            store.add_unreliable_segment("b", 30.0, 30.0)

    def test_can_be_removed(self, store):
        store.add_unreliable_segment("b", 30.0, 35.0)
        sid = store.load("b")["unreliable_segments"][0]["id"]
        assert store.remove_unreliable_segment("b", sid)
        assert store.load("b")["unreliable_segments"] == []


# -------------------- action 4: re-anchor a fencer --------------------

class TestReanchor:
    def test_records_as_pending_not_applied(self, store):
        """
        Re-anchoring changes tracking rather than interpretation, so it cannot
        take effect until the pipeline is rerun. Storing it as pending keeps the
        interface honest about that instead of implying the fix is already live.
        """
        store.add_reanchor("b", 12.0, 1, 640.0, 400.0)
        a = store.load("b")["reanchors"][0]
        assert a["applied"] is False and a["slot"] == 1

    def test_rejects_an_invalid_slot(self, store):
        with pytest.raises(ValueError):
            store.add_reanchor("b", 12.0, 5, 100.0, 100.0)

    def test_pending_count_surfaces_in_progress(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        store.add_reanchor("b", 12.0, 0, 100.0, 100.0)
        assert store.review_progress("b", proposed)["pending_reanchors"] == 1


# -------------------- derived confirmed-touch list --------------------

class TestConfirmedTouchTimes:
    def test_excludes_pending_and_rejected(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        store.set_touch_state("b", "p1", REJECTED)
        # p0 left pending on purpose
        assert store.confirmed_touch_times("b", proposed) == []

    def test_includes_confirmed_at_the_corrected_time(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        store.set_touch_state("b", "p0", CONFIRMED, scorer="left", time_s=10.6)
        got = store.confirmed_touch_times("b", proposed)
        assert len(got) == 1 and got[0]["time_s"] == 10.6 and got[0]["scorer"] == "left"

    def test_includes_user_added_and_sorts(self, store, results_dir):
        proposed = load_proposed_touches(str(results_dir / "mybout_distance_touches.csv"))
        store.set_touch_state("b", "p1", CONFIRMED)
        store.add_touch("b", 5.0, "right")
        got = store.confirmed_touch_times("b", proposed)
        assert [g["time_s"] for g in got] == [5.0, 20.0]
        assert got[0]["origin"] == "user-added"
        assert got[1]["origin"] == "detector-confirmed"


# -------------------- persistence --------------------

class TestPersistence:
    def test_state_survives_a_new_store_instance(self, tmp_path):
        root = str(tmp_path / "ann")
        AnnotationStore(root).set_touch_state("b", "p0", CONFIRMED, scorer="left")
        again = AnnotationStore(root).load("b")
        assert again["touch_states"]["p0"]["scorer"] == "left"

    def test_pipeline_artefacts_are_never_modified(self, store, results_dir):
        """
        The design requires pipeline output to stay editable-but-intact, so
        reprocessing never destroys user work and user work never has to be
        recovered from a mutated artefact.
        """
        csv_path = results_dir / "mybout_distance_touches.csv"
        before = csv_path.read_text()
        store.set_touch_state("b", "p0", CONFIRMED)
        store.add_touch("b", 99.0)
        store.add_unreliable_segment("b", 1.0, 2.0)
        assert csv_path.read_text() == before

    def test_a_missing_file_yields_empty_state(self, store):
        d = store.load("never-seen")
        assert d["touch_states"] == {} and d["added_touches"] == []

    def test_tolerates_a_file_from_an_earlier_version(self, tmp_path):
        root = tmp_path / "ann"
        root.mkdir()
        (root / "b.json").write_text(json.dumps({"bout_id": "b"}))
        d = AnnotationStore(str(root)).load("b")
        assert d["touch_states"] == {} and d["reanchors"] == []
