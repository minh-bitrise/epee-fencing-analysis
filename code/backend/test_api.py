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


# -------------------- metrics endpoint --------------------

@pytest.fixture
def metrics_bout(results_dir, tmp_path, monkeypatch):
    """
    Point the app at the synthetic results directory and return its bout id.

    The endpoint is exercised by calling the handler directly rather than over
    HTTP, because what is under test is the metric selection, not FastAPI.
    """
    import app as app_module
    monkeypatch.setattr(app_module, "RESULTS_DIRS", [str(results_dir)])
    monkeypatch.setattr(app_module, "store",
                        AnnotationStore(str(tmp_path / "ann_metrics")))
    return app_module, list(app_module._bouts())[0]


class TestMetricsEndpoint:
    """
    The interface must present the movement figures that survived measurement,
    not the cumulative push and pull totals. Path length changes fivefold with
    the smoothing window while net displacement does not move at all, so a total
    displayed next to a real measurement misrepresents itself as one.
    """

    def test_reliable_movement_figures_are_returned(self, metrics_bout):
        app_module, bout_id = metrics_bout
        m = app_module.get_metrics(bout_id)
        for fencer in ("fencer_1", "fencer_2"):
            block = m["in_play"][fencer]
            assert "net_forward_movement_m" in block
            assert "closing_share_pct" in block

    def test_cumulative_totals_are_not_returned_at_all(self, metrics_bout):
        """
        The totals are wrong rather than approximate: B1g measured the error at 24 m
        on a 14 m piste and traced it to the movement cap. An API that returns them,
        under any name, invites a client to display them next to a real
        measurement, where they read as one. They stay in the CSV instead.
        """
        app_module, bout_id = metrics_bout
        block = app_module.get_metrics(bout_id)["in_play"]["fencer_1"]
        for k in ("push_m", "pull_m", "push_m_indicative", "pull_m_indicative"):
            assert k not in block

    def test_whole_recording_net_displacement_is_available(self, metrics_bout):
        """
        The interface shows both, and only the whole-recording figure is a true
        displacement, so the endpoint has to carry it.
        """
        app_module, bout_id = metrics_bout
        w = app_module.get_metrics(bout_id)["whole_recording"]
        assert "net_displacement_m" in w["fencer_1"]

    def test_synthetic_bout_advances_both_fencers(self, metrics_bout):
        """
        The fixture's CSV has both fencers advancing monotonically and never
        retreating, so both net figures must be positive. This checks the sign
        convention survives the endpoint, which is where an earlier defect lived.
        """
        app_module, bout_id = metrics_bout
        m = app_module.get_metrics(bout_id)
        assert m["whole_recording"]["fencer_1"]["net_displacement_m"] > 0
        assert m["in_play"]["fencer_1"]["net_forward_movement_m"] > 0

    def test_derivation_route_is_reported(self, metrics_bout):
        """
        The fixture CSV predates the position columns, so the response must say
        the fallback route was used. Reporting it is what stops a silent
        degradation: the fallback disagreed with the position route by 3.5 m of
        net displacement on real footage.
        """
        app_module, bout_id = metrics_bout
        basis = app_module.get_metrics(bout_id)["movement_basis"]
        assert "cumulative" in basis["source"]

    def test_the_sets_with_position_columns_are_discoverable(self):
        """
        Outputs carrying the raw position columns must be reachable, or every bout
        the interface can open silently uses the cumulative fallback. results_current
        is the reference set and must be first, since a selection that silently falls
        through to another directory is how a 180p ablation clip got measured in place
        of the clip under test.
        """
        import app as app_module
        names = [os.path.basename(d) for d in app_module.RESULTS_DIRS]
        assert names[0] == "results_current"
        for wanted in ("results_current", "results_fixed", "results_pose"):
            assert wanted in names

    def test_the_endpoint_keeps_that_order_rather_than_sorting(self, tmp_path, monkeypatch):
        """
        The order above is only worth having if it survives to the client. It did
        not: /api/bouts sorted alphabetically, so the interface opened on
        `results_ablation:ablation_180p`, the deliberately degraded clip kept so
        the resolution ablation stays openable. Asserting the constant was first
        passed while the application opened on its own worst artefact.
        """
        import app as app_module
        ref, abl = tmp_path / "results_current", tmp_path / "results_ablation"
        for d, base in ((ref, "fencing_clip3"), (abl, "ablation_180p")):
            d.mkdir()
            (d / f"{base}_distance.csv").write_text(
                "frame,time_s,distance_raw_m,distance_smooth_m,method\n"
                "0,0.0,2.0,2.0,pose\n")
        # Alphabetically results_ablation comes first; by intent it comes last.
        monkeypatch.setattr(app_module, "RESULTS_DIRS", [str(ref), str(abl)])
        monkeypatch.setattr(app_module, "store",
                            AnnotationStore(str(tmp_path / "ann_order")))
        app_module._bouts_cache["key"] = None
        ids = [b["bout_id"] for b in app_module.list_bouts()["bouts"]]
        assert ids[0].startswith("results_current:")


# -------------------- lunge labelling --------------------

class TestLungeLabels:
    """
    Lunge labels grade the pose model rather than correcting it, so they are not a
    fifth annotation action. B1h found pose stance features do not mark awarded
    touches, but touch times are a weak proxy for lunges in both directions, so
    these labels are what tests the hypothesis directly.
    """

    def test_a_lunge_is_stored_with_its_fencer(self, store):
        store.add_lunge("b", 12.5, 0)
        lunges = store.load("b")["lunges"]
        assert len(lunges) == 1
        assert lunges[0]["time_s"] == 12.5 and lunges[0]["slot"] == 0

    def test_sub_second_precision_survives(self):
        """
        A lunge lasts about ten frames at 30 fps, so the peak has to be recordable
        to better than a second or the label cannot land on the right frame.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            st = AnnotationStore(d)
            st.add_lunge("b", 12.567, 1)
            assert st.load("b")["lunges"][0]["time_s"] == 12.567

    def test_ids_are_unique_across_several_labels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            st = AnnotationStore(d)
            for i in range(5):
                st.add_lunge("b", float(i), i % 2)
            ids = [l["id"] for l in st.load("b")["lunges"]]
            assert len(set(ids)) == 5

    def test_a_bad_slot_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.add_lunge("b", 1.0, 2)

    def test_a_negative_time_is_rejected(self, store):
        with pytest.raises(ValueError):
            store.add_lunge("b", -1.0, 0)

    def test_removal(self, store):
        store.add_lunge("b", 1.0, 0)
        store.add_lunge("b", 2.0, 1)
        assert store.remove_lunge("b", "l0") is True
        remaining = store.load("b")["lunges"]
        assert [l["id"] for l in remaining] == ["l1"]

    def test_removing_an_unknown_lunge_reports_false(self, store):
        assert store.remove_lunge("b", "nope") is False

    def test_a_file_from_before_lunges_existed_still_loads(self, tmp_path):
        """
        Annotation files already exist on disk without this key, and a review
        session must not fail on one.
        """
        root = tmp_path / "ann"
        root.mkdir()
        (root / "b.json").write_text(json.dumps(
            {"bout_id": "b", "touch_states": {}, "added_touches": []}))
        assert AnnotationStore(str(root)).load("b")["lunges"] == []

    def test_the_api_returns_lunges_sorted_by_time(self, metrics_bout):
        app_module, bout_id = metrics_bout
        for t in (30.0, 10.0, 20.0):
            app_module.store.add_lunge(bout_id, t, 0)
        times = [l["time_s"] for l in app_module.get_touches(bout_id)["lunges"]]
        assert times == [10.0, 20.0, 30.0]

    def test_labelling_does_not_disturb_the_four_designed_actions(self, store):
        """
        Grading the model must not change the state the user's corrections live in,
        or an evaluation session would quietly alter the thing being evaluated.
        """
        store.set_touch_state("b", "p0", CONFIRMED, scorer="left")
        store.add_unreliable_segment("b", 1.0, 2.0)
        before = store.load("b")
        snapshot = (dict(before["touch_states"]), list(before["unreliable_segments"]))
        store.add_lunge("b", 5.0, 1)
        after = store.load("b")
        assert (after["touch_states"], after["unreliable_segments"]) == snapshot


# -------------------- exporting confirmed touches --------------------

class TestExportConfirmedTouches:
    """
    The loop was open at the last step: a user could confirm and correct touches,
    and nothing downstream could read the result, so the summary a reader sees was
    still built from unreviewed detector output. That undercuts the project's
    central claim, which is that user correction improves the output.
    """

    def _confirm_first(self, app_module, bout_id):
        proposed = load_proposed_touches(app_module._get_bout(bout_id).touches_csv)
        app_module.store.set_touch_state(bout_id, proposed[0]["id"], CONFIRMED,
                                        scorer="left")
        return proposed

    def test_export_writes_a_file_the_pipeline_can_read(self, metrics_bout):
        """
        The schema has to match ground_truth/*_touches.csv, because in_play.py and
        generate_summary.py take a touch file by path and must accept this one
        unchanged.
        """
        app_module, bout_id = metrics_bout
        self._confirm_first(app_module, bout_id)
        r = app_module.export_touches(bout_id)

        import csv as _csv
        with open(r["path"]) as f:
            rows = list(_csv.DictReader(l for l in f if not l.startswith("#")))
        assert set(rows[0]) == {"time_s", "scorer", "annulled", "notes"}
        assert float(rows[0]["time_s"]) == 10.0
        assert rows[0]["scorer"] == "left"

    def test_in_play_accepts_the_exported_file(self, metrics_bout):
        """
        The whole point is that the export is consumable, so read it back with the
        real loader rather than trusting the header row.
        """
        from in_play import load_touch_times, touch_provenance
        app_module, bout_id = metrics_bout
        self._confirm_first(app_module, bout_id)
        r = app_module.export_touches(bout_id)
        assert load_touch_times(r["path"]) == [10.0]
        # a scorer column means these read as confirmed rather than as proposals,
        # which is what they are, and it changes what the summary may claim
        assert touch_provenance(r["path"]) == "human_confirmed"

    def test_pending_proposals_are_excluded_and_flagged(self, metrics_bout):
        """
        A proposal nobody has looked at is not evidence, so it must not appear. That
        makes a partial export a lower bound, and the file has to say so or a reader
        will treat the count as complete.
        """
        app_module, bout_id = metrics_bout
        self._confirm_first(app_module, bout_id)      # one of two confirmed
        r = app_module.export_touches(bout_id)
        assert r["review_complete"] is False
        header = "".join(l for l in open(r["path"]) if l.startswith("#"))
        assert "REVIEW INCOMPLETE" in header

    def test_a_finished_review_is_not_flagged_incomplete(self, metrics_bout):
        app_module, bout_id = metrics_bout
        proposed = self._confirm_first(app_module, bout_id)
        app_module.store.set_touch_state(bout_id, proposed[1]["id"], REJECTED)
        r = app_module.export_touches(bout_id)
        assert r["review_complete"] is True
        assert "REVIEW INCOMPLETE" not in open(r["path"]).read()

    def test_manually_added_touches_are_included(self, metrics_bout):
        app_module, bout_id = metrics_bout
        self._confirm_first(app_module, bout_id)
        app_module.store.add_touch(bout_id, 55.5, scorer="double")
        r = app_module.export_touches(bout_id)
        import csv as _csv
        with open(r["path"]) as f:
            rows = list(_csv.DictReader(l for l in f if not l.startswith("#")))
        assert [float(x["time_s"]) for x in rows] == [10.0, 55.5]
        assert rows[1]["scorer"] == "double"

    def test_exporting_nothing_is_refused(self, metrics_bout):
        """
        An empty touch file would scope every metric to nothing, which is a worse
        failure than having no file: it looks like a valid result.
        """
        from fastapi import HTTPException
        app_module, bout_id = metrics_bout
        with pytest.raises(HTTPException) as e:
            app_module.export_touches(bout_id)
        assert e.value.status_code == 400

    def test_the_pipeline_artefacts_are_not_modified(self, metrics_bout, results_dir):
        app_module, bout_id = metrics_bout
        before = {p.name: p.read_text() for p in results_dir.iterdir()}
        self._confirm_first(app_module, bout_id)
        app_module.export_touches(bout_id)
        for name, text in before.items():
            assert (results_dir / name).read_text() == text, f"{name} was modified"


# -------------------- reading the generated summary --------------------

class TestSummaryEndpoint:
    """
    A summary is prose on disk with no link to the data it describes, so re-running
    detection leaves a confident paragraph about numbers that no longer exist. The
    movement metrics have been redefined three times on this project, so this is a
    hazard that has already had three chances to bite.
    """

    def _write_summary(self, results_dir, stats):
        base = results_dir / "mybout_distance"
        (results_dir / "mybout_distance_summary.md").write_text(
            "## Bout summary\nThe fencers held lunge distance.\n")
        (results_dir / "mybout_distance_summary.meta.json").write_text(
            json.dumps({"cache_key": "abc", "model": "claude-opus-4-8",
                        "stats": stats}))

    def test_absent_summary_reports_how_to_make_one(self, metrics_bout):
        app_module, bout_id = metrics_bout
        r = app_module.get_summary(bout_id)
        assert r["exists"] is False
        assert "generate_summary.py" in r["hint"]

    def test_a_current_summary_is_not_stale(self, metrics_bout, results_dir):
        from generate_summary import compute_stats, load_rows
        app_module, bout_id = metrics_bout
        b = app_module._get_bout(bout_id)
        self._write_summary(results_dir, compute_stats(load_rows(b.metrics_csv)))
        r = app_module.get_summary(bout_id)
        assert r["exists"] is True and r["stale"] is False
        assert r["model"] == "claude-opus-4-8"
        assert "lunge distance" in r["markdown"]

    def test_changed_metrics_make_it_stale_and_say_which(self, metrics_bout, results_dir):
        """
        The reason matters as much as the flag. "Stale" alone tells a reader to
        regenerate; naming the field that moved tells them whether it mattered.
        """
        from generate_summary import compute_stats, load_rows
        app_module, bout_id = metrics_bout
        b = app_module._get_bout(bout_id)
        stats = compute_stats(load_rows(b.metrics_csv))
        stats["fencer_1"]["net_displacement_m"] += 5.0     # as if remeasured
        self._write_summary(results_dir, stats)
        r = app_module.get_summary(bout_id)
        assert r["stale"] is True
        assert "fencer_1" in r["stale_reason"]

    def test_a_summary_with_no_cached_stats_is_stale(self, metrics_bout, results_dir):
        """Nothing to compare against is not the same as matching."""
        app_module, bout_id = metrics_bout
        (results_dir / "mybout_distance_summary.md").write_text("## Bout summary\n")
        (results_dir / "mybout_distance_summary.meta.json").write_text(
            json.dumps({"cache_key": "abc", "model": "m"}))
        r = app_module.get_summary(bout_id)
        assert r["stale"] is True and "no cached stats" in r["stale_reason"]

    def test_a_corrupt_meta_file_does_not_break_the_read(self, metrics_bout, results_dir):
        """
        The summary itself is still readable and useful, so a damaged sidecar must
        degrade to "stale, unknown" rather than failing the request.
        """
        app_module, bout_id = metrics_bout
        (results_dir / "mybout_distance_summary.md").write_text("## Bout summary\nx\n")
        (results_dir / "mybout_distance_summary.meta.json").write_text("{not json")
        r = app_module.get_summary(bout_id)
        assert r["exists"] is True and r["stale"] is True

    def test_a_differing_touch_file_does_not_count_as_stale(self, metrics_bout, results_dir):
        """
        A summary may have been generated from ground truth, detector output or an
        exported review. Disagreeing with whichever file is on disk now is not the
        same as describing out-of-date metrics, and conflating them would mark every
        summary stale forever.
        """
        from generate_summary import compute_stats, load_rows
        app_module, bout_id = metrics_bout
        b = app_module._get_bout(bout_id)
        stats = compute_stats(load_rows(b.metrics_csv))
        stats["touches"] = {"count": 99, "times_s": [1.0], "provenance": "whatever"}
        self._write_summary(results_dir, stats)
        assert app_module.get_summary(bout_id)["stale"] is False

    def test_only_one_route_serves_the_summary_path(self):
        """
        Two handlers were registered on this path at one point and the second was
        unreachable, so the endpoint's behaviour depended on definition order.
        """
        import app as app_module
        paths = [r.path for r in app_module.app.routes
                 if getattr(r, "path", "") == "/api/bouts/{bout_id}/summary"]
        assert len(paths) == 1


class TestRecordIdsStayUniqueAmongLiveRecords:
    """
    Ids were derived from list length, so removing a record lowered the count and the
    next insert reused an id that was still live. Two records then shared one id and
    a delete took both. Found in real data after a review session that used the
    remove button: two `l1` records and a hole where `l32` had been. All four record
    types had the same line, so all four are covered.

    The guarantee asserted here is uniqueness among records PRESENT, not that an id
    is never issued twice in the lifetime of a bout. An id freed by a removal may
    come round again, which is harmless because the interface reloads after every
    mutation and so never holds a stale id.
    """

    def test_no_two_live_lunges_share_an_id(self, store):
        store.add_lunge("b", 1.0, 0)
        store.add_lunge("b", 2.0, 0)
        store.remove_lunge("b", "l0")          # free an id from the middle
        store.add_lunge("b", 3.0, 0)
        store.add_lunge("b", 4.0, 0)
        ids = [l["id"] for l in store.load("b")["lunges"]]
        assert len(set(ids)) == len(ids), ids

    def test_removing_one_record_cannot_take_another(self, store):
        """The failure this actually caused, asserted directly."""
        for t in (1.0, 2.0, 3.0):
            store.add_lunge("b", t, 0)
        store.remove_lunge("b", "l1")
        store.add_lunge("b", 4.0, 0)           # reuses l1 under the old code
        before = len(store.load("b")["lunges"])
        store.remove_lunge("b", store.load("b")["lunges"][-1]["id"])
        assert len(store.load("b")["lunges"]) == before - 1

    def test_the_old_length_scheme_would_have_failed_this(self, store):
        """
        Pins the specific sequence, so a revert to len()-based ids fails here rather
        than silently corrupting a review session.
        """
        store.add_lunge("b", 1.0, 0)           # l0
        store.add_lunge("b", 2.0, 0)           # l1
        store.remove_lunge("b", "l1")          # len() now 1, so len() gives l1 again
        store.add_lunge("b", 3.0, 0)
        live = store.load("b")["lunges"]
        assert len({l["id"] for l in live}) == 2

    def test_added_touch_ids_stay_unique(self, store):
        store.add_touch("b", 1.0)
        store.add_touch("b", 2.0)
        store.remove_added_touch("b", "u0")
        store.add_touch("b", 3.0)
        ids = [t["id"] for t in store.load("b")["added_touches"]]
        assert len(set(ids)) == len(ids), ids

    def test_segment_ids_stay_unique(self, store):
        store.add_unreliable_segment("b", 1.0, 2.0)
        store.add_unreliable_segment("b", 3.0, 4.0)
        store.remove_unreliable_segment("b", "s0")
        store.add_unreliable_segment("b", 5.0, 6.0)
        ids = [x["id"] for x in store.load("b")["unreliable_segments"]]
        assert len(set(ids)) == len(ids), ids

    def test_ids_restart_when_the_list_is_emptied(self, store):
        """Nothing is left to collide with, so restarting is safe and tidy."""
        store.add_lunge("b", 1.0, 0)
        store.remove_lunge("b", "l0")
        store.add_lunge("b", 2.0, 0)
        assert [l["id"] for l in store.load("b")["lunges"]] == ["l0"]


class TestReanchorExport:
    """
    Action 4 was stored and never consumed, so the interface offered a repair that
    did nothing. The export is what makes it reach the pipeline.
    """

    def test_export_writes_what_the_pipeline_expects(self, metrics_bout):
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 12.5, 1, 640.0, 300.0)
        r = app_module.export_reanchors(bout_id)
        entries = json.loads(open(r["path"]).read())
        assert entries == [{"time_s": 12.5, "slot": 1, "x": 640.0, "y": 300.0}]
        assert "--reanchors" in r["next"]

    def test_the_pipeline_loader_accepts_the_export(self, metrics_bout):
        """The two sides have to agree, so read it back with the real loader."""
        from run_detection import load_reanchors
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 2.0, 0, 100.0, 200.0)
        r = app_module.export_reanchors(bout_id)
        assert load_reanchors(r["path"], 30.0) == {60: [(0, 100.0, 200.0)]}

    def test_already_applied_corrections_are_still_exported(self, metrics_bout):
        """
        A rerun starts from the original video every time, so every correction is
        needed on every run. Filtering out the applied ones would silently undo them.
        """
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 10.0, 20.0)
        app_module.store.mark_reanchors_applied(bout_id)
        app_module.store.add_reanchor(bout_id, 2.0, 1, 30.0, 40.0)
        r = app_module.export_reanchors(bout_id)
        assert r["corrections"] == 2 and r["pending"] == 1

    def test_exporting_nothing_is_refused(self, metrics_bout):
        from fastapi import HTTPException
        app_module, bout_id = metrics_bout
        with pytest.raises(HTTPException) as e:
            app_module.export_reanchors(bout_id)
        assert e.value.status_code == 400

    def test_marking_applied_clears_the_pending_count(self, metrics_bout):
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 10.0, 20.0)
        assert app_module.mark_reanchors_applied(bout_id)["marked"] == 1
        data = app_module.store.load(bout_id)
        assert all(a["applied"] for a in data["reanchors"])
        # idempotent: saying it twice marks nothing further
        assert app_module.mark_reanchors_applied(bout_id)["marked"] == 0


class TestReanchorRerunCommand:
    """
    The export hands the user a command to run. Getting it wrong is worse than
    omitting it, because a command that looks authoritative will be pasted.
    """

    def test_the_command_names_the_source_video_not_the_annotated_one(self, metrics_bout, tmp_path, monkeypatch):
        """
        b.video is the annotated output. Rerunning detection over a clip that already
        has boxes and text burnt into it would be detecting on top of the overlay.
        """
        import app as app_module
        monkeypatch.setattr(app_module, "PROTOTYPE_DIR", str(tmp_path))
        (tmp_path / "mybout.mp4").write_bytes(b"x")
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 5.0, 6.0)
        r = app_module.export_reanchors(bout_id)
        assert r["source_video_found"] is True
        assert "mybout.mp4" in r["next"]
        assert "_annotated" not in r["next"]

    def test_a_missing_source_is_reported_rather_than_faked(self, metrics_bout, tmp_path, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "PROTOTYPE_DIR", str(tmp_path))
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 5.0, 6.0)
        assert app_module.export_reanchors(bout_id)["source_video_found"] is False

    def test_the_piste_config_is_carried_into_the_command(self, metrics_bout, tmp_path, monkeypatch):
        """
        A rerun without the piste configuration changes two things at once. Omitting
        it once cost 18 points of coverage on clip 2 and read as a code regression.
        """
        import app as app_module
        monkeypatch.setattr(app_module, "PROTOTYPE_DIR", str(tmp_path))
        (tmp_path / "mybout.mp4").write_bytes(b"x")
        (tmp_path / "piste_mybout.json").write_text("{}")
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 5.0, 6.0)
        r = app_module.export_reanchors(bout_id)
        assert r["piste_config"] == "piste_mybout.json"
        assert "--piste-config piste_mybout.json" in r["next"]

    def test_no_piste_config_means_none_is_suggested(self, metrics_bout, tmp_path, monkeypatch):
        """Clip 3 has no configuration, and inventing one would change the run."""
        import app as app_module
        monkeypatch.setattr(app_module, "PROTOTYPE_DIR", str(tmp_path))
        (tmp_path / "mybout.mp4").write_bytes(b"x")
        app_module, bout_id = metrics_bout
        app_module.store.add_reanchor(bout_id, 1.0, 0, 5.0, 6.0)
        r = app_module.export_reanchors(bout_id)
        assert r["piste_config"] is None
        assert "--piste-config" not in r["next"]


class TestReviewSessions:
    """
    The effort measurement. The project's central claim is about effort and
    nothing in it measured effort until this existed, so these tests are mostly
    about the recorded figure meaning what it says.
    """

    def test_records_a_run_and_its_rate(self, store):
        store.add_session("b", "assisted", 60.0, 15)
        c = store.session_comparison("b")
        assert c["assisted"]["runs"] == 1
        assert c["assisted"]["mean_seconds_per_decision"] == 4.0

    def test_the_rate_is_per_decision_not_per_bout(self, store):
        # The two modes do not produce the same NUMBER of decisions: assisted
        # answers one question per proposal, manual creates one entry per touch
        # found. A per-bout figure would compare different amounts of work.
        store.add_session("b", "assisted", 100.0, 20)
        store.add_session("b", "manual", 100.0, 5)
        c = store.session_comparison("b")
        assert c["assisted"]["mean_seconds_per_decision"] == 5.0
        assert c["manual"]["mean_seconds_per_decision"] == 20.0
        assert c["speedup"] == 4.0

    def test_one_run_each_is_labelled_as_an_illustration(self, store):
        """
        A speed-up computed from a single pass in each mode is an anecdote about
        one afternoon, and it will be read as a result unless it says otherwise.
        """
        store.add_session("b", "assisted", 60.0, 15)
        store.add_session("b", "manual", 60.0, 5)
        c = store.session_comparison("b")
        assert "illustration, not a measurement" in c["strength"]

    def test_more_runs_stop_it_being_called_an_illustration(self, store):
        for _ in range(2):
            store.add_session("b", "assisted", 60.0, 15)
        store.add_session("b", "manual", 60.0, 5)
        c = store.session_comparison("b")
        assert "illustration" not in c["strength"]
        assert "2 assisted" in c["strength"]

    def test_no_speedup_until_both_modes_have_a_run(self, store):
        store.add_session("b", "assisted", 60.0, 15)
        c = store.session_comparison("b")
        assert c["speedup"] is None
        assert "manual" in c["strength"]

    def test_runs_accumulate_rather_than_replacing(self, store):
        # Including the ones that went badly. Keeping only the latest would keep
        # only the runs the user was happy with, which is the shape of a result
        # that flatters itself.
        store.add_session("b", "assisted", 60.0, 10)
        store.add_session("b", "assisted", 200.0, 10)
        c = store.session_comparison("b")
        assert c["assisted"]["runs"] == 2
        assert c["assisted"]["mean_seconds_per_decision"] == 13.0

    def test_a_run_with_no_decisions_has_no_rate(self, store):
        # It is recorded, because a pass that produced nothing is a real event,
        # but it cannot contribute a rate and must not contribute a zero.
        store.add_session("b", "manual", 60.0, 0)
        c = store.session_comparison("b")
        assert c["manual"]["runs"] == 1
        assert c["manual"]["mean_seconds_per_decision"] is None

    def test_rejects_an_unknown_mode(self, store):
        # The comparison is only meaningful between the two conditions it was
        # designed around; free text would let a third one into a table of two.
        with pytest.raises(ValueError):
            store.add_session("b", "quick", 60.0, 10)

    def test_rejects_a_zero_or_negative_duration(self, store):
        with pytest.raises(ValueError):
            store.add_session("b", "manual", 0.0, 10)

    def test_sessions_survive_a_file_written_before_they_existed(self, store):
        store.add_touch("b", 10.0)
        assert store.load("b")["sessions"] == []


class TestGreenLampSide:
    """Which side the green lamp is on is a fact about the RECORDING.

    Nothing in the image reveals it, it never changes for a given bout, and
    every attributed touch depends on it. It used to be sent with each request
    and thrown away, so it had to be re-entered after every reload, and a
    misremembered answer silently inverts who scored every touch. The report
    describes it as confirmed once per bout, which was true of the intention
    and not of the system.
    """

    def test_it_survives_a_reload(self, tmp_path):
        s = AnnotationStore(str(tmp_path / "ann"))
        assert s.load("b")["green_is"] is None
        s.set_green_is("b", "right")
        assert AnnotationStore(str(tmp_path / "ann")).load("b")["green_is"] == "right"

    def test_it_can_be_corrected(self, tmp_path):
        s = AnnotationStore(str(tmp_path / "ann"))
        s.set_green_is("b", "left")
        s.set_green_is("b", "right")
        assert s.load("b")["green_is"] == "right"

    def test_it_refuses_anything_but_a_side(self, tmp_path):
        s = AnnotationStore(str(tmp_path / "ann"))
        with pytest.raises(ValueError, match="left or right"):
            s.set_green_is("b", "green")
        assert s.load("b")["green_is"] is None

    def test_a_file_from_before_this_existed_still_loads(self, tmp_path):
        """Older annotation files have no such key and must not break."""
        root = tmp_path / "ann"
        os.makedirs(root)
        (root / "b.json").write_text('{"bout_id": "b", "touch_states": {}}')
        assert AnnotationStore(str(root)).load("b")["green_is"] is None


class TestDeletingAJobDoesNotDeleteTheEvaluationSet:
    """The worst defect found on this project, and the only one that destroyed
    data rather than producing a wrong number.

    A summary job sets output_dir to the directory holding the metrics CSV. For
    an evaluation bout that is a shared results directory containing every other
    bout, so deleting the job card ran shutil.rmtree over the whole reference
    set: nine bouts, from one click. It was recoverable only because most of
    those files were in version control.
    """

    def _app(self, tmp_path, monkeypatch, out_dir):
        import app as app_module
        monkeypatch.setattr(app_module, "UPLOAD_ROOT", str(tmp_path / "uploads"))
        monkeypatch.setattr(app_module, "UPLOAD_RESULTS_ROOT",
                            str(tmp_path / "results_uploads"))
        job = {"job_id": "j1", "state": "done", "output_dir": str(out_dir)}
        monkeypatch.setattr(app_module.job_store, "load", lambda i: job)
        monkeypatch.setattr(app_module.job_store, "delete", lambda i: None)
        return app_module

    def test_a_shared_results_directory_survives(self, tmp_path, monkeypatch):
        shared = tmp_path / "results_current"
        shared.mkdir()
        (shared / "fencing_clip3_distance.csv").write_text("frame,time_s\n")
        app_module = self._app(tmp_path, monkeypatch, shared)
        app_module.delete_job("j1")
        assert shared.exists(), "deleting a job removed the evaluation set"
        assert (shared / "fencing_clip3_distance.csv").exists()

    def test_the_job_s_own_output_is_still_removed(self, tmp_path, monkeypatch):
        owned = tmp_path / "results_uploads" / "upload_j1"
        owned.mkdir(parents=True)
        (owned / "out.csv").write_text("x")
        app_module = self._app(tmp_path, monkeypatch, owned)
        app_module.delete_job("j1")
        assert not owned.exists(), "a job must still clean up after itself"


class TestPlaybackSurvivesALostRender:
    """The annotated render is large and not in version control; the browser
    transcode is small and is what actually gets served. When a render was
    destroyed, every bout reported no video although a playable copy of each was
    sitting on disk. Refusing to play a file that is there is a self-inflicted
    failure.
    """

    def test_a_cached_copy_is_found_when_the_render_is_gone(self, tmp_path, monkeypatch):
        import app as app_module
        wv = tmp_path / "web_video"
        wv.mkdir()
        (wv / "results_current__fencing_clip3_annotated.h264.mp4").write_bytes(b"x")
        monkeypatch.setattr(app_module, "WEB_VIDEO_DIR", str(wv))
        assert app_module._cached_playable("results_current:fencing_clip3")

    def test_nothing_is_invented_when_there_is_no_copy(self, tmp_path, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "WEB_VIDEO_DIR", str(tmp_path / "empty"))
        assert app_module._cached_playable("results_current:fencing_clip3") == ""

    def test_a_malformed_bout_id_is_not_treated_as_a_path(self, tmp_path, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "WEB_VIDEO_DIR", str(tmp_path))
        assert app_module._cached_playable("no-colon-here") == ""
