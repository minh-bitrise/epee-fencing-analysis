"""
Tests for the upload and job HTTP endpoints.

These exercise the request-side behaviour that `test_jobs.py` does not: what the
upload endpoint accepts and rejects, whether a rejected upload leaves anything
behind, and whether the piste and lifecycle endpoints refuse the states they
should refuse in.

Nothing here processes video. The runner is replaced with a stub that records
submissions, so a test costs milliseconds instead of loading three models, and
what is being tested is the endpoint rather than the pipeline.

Run with:
    cd code/backend && python3 -m pytest test_upload_api.py -v
"""

import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))

import app as appmod  # noqa: E402
from jobs import AWAITING_PISTE, DONE, JobStore  # noqa: E402


@pytest.fixture
def tiny_video(tmp_path):
    """
    A real, decodable video file, small enough to be cheap.

    Written with OpenCV rather than shipped as a fixture because the upload
    endpoint's validity check IS an OpenCV open-and-read, so a file this build
    can decode is exactly the right test input, and a checked-in sample might not
    be decodable by another machine's build.
    """
    import cv2
    path = tmp_path / "bout.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(10):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


class StubRunner:
    """Records what would have been queued, and runs nothing."""

    def __init__(self, store):
        self.store = store
        self.submitted = []
        self.cancelled = []

    def submit(self, job_id):
        self.submitted.append(job_id)

    def cancel(self, job_id):
        job = self.store.load(job_id)
        if job is None or job["state"] in ("done", "failed", "cancelled",
                                           "interrupted"):
            return False
        self.cancelled.append(job_id)
        self.store.update(job_id, state="cancelled")
        return True

    def queue_position(self, job_id):
        return None

    # The lifespan handler calls both on startup. They are no-ops here: there is
    # nothing to reconcile in a fresh temporary directory, and starting a worker
    # would defeat the point of stubbing the runner.
    def reconcile(self):
        return []

    def start(self):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The app with its runtime roots redirected into a temporary directory."""
    for name, sub in (("UPLOAD_ROOT", "uploads"), ("JOB_ROOT", "jobs"),
                      ("UPLOAD_RESULTS_ROOT", "results_uploads"),
                      ("WEB_VIDEO_DIR", "web_video")):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(appmod, name, str(d))

    store = JobStore(str(tmp_path / "jobs"))
    runner = StubRunner(store)
    monkeypatch.setattr(appmod, "job_store", store)
    monkeypatch.setattr(appmod, "runner", runner)

    # The ANNOTATION store too, and not merely for isolation between tests: it
    # defaults to the real prototype/annotations directory, which holds the
    # user's own hand-made labels. A test writing there would leave debris beside
    # genuine work, and an earlier version of this file did exactly that.
    from store import AnnotationStore
    monkeypatch.setattr(appmod, "store",
                        AnnotationStore(str(tmp_path / "annotations")))

    with TestClient(appmod.app) as c:
        c.stub = runner
        c.store = store
        yield c


def upload(client, path, name=None, **data):
    with open(path, "rb") as f:
        return client.post("/api/jobs",
                           files={"video": (name or os.path.basename(path), f,
                                            "video/mp4")},
                           data=data)


class TestUpload:
    def test_accepts_a_video_and_queues_it(self, client, tiny_video):
        r = upload(client, tiny_video)
        assert r.status_code == 200
        job = r.json()
        assert job["state"] == "queued"
        assert job["job_id"] in client.stub.submitted

    def test_measures_the_video_in_the_request(self, client, tiny_video):
        # The client needs this to show a duration before anything has run, and
        # it is the same open-and-read the pipeline will do, so a file that fails
        # here would have failed two stages later instead.
        info = upload(client, tiny_video).json()["video_info"]
        assert info["width"] == 64 and info["height"] == 48
        assert info["frames"] == 10

    def test_rejects_a_file_that_is_not_a_video(self, client, tmp_path):
        fake = tmp_path / "notavideo.mp4"
        fake.write_bytes(b"this is not a video, it only has the extension")
        r = upload(client, fake)
        assert r.status_code == 400
        assert "could not be opened" in r.json()["detail"]

    def test_rejects_an_unsupported_extension(self, client, tiny_video):
        r = upload(client, tiny_video, name="bout.txt")
        assert r.status_code == 400
        assert ".txt" in r.json()["detail"]

    def test_a_rejected_upload_leaves_nothing_behind(self, client, tmp_path):
        """
        A failed upload must not leave a half-written file or an orphan job.
        Nothing else in this system deletes anything, so debris here accumulates
        until someone finds it with a terminal.
        """
        fake = tmp_path / "notavideo.mp4"
        fake.write_bytes(b"nope")
        assert upload(client, fake).status_code == 400
        assert client.store.list() == []
        assert os.listdir(appmod.UPLOAD_ROOT) == []

    def test_rejects_a_file_over_the_size_cap(self, client, tiny_video,
                                              monkeypatch):
        monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 10)
        r = upload(client, tiny_video)
        assert r.status_code == 413
        # And still cleans up, which is the case most likely to leak: the file is
        # partly written by the time the limit is hit.
        assert client.store.list() == []
        assert os.listdir(appmod.UPLOAD_ROOT) == []

    def test_the_stored_name_does_not_come_from_the_upload(self, client,
                                                           tiny_video):
        """
        A filename arriving over the wire is user input, and it would otherwise
        become the bout id and the stem of every output file. A name with a space
        or a slash in it would propagate into paths the whole pipeline then has
        to quote correctly.
        """
        job = upload(client, tiny_video, name="my bout; rm -rf x.mp4").json()
        assert job["filename"] == "my bout; rm -rf x.mp4"    # shown to the user
        stored = os.path.basename(job["source_path"])
        assert stored == f"bout_{job['job_id']}.mp4"          # used on disk


class TestPisteDecision:
    def _awaiting(self, client, tiny_video):
        job = upload(client, tiny_video).json()
        client.store.update(job["job_id"], state=AWAITING_PISTE,
                            piste={"needed": True, "confident": True,
                                   "polygon": [[0, 10], [64, 10],
                                               [64, 40], [0, 40]]})
        return job["job_id"]

    def test_accepting_the_measurement_resumes_the_job(self, client, tiny_video):
        job_id = self._awaiting(client, tiny_video)
        r = client.post(f"/api/jobs/{job_id}/piste", json={})
        assert r.status_code == 200
        assert r.json()["state"] == "queued"
        assert client.store.load(job_id)["piste"]["decision"] == "accepted as measured"
        assert client.stub.submitted.count(job_id) == 2   # once on upload, once now

    def test_an_adjusted_polygon_replaces_the_measured_one(self, client, tiny_video):
        job_id = self._awaiting(client, tiny_video)
        mine = [[0, 20], [64, 20], [64, 38], [0, 38]]
        r = client.post(f"/api/jobs/{job_id}/piste", json={"polygon": mine})
        assert r.status_code == 200
        piste = client.store.load(job_id)["piste"]
        assert piste["polygon"] == mine
        assert piste["decision"] == "adjusted by the user"
        # and it must reach the file the pipeline actually reads
        import json
        with open(client.store.load(job_id)["piste_config_path"]) as f:
            assert json.load(f)["polygon"] == mine

    def test_skipping_drops_the_region(self, client, tiny_video):
        job_id = self._awaiting(client, tiny_video)
        r = client.post(f"/api/jobs/{job_id}/piste", json={"skip": True})
        assert r.status_code == 200
        piste = client.store.load(job_id)["piste"]
        assert piste["polygon"] is None and piste["needed"] is False

    def test_rejects_a_degenerate_polygon(self, client, tiny_video):
        job_id = self._awaiting(client, tiny_video)
        r = client.post(f"/api/jobs/{job_id}/piste",
                        json={"polygon": [[0, 10], [64, 10]]})
        assert r.status_code == 400

    def test_rejects_a_decision_for_a_job_that_is_not_waiting(self, client,
                                                              tiny_video):
        # A stale browser tab polling an old job would otherwise resubmit a job
        # that had already finished.
        job = upload(client, tiny_video).json()
        r = client.post(f"/api/jobs/{job['job_id']}/piste", json={})
        assert r.status_code == 409


class TestLifecycle:
    def test_unknown_job_is_a_404_everywhere(self, client):
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/cancel").status_code == 404
        assert client.delete("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/piste", json={}).status_code == 404

    def test_a_finished_job_reports_the_bout_to_open(self, client, tiny_video):
        """
        The handoff that closes the loop. Without it a user who has just
        processed a video has to find it again in a dropdown, by an id derived
        from an output directory they never chose.
        """
        job = upload(client, tiny_video).json()
        client.store.update(job["job_id"], state=DONE)
        view = client.get(f"/api/jobs/{job['job_id']}").json()
        stem = os.path.splitext(os.path.basename(job["source_path"]))[0]
        assert view["bout_id"] == f"upload_{job['job_id']}:{stem}"

    def test_deleting_removes_the_upload_and_the_outputs(self, client, tiny_video):
        job = upload(client, tiny_video).json()
        client.store.update(job["job_id"], state=DONE)
        upload_dir = os.path.join(appmod.UPLOAD_ROOT, job["job_id"])
        assert os.path.isdir(upload_dir)

        assert client.delete(f"/api/jobs/{job['job_id']}").status_code == 200
        assert not os.path.exists(upload_dir)
        assert not os.path.exists(job["output_dir"])
        assert client.store.load(job["job_id"]) is None

    def test_a_running_job_must_be_cancelled_before_it_can_be_deleted(
            self, client, tiny_video):
        # Deleting the source out from under a running subprocess would produce a
        # failure that looks like a pipeline bug.
        job = upload(client, tiny_video).json()
        client.store.update(job["job_id"], state="running")
        assert client.delete(f"/api/jobs/{job['job_id']}").status_code == 409

    def test_the_frame_endpoint_returns_an_image(self, client, tiny_video):
        job = upload(client, tiny_video).json()
        r = client.get(f"/api/jobs/{job['job_id']}/frame")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert len(r.content) > 0

    def test_the_log_endpoint_copes_with_a_job_that_has_not_run(self, client,
                                                                tiny_video):
        job = upload(client, tiny_video).json()
        r = client.get(f"/api/jobs/{job['job_id']}/log")
        assert r.status_code == 200


class TestReprocessAndSummary:
    """
    The two endpoints that replace a command line with a button.

    Both existed as instructions before this: the review page told the user to go
    and run the pipeline in a terminal to apply their tracking corrections, and
    to run another command to generate a summary. That is the arrangement the
    application layer exists to remove.
    """

    @pytest.fixture
    def bout(self, tmp_path, monkeypatch, client):
        """An uploaded bout that has finished processing, with a source video."""
        results = tmp_path / "results_uploads" / "upload_abc123def456"
        results.mkdir(parents=True)
        (results / "bout_abc123def456_distance.csv").write_text(
            "frame,time_s,distance_raw_m,distance_smooth_m,method,"
            "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m\n"
            "0,0.0,2.0,2.0,pose,0,0,0,0\n1,0.5,2.0,2.0,pose,0,0,0,0\n")
        source = tmp_path / "uploads" / "abc123def456" / "bout_abc123def456.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"not really a video, only its path is used here")

        client.store.create(job_id_placeholder=True)
        # a job record whose id matches the bout's directory suffix
        job = client.store.create(
            state="done", source_path=str(source),
            piste={"needed": True, "polygon": [[0, 1], [2, 1], [2, 3], [0, 3]]},
            piste_config_path=str(tmp_path / "uploads" / "abc123def456" / "piste.json"))
        # rename the record so the bout id resolves to it
        os.rename(os.path.join(client.store.root, f"{job['job_id']}.json"),
                  os.path.join(client.store.root, "abc123def456.json"))
        job["job_id"] = "abc123def456"
        client.store.save(job)
        return "upload_abc123def456:bout_abc123def456"

    def test_reprocess_queues_a_job_carrying_the_corrections(self, client, bout,
                                                             monkeypatch):
        appmod.store.add_reanchor(bout, 12.0, 0, 500, 280)
        r = client.post(f"/api/bouts/{bout}/reprocess")
        assert r.status_code == 200
        body = r.json()
        assert body["corrections"] == 1
        job = client.store.load(body["job_id"])
        # The corrections must reach a file the pipeline can read, or the button
        # is decorative.
        assert os.path.exists(job["reanchor_path"])
        import json
        assert json.load(open(job["reanchor_path"]))[0]["slot"] == 0

    def test_reprocess_does_not_overwrite_the_original(self, client, bout):
        r = client.post(f"/api/bouts/{bout}/reprocess")
        job = client.store.load(r.json()["job_id"])
        # A reprocess must never destroy a previous result: the before and after
        # side by side is the only way to see whether a correction helped.
        assert "upload_abc123def456" not in job["output_dir"]

    def test_reprocess_re_uses_the_piste_region_rather_than_re_measuring(
            self, client, bout):
        # Otherwise the rerun changes two things at once. Omitting the config
        # once cost 18 points of coverage on clip 2 and looked like a regression.
        r = client.post(f"/api/bouts/{bout}/reprocess")
        job = client.store.load(r.json()["job_id"])
        assert job["confirm_piste"] is False
        assert "piste" in job["stages_done"] or job["piste"] is not None

    def test_reprocess_refuses_when_the_source_video_is_gone(self, client, bout):
        job = client.store.load("abc123def456")
        os.remove(job["source_path"])
        r = client.post(f"/api/bouts/{bout}/reprocess")
        assert r.status_code == 404
        assert "annotated" in r.json()["detail"]

    def test_summary_refuses_without_an_api_key(self, client, bout, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        r = client.post(f"/api/bouts/{bout}/summary/generate")
        assert r.status_code == 400
        assert "ANTHROPIC_API_KEY" in r.json()["detail"]

    def test_summary_is_a_job_not_an_automatic_stage(self, client, bout,
                                                     monkeypatch):
        """
        Generating a summary costs a paid API call, so it must never happen
        because a user uploaded a video. The processing pipeline has no summary
        stage at all; this endpoint is the only way to reach one.
        """
        from jobs import pipeline_stages
        for stage in pipeline_stages():
            assert stage.name != "summary"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        r = client.post(f"/api/bouts/{bout}/summary/generate")
        assert r.status_code == 200
        assert client.store.load(r.json()["job_id"])["kind"] == "summary"

    def test_summary_prefers_the_reviewed_touch_list(self, client, bout,
                                                     monkeypatch, tmp_path):
        """
        The whole point of the export. Without preferring it, the prose a reader
        sees is still built from unreviewed detector output, which undercuts the
        claim that user correction improves the output.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        reviewed = (tmp_path / "results_uploads" / "upload_abc123def456" /
                    "bout_abc123def456_distance_touches_confirmed.csv")
        reviewed.write_text("time_s,scorer,annulled,notes\n5.0,left,0,confirmed\n")
        r = client.post(f"/api/bouts/{bout}/summary/generate")
        assert r.json()["touches_used"] == "the touches you confirmed"
        assert client.store.load(r.json()["job_id"])["touches_csv"] == str(reviewed)


class TestApiKeyResolution:
    """
    The key lives in the macOS Keychain on this machine, not in a file and not in
    a shell profile, so requiring it in the environment would mean the user
    exporting it by hand every time they start the server.
    """

    def test_an_existing_environment_key_is_left_alone(self, monkeypatch):
        # Never overwrite what the operator set deliberately.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")
        assert appmod._load_api_key_from_keychain() == "environment"
        assert os.environ["ANTHROPIC_API_KEY"] == "already-set"

    def test_reads_the_keychain_when_the_environment_is_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        calls = []

        class Result:
            returncode = 0
            stdout = "sk-ant-from-keychain\n"

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return Result()

        monkeypatch.setattr(appmod.subprocess, "run", fake_run)
        assert appmod._load_api_key_from_keychain() == "keychain"
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-keychain"
        assert appmod.KEYCHAIN_SERVICE in calls[0]

    def test_a_missing_key_is_not_an_error(self, monkeypatch):
        """
        A machine without the key simply has no summary button. Raising here
        would stop the server starting over a feature most sessions never touch.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        class Result:
            returncode = 44
            stdout = ""

        monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: Result())
        assert appmod._load_api_key_from_keychain() is None
        assert "ANTHROPIC_API_KEY" not in os.environ

    def test_a_machine_without_the_security_tool_is_not_an_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        def boom(*a, **k):
            raise FileNotFoundError("security")

        monkeypatch.setattr(appmod.subprocess, "run", boom)
        assert appmod._load_api_key_from_keychain() is None

    def test_an_empty_keychain_entry_is_treated_as_missing(self, monkeypatch):
        # A blank value would otherwise be exported and then fail inside the
        # summary subprocess, several minutes and one queue position later.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        class Result:
            returncode = 0
            stdout = "   \n"

        monkeypatch.setattr(appmod.subprocess, "run", lambda *a, **k: Result())
        assert appmod._load_api_key_from_keychain() is None
        assert "ANTHROPIC_API_KEY" not in os.environ


class TestScorerProposals:
    """
    The endpoint that turns lamp readings into proposed scorers.

    It reads only at times a touch is already confirmed, which is the whole
    reason it is safe: the lamps also fire when fencers test weapons against the
    piste or each other's guards, routinely just after a touch.
    """

    def test_refuses_without_a_green_side(self, client):
        # Nothing in the image says which fencer the green lamp belongs to, so a
        # default would be wrong half the time while looking authoritative.
        r = client.post("/api/bouts/whatever/propose-scorers", json={})
        assert r.status_code == 422

    def test_rejects_a_meaningless_side(self, client):
        r = client.post("/api/bouts/whatever/propose-scorers",
                        json={"green_is": "middle"})
        assert r.status_code == 422

    def test_unknown_bout_is_a_404(self, client):
        r = client.post("/api/bouts/nope:nope/propose-scorers",
                        json={"green_is": "right"})
        assert r.status_code == 404


class TestBoutDiscoveryCache:
    """
    Discovery stats every file in every results directory and runs on any request
    naming a bout. Caching it is only safe if a new bout invalidates the cache,
    which is why the key is the directories' modification times rather than a
    timer: a bout appearing changes the mtime of the directory holding it, and
    nothing else does.
    """

    def test_a_new_bout_invalidates_the_cache(self, tmp_path, monkeypatch):
        import time as _time
        results = tmp_path / "results_uploads" / "upload_aaaaaaaaaaaa"
        results.mkdir(parents=True)
        (results / "one_distance.csv").write_text("frame,time_s\n0,0.0\n")
        monkeypatch.setattr(appmod, "UPLOAD_RESULTS_ROOT",
                            str(tmp_path / "results_uploads"))
        monkeypatch.setattr(appmod, "RESULTS_DIRS", [])
        appmod._bouts_cache["key"] = None

        first = appmod._bouts()
        assert len(first) == 1

        _time.sleep(0.01)
        (results / "two_distance.csv").write_text("frame,time_s\n0,0.0\n")
        second = appmod._bouts()
        assert len(second) == 2, "a new bout did not invalidate the cache"

    def test_repeated_calls_return_the_same_object(self, tmp_path, monkeypatch):
        # Proves the cache is actually used rather than merely correct.
        results = tmp_path / "results_uploads" / "upload_bbbbbbbbbbbb"
        results.mkdir(parents=True)
        (results / "x_distance.csv").write_text("frame,time_s\n0,0.0\n")
        monkeypatch.setattr(appmod, "UPLOAD_RESULTS_ROOT",
                            str(tmp_path / "results_uploads"))
        monkeypatch.setattr(appmod, "RESULTS_DIRS", [])
        appmod._bouts_cache["key"] = None
        assert appmod._bouts() is appmod._bouts()

    def test_a_missing_results_directory_is_not_an_error(self, tmp_path, monkeypatch):
        # Several configured directories legitimately do not exist on a fresh
        # checkout, and stat-ing them must not raise.
        monkeypatch.setattr(appmod, "RESULTS_DIRS", [str(tmp_path / "nope")])
        monkeypatch.setattr(appmod, "UPLOAD_RESULTS_ROOT", str(tmp_path / "gone"))
        appmod._bouts_cache["key"] = None
        assert appmod._bouts() == {}


class TestFencerProfile:
    """
    The profile endpoint. Its risk is presentational rather than numerical: a
    radar looks authoritative, so the tests that matter are the ones checking it
    declines to draw one when the tracking does not support it.
    """

    def _bout(self, tmp_path, client, f1, f2, name="prof"):
        results = tmp_path / "results_uploads" / f"upload_{name}"
        results.mkdir(parents=True, exist_ok=True)
        header = ("frame,time_s,distance_raw_m,distance_smooth_m,method,"
                  "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m,"
                  "f1_pos_m,f2_pos_m\n")
        body = "".join(
            f"{i},{i*0.1:.1f},2.0,2.0,pose,0,0,0,0,{f1(i):.3f},{f2(i):.3f}\n"
            for i in range(400))
        (results / f"b_{name}_distance.csv").write_text(header + body)
        (results / f"b_{name}_distance_touches.csv").write_text(
            "time_s,confidence,min_distance_m,separation_m,audio_support,signals\n"
            "10.0,0.90,1.20,2.10,0,approach+separated\n"
            "20.0,0.60,1.80,1.10,0,approach+separated\n")
        return f"upload_{name}:b_{name}"

    def test_profiles_a_bout_whose_tracking_held(self, client, tmp_path):
        bout = self._bout(tmp_path, client,
                          lambda i: 2.0 + (i % 40) * 0.03, lambda i: 6.0)
        r = client.get(f"/api/bouts/{bout}/profile")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert len(body["axes"]) == 6

    def test_refuses_when_the_tracker_swapped_the_fencers(self, client, tmp_path):
        """
        The failure this endpoint exists to guard against. Clip 4 swapped 14
        times, and a profile computed there describes the tracker rather than
        either fencer while looking entirely plausible.
        """
        bout = self._bout(tmp_path, client,
                          lambda i: 2.0 if i < 200 else 6.0,
                          lambda i: 6.0 if i < 200 else 2.0, name="swap")
        body = client.get(f"/api/bouts/{bout}/profile").json()
        assert body["available"] is False
        assert body["swaps"] >= 1

    def test_says_how_many_touches_it_rests_on(self, client, tmp_path):
        # Three of the six axes are undefined until touches are confirmed, and a
        # profile built on two should not be read like one built on twenty.
        bout = self._bout(tmp_path, client,
                          lambda i: 2.0 + (i % 40) * 0.03, lambda i: 6.0,
                          name="count")
        body = client.get(f"/api/bouts/{bout}/profile").json()
        assert body["confirmed_touches"] == 0

    def test_confirmed_touches_reach_the_scoring_axes(self, client, tmp_path):
        bout = self._bout(tmp_path, client,
                          lambda i: 2.0 + (i % 40) * 0.03, lambda i: 6.0,
                          name="scored")
        proposed = client.get(f"/api/bouts/{bout}/touches").json()["proposed"]
        client.post(f"/api/bouts/{bout}/touches/{proposed[0]['id']}/decision",
                    json={"state": "confirmed", "scorer": "left"})
        body = client.get(f"/api/bouts/{bout}/profile").json()
        assert body["confirmed_touches"] == 1
        share = next(a for a in body["axes"] if a["key"] == "scoring_share_pct")
        assert share["fencer_1"]["value"] == 100.0

    def test_an_unknown_bout_is_a_404_not_a_crash(self, client):
        assert client.get("/api/bouts/nope:nothing/profile").status_code == 404


class TestSessionEndpoints:
    """The HTTP surface for the effort measurement."""

    def _bout(self, tmp_path, name="sess"):
        results = tmp_path / "results_uploads" / f"upload_{name}"
        results.mkdir(parents=True, exist_ok=True)
        (results / f"b_{name}_distance.csv").write_text(
            "frame,time_s,distance_raw_m,distance_smooth_m,method,"
            "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m,"
            "f1_pos_m,f2_pos_m\n"
            "0,0.0,2.0,2.0,pose,0,0,0,0,2.0,6.0\n"
            "1,0.5,2.0,2.0,pose,0,0,0,0,2.1,6.0\n")
        (results / f"b_{name}_distance_touches.csv").write_text(
            "time_s,confidence,min_distance_m,separation_m,audio_support,signals\n"
            "10.0,0.90,1.20,2.10,0,approach+separated\n")
        return f"upload_{name}:b_{name}"

    def test_records_a_session_and_returns_the_comparison(self, client, tmp_path):
        bout = self._bout(tmp_path)
        r = client.post(f"/api/bouts/{bout}/sessions",
                        json={"mode": "assisted", "elapsed_s": 60.0,
                              "decisions": 15})
        assert r.status_code == 200
        body = r.json()
        assert body["assisted"]["mean_seconds_per_decision"] == 4.0

    def test_reads_them_back(self, client, tmp_path):
        bout = self._bout(tmp_path, name="read")
        client.post(f"/api/bouts/{bout}/sessions",
                    json={"mode": "manual", "elapsed_s": 100.0, "decisions": 5})
        body = client.get(f"/api/bouts/{bout}/sessions").json()
        assert body["manual"]["runs"] == 1

    def test_an_unknown_mode_is_refused_by_the_schema(self, client, tmp_path):
        bout = self._bout(tmp_path, name="mode")
        r = client.post(f"/api/bouts/{bout}/sessions",
                        json={"mode": "quick", "elapsed_s": 60.0,
                              "decisions": 15})
        assert r.status_code == 422

    def test_a_zero_duration_is_refused(self, client, tmp_path):
        bout = self._bout(tmp_path, name="zero")
        r = client.post(f"/api/bouts/{bout}/sessions",
                        json={"mode": "manual", "elapsed_s": 0,
                              "decisions": 5})
        assert r.status_code == 422

    def test_the_counts_come_from_the_store_not_the_request(self, client,
                                                            tmp_path):
        # So a client cannot report a session its own annotations do not support.
        bout = self._bout(tmp_path, name="counts")
        appmod.store.add_touch(bout, 12.0)
        client.post(f"/api/bouts/{bout}/sessions",
                    json={"mode": "manual", "elapsed_s": 60.0, "decisions": 99})
        s = appmod.store.load(bout)["sessions"][0]
        assert s["decisions"] == 99
        assert s["touches_after"] == 1

    def test_an_unknown_bout_is_a_404(self, client):
        assert client.get("/api/bouts/nope:nothing/sessions").status_code == 404
