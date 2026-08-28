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
