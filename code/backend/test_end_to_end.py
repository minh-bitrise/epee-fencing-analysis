"""
End-to-end: a video goes in, a reviewable bout comes out.

WHY THIS EXISTS SEPARATELY. `test_jobs.py` tests the runner with stand-in stages,
`test_upload_api.py` tests the endpoints with a stubbed runner, and neither can
catch the thing most likely to break: the joins between them. The stages are
connected by filename conventions rather than by return values, the review
interface finds bouts by scanning directories for a naming pattern, and the bout
id is assembled from a directory name in one module and parsed apart in another.
Every one of those is invisible to a unit test and fatal in use.

This runs the REAL pipeline on a tiny synthetic video: real detection, real touch
proposal, real discovery, real review endpoints. It is slow by the standards of
the rest of the suite, a minute or so, because it loads the models.

WHAT THE SYNTHETIC VIDEO CAN AND CANNOT SHOW. It contains two moving rectangles,
not two fencers, so nothing here says anything about whether detection or touch
proposal WORKS. Those questions are answered by measurement against hand-labelled
footage. What this asserts is narrower and not covered anywhere else: that the
stages hand their outputs to each other correctly, and that what comes out the far
end is reachable through the interface a user actually uses.

Marked slow, so the fast suite stays fast:
    python3 -m pytest test_end_to_end.py -v
    python3 -m pytest -m "not slow"        # everything else
"""

import os
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))

import app as appmod  # noqa: E402
from jobs import JobRunner, JobStore, pipeline_stages  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def synthetic_bout(tmp_path):
    """A short video with two person-sized shapes moving toward each other.

    Person-SHAPED on purpose: roughly 2:1 tall, moving smoothly, on a plain
    background.
    """
    import cv2
    path = tmp_path / "synthetic.mp4"
    w, h, n = 320, 240, 90
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        left = 40 + int(60 * i / n)
        right = 240 - int(60 * i / n)
        for x in (left, right):
            cv2.rectangle(frame, (x - 15, 90), (x + 15, 180), (60, 60, 60), -1)
            cv2.circle(frame, (x, 80), 12, (60, 60, 60), -1)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture
def live_client(tmp_path, monkeypatch):
    """The real app with a real runner, writing into a temporary directory."""
    for name, sub in (("UPLOAD_ROOT", "uploads"), ("JOB_ROOT", "jobs"),
                      ("UPLOAD_RESULTS_ROOT", "results_uploads"),
                      ("WEB_VIDEO_DIR", "web_video")):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(appmod, name, str(d))

    store = JobStore(str(tmp_path / "jobs"))
    runner = JobRunner(store, appmod._stages_for, cwd=appmod.PROTOTYPE_DIR)
    monkeypatch.setattr(appmod, "job_store", store)
    monkeypatch.setattr(appmod, "runner", runner)

    from store import AnnotationStore
    monkeypatch.setattr(appmod, "store",
                        AnnotationStore(str(tmp_path / "annotations")))

    with TestClient(appmod.app) as c:
        c.store = store
        yield c


def wait_for(client, job_id, states, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(1.0)
    return client.get(f"/api/jobs/{job_id}").json()


def test_a_video_becomes_a_reviewable_bout(live_client, synthetic_bout):
    """The whole loop, in the order a user meets it.

    Every assertion here is about a JOIN. The stages pass work by filename
    convention, the bout id is built in one module and taken apart in another,
    and discovery finds bouts by scanning for a naming pattern.
    """
    with open(synthetic_bout, "rb") as f:
        r = live_client.post("/api/jobs",
                             files={"video": ("synthetic.mp4", f, "video/mp4")},
                             data={"confirm_piste": "false"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    job = wait_for(live_client, job_id,
                   ("done", "failed", "cancelled", "interrupted"))
    assert job["state"] == "done", (
        f"pipeline failed: {job.get('error')}\n"
        + "\n".join(job.get("log_tail", [])[-15:]))

    # every stage ran, in order, and none was silently skipped
    assert job["stages_done"][:3] == ["piste", "detect", "touches"]

    # the detect stage's progress was parsed from its output, which is the only
    # thing standing between the user and an inert spinner on a long run
    assert job["progress"]["pct"] > 0

    bout_id = job["bout_id"]
    assert bout_id, "a finished job must name a bout the review interface can open"

    # the bout is DISCOVERABLE: the id assembled from the job must match what
    # discover_bouts derives independently from the directory
    listed = live_client.get("/api/bouts").json()["bouts"]
    assert any(b["bout_id"] == bout_id for b in listed), (
        f"{bout_id} was not found by discovery among "
        f"{[b['bout_id'] for b in listed][:5]}")

    # and it is named after the file rather than after a directory
    mine = next(b for b in listed if b["bout_id"] == bout_id)
    assert mine["label"] == "synthetic.mp4"

    # the review endpoints work against it
    touches = live_client.get(f"/api/bouts/{bout_id}/touches")
    assert touches.status_code == 200, touches.text
    assert touches.json()["duration_s"] > 0

    # Metrics either work or explain themselves. Two rectangles are not
    # necessarily people to YOLO, so this fixture cannot guarantee measurements
    # exist, and a bout with none is a real outcome rather than a broken one: an
    # upload shot from behind the piste produces exactly it.
    metrics = live_client.get(f"/api/bouts/{bout_id}/metrics")
    assert metrics.status_code in (200, 422), metrics.text
    if metrics.status_code == 200:
        body = metrics.json()
        assert "in_play" in body
        # nothing has been reviewed, so scoping must say so rather than implying
        # the figures were scoped to confirmed play
        assert "none" in body["scoping_basis"].lower()
    else:
        # the refusal has to be actionable, not just a status code
        detail = metrics.json()["detail"]
        assert "no usable measurements" in detail
        assert "framing" in detail


def test_a_corrupt_video_fails_the_job_rather_than_the_server(live_client,
                                                              tmp_path):
    """
    A file that opens but cannot be processed must produce a failed job carrying
    its error, not a dead worker. The whole reason stages run as subprocesses is
    that the native libraries can die in ways a thread cannot survive.
    """
    import cv2
    path = tmp_path / "oneframe.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (32, 32))
    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    with open(path, "rb") as f:
        r = live_client.post("/api/jobs",
                             files={"video": ("oneframe.mp4", f, "video/mp4")},
                             data={"confirm_piste": "false"})
    if r.status_code != 200:
        pytest.skip("this build rejects the file at upload, which is also fine")

    job_id = r.json()["job_id"]
    job = wait_for(live_client, job_id,
                   ("done", "failed", "cancelled", "interrupted"))
    assert job["state"] in ("done", "failed")
    if job["state"] == "failed":
        assert job["error"], "a failed job must say why"
        assert job.get("log_tail"), "a failure without its output is unactionable"

    # the server is still answering, which is the point
    assert live_client.get("/api/jobs").status_code == 200
