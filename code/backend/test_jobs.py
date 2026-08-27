"""
Tests for the processing job runner.

WHAT THESE CAN AND CANNOT SHOW. The stages here are stand-ins: small Python
commands that print progress lines, exit with a chosen code, or sleep until they
are killed. That is deliberate, because it makes the runner testable at all. A
test that ran the real pipeline would need real footage, would take minutes, and
would be testing YOLO rather than the queueing, cancellation and failure
handling that this file is about.

It also means these tests cannot show that the real commands are correct. That
distinction has bitten this project four separate times: a feature implemented
exactly as specified, passing a full suite, and measuring the wrong thing on real
footage. The commands `pipeline_stages` builds are therefore checked separately,
in `test_stage_commands`, and neither that nor this substitutes for one real
end-to-end run.

Run with:
    cd code/backend && python3 -m pytest test_jobs.py -v
"""

import json
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from jobs import (  # noqa: E402
    AWAITING_PISTE, CANCELLED, DONE, FAILED, INTERRUPTED, QUEUED, RUNNING,
    JobRunner, JobStore, Stage, parse_progress, pipeline_stages,
)


# -------------------- helpers --------------------

def py(code):
    """A stage command that runs a snippet of Python."""
    return [sys.executable, "-c", code]


def wait_for(predicate, timeout=10.0, interval=0.02):
    """
    Poll until a condition holds.

    The runner is a real thread driving real subprocesses, so tests have to wait
    for it rather than assert immediately. A fixed sleep would either be flaky or
    slow; this is neither.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def wait_for_state(store, job_id, *states, timeout=10.0):
    return wait_for(lambda: (store.load(job_id) or {}).get("state") in states,
                    timeout=timeout)


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path / "jobs"))


@pytest.fixture
def make_job(store, tmp_path):
    def _make(**fields):
        base = {
            "source_path": str(tmp_path / "bout.mp4"),
            "output_dir": str(tmp_path / "out"),
            "log_path": str(tmp_path / "job.log"),
            "piste_result_path": str(tmp_path / "piste_measurement.json"),
            "piste_config_path": str(tmp_path / "piste.json"),
            "web_video_path": str(tmp_path / "web.mp4"),
        }
        base.update(fields)
        return store.create(**base)
    return _make


# -------------------- progress parsing --------------------

class TestParseProgress:
    def test_reads_a_progress_line(self):
        assert parse_progress("PROGRESS 50 200\n") == {
            "done": 50, "total": 200, "pct": 25.0}

    def test_ignores_ordinary_output(self):
        assert parse_progress("  1200/5246 frames processed") is None
        assert parse_progress("Loading model: yolov8n.pt") is None

    def test_ignores_lines_that_merely_contain_numbers(self):
        # The reason the parser is strict. A looser version read the frame
        # counter out of ultralytics' own logging and reported a percentage of
        # an unrelated quantity.
        assert parse_progress("image 1/1 640x384 2 persons, 51.4ms") is None

    def test_rejects_a_zero_total(self):
        # Videos whose container reports no frame count would otherwise divide
        # by zero mid-run.
        assert parse_progress("PROGRESS 5 0") is None

    def test_caps_at_one_hundred_percent(self):
        # The frame count from a container is a claim, not a guarantee, and
        # OpenCV routinely reads a frame or two more than it promised.
        assert parse_progress("PROGRESS 210 200")["pct"] == 100.0


# -------------------- the store --------------------

class TestJobStore:
    def test_creates_and_loads(self, store):
        job = store.create(filename="a.mp4")
        assert store.load(job["job_id"])["filename"] == "a.mp4"

    def test_unknown_job_is_none(self, store):
        assert store.load("nope") is None

    def test_lists_newest_first(self, store):
        a = store.create(filename="a.mp4")
        time.sleep(0.01)
        b = store.create(filename="b.mp4")
        assert [j["job_id"] for j in store.list()] == [b["job_id"], a["job_id"]]

    def test_a_partial_write_is_never_visible(self, store):
        """
        Records are replaced atomically, so a poll arriving mid-save reads the
        previous record rather than half of the next one. Polling makes that a
        routine event: the client asks several times a second while the worker
        writes on every progress line.
        """
        job = store.create(filename="a.mp4")
        for i in range(50):
            store.update(job["job_id"], progress={"done": i, "total": 50})
            loaded = store.load(job["job_id"])
            assert loaded is not None and "progress" in loaded
        assert not [f for f in os.listdir(store.root) if f.endswith(".tmp")]


# -------------------- running --------------------

class TestRunning:
    def test_runs_stages_in_order_and_finishes(self, store, make_job, tmp_path):
        marker = tmp_path / "order.txt"
        stages = [
            Stage("one", lambda j: py(f"open({str(marker)!r}, 'a').write('one\\n')")),
            Stage("two", lambda j: py(f"open({str(marker)!r}, 'a').write('two\\n')")),
        ]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], DONE)
        assert marker.read_text().split() == ["one", "two"]
        assert store.load(job["job_id"])["stages_done"] == ["one", "two"]

    def test_reports_progress_as_it_goes(self, store, make_job):
        stages = [Stage("slow", lambda j: py(
            "import time\n"
            "for i in range(1, 6):\n"
            "    print(f'PROGRESS {i*20} 100', flush=True)\n"
            "    time.sleep(0.05)\n"))]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        # The point of the feature is that progress is visible DURING the run,
        # not recoverable after it, so this asserts on a mid-run reading.
        seen = wait_for(lambda: (store.load(job["job_id"])["progress"]["pct"]
                                 if 0 < store.load(job["job_id"])["progress"]["pct"] < 100
                                 else None), timeout=5.0)
        assert seen is not None
        assert wait_for_state(store, job["job_id"], DONE)

    def test_a_failing_stage_fails_the_job_and_keeps_the_output(self, store, make_job):
        stages = [
            Stage("boom", lambda j: py("import sys; print('it went wrong'); sys.exit(3)"),
                  label="detection"),
            Stage("never", lambda j: py("print('should not run')")),
        ]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], FAILED)
        final = store.load(job["job_id"])
        assert "detection" in final["error"] and "3" in final["error"]
        # Without the output a failure is unactionable: the whole diagnosis of a
        # pipeline failure is in the traceback it printed.
        assert any("it went wrong" in line for line in final["log_tail"])
        assert "never" not in final["stages_done"]

    def test_a_stage_may_skip_itself(self, store, make_job):
        stages = [
            Stage("optional", lambda j: None),
            Stage("real", lambda j: py("print('ran')")),
        ]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], DONE)
        assert store.load(job["job_id"])["stages_done"] == ["optional", "real"]

    def test_output_is_written_to_the_log_file(self, store, make_job, tmp_path):
        stages = [Stage("talk", lambda j: py("print('hello from the pipeline')"))]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], DONE)
        assert "hello from the pipeline" in open(job["log_path"]).read()

    def test_jobs_run_one_at_a_time(self, store, make_job, tmp_path):
        """
        Concurrency is 1 on purpose. Detection is CPU-bound and pose estimation
        is the bottleneck without a GPU, so two runs at once take more than
        twice as long as one and make the machine unusable meanwhile.
        """
        marker = tmp_path / "concurrent.txt"
        stages = [Stage("busy", lambda j: py(
            f"import time\n"
            f"p = {str(marker)!r}\n"
            f"open(p, 'a').write('start\\n')\n"
            f"time.sleep(0.3)\n"
            f"open(p, 'a').write('end\\n')\n"))]
        runner = JobRunner(store, stages)
        runner.start()
        ids = [make_job()["job_id"] for _ in range(3)]
        for job_id in ids:
            runner.submit(job_id)

        for job_id in ids:
            assert wait_for_state(store, job_id, DONE, timeout=15.0)
        # Interleaving would show up as two starts before the first end.
        assert marker.read_text().split() == ["start", "end"] * 3


# -------------------- cancellation --------------------

class TestCancellation:
    def test_cancels_a_running_job(self, store, make_job):
        stages = [Stage("forever", lambda j: py("import time; time.sleep(60)"))]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], RUNNING)
        assert runner.cancel(job["job_id"]) is True
        assert wait_for_state(store, job["job_id"], CANCELLED)

    def test_cancels_a_job_still_in_the_queue(self, store, make_job):
        stages = [Stage("slow", lambda j: py("import time; time.sleep(0.4)"))]
        runner = JobRunner(store, stages)
        runner.start()
        first, second = make_job()["job_id"], make_job()["job_id"]
        runner.submit(first)
        runner.submit(second)

        assert wait_for_state(store, first, RUNNING)
        assert runner.cancel(second) is True
        assert wait_for_state(store, second, CANCELLED)
        assert wait_for_state(store, first, DONE)
        # The cancelled job must never start, not merely stop early.
        assert store.load(second)["stages_done"] == []

    def test_cancelling_a_finished_job_does_nothing(self, store, make_job):
        stages = [Stage("quick", lambda j: py("pass"))]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], DONE)
        assert runner.cancel(job["job_id"]) is False
        assert store.load(job["job_id"])["state"] == DONE

    def test_a_killed_stage_is_cancelled_not_failed(self, store, make_job):
        """
        A killed process exits non-zero, which is indistinguishable from a crash
        unless the runner remembers it did the killing. Reporting a user's own
        cancellation as a pipeline failure would send them looking for a bug.
        """
        stages = [Stage("forever", lambda j: py("import time; time.sleep(60)"))]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], RUNNING)
        runner.cancel(job["job_id"])
        assert wait_for_state(store, job["job_id"], CANCELLED)
        assert store.load(job["job_id"])["error"] is None


# -------------------- pausing and resuming --------------------

class TestPauseAndResume:
    def test_a_hook_can_pause_the_job(self, store, make_job, tmp_path):
        marker = tmp_path / "after.txt"

        def pause(job, st):
            st.update(job["job_id"], state=AWAITING_PISTE)
            return False

        stages = [
            Stage("measure", lambda j: py("print('measured')"), on_success=pause),
            Stage("after", lambda j: py(f"open({str(marker)!r}, 'a').write('ran')")),
        ]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], AWAITING_PISTE)
        time.sleep(0.2)
        assert not marker.exists()

    def test_resuming_does_not_repeat_completed_stages(self, store, make_job, tmp_path):
        """
        The pause is what lets a user confirm the piste region mid-run. Repeating
        the measurement on resume would waste the time the pause was meant to
        save, and repeating detection would be worse than that.
        """
        counter = tmp_path / "count.txt"

        def pause_once(job, st):
            if job.get("paused_already"):
                return True
            st.update(job["job_id"], state=AWAITING_PISTE, paused_already=True)
            return False

        stages = [
            Stage("measure", lambda j: py(
                f"open({str(counter)!r}, 'a').write('x')"), on_success=pause_once),
            Stage("after", lambda j: py("print('done')")),
        ]
        runner = JobRunner(store, stages)
        runner.start()
        job = make_job()
        runner.submit(job["job_id"])

        assert wait_for_state(store, job["job_id"], AWAITING_PISTE)
        assert counter.read_text() == "x"

        store.update(job["job_id"], state=QUEUED)
        runner.submit(job["job_id"])
        assert wait_for_state(store, job["job_id"], DONE)
        assert counter.read_text() == "x", "the measurement stage ran twice"


# -------------------- restart --------------------

class TestReconcile:
    def test_a_job_left_running_is_marked_interrupted(self, store, make_job):
        """
        A record saying `running` after a restart is a claim with no process
        behind it. Nothing in this system writes back to a job on shutdown,
        because a hard kill would skip any handler that tried.
        """
        job = store.create(state=RUNNING, source_path="x", output_dir="y",
                           log_path="z")
        runner = JobRunner(store, [])
        assert runner.reconcile() == [job["job_id"]]
        final = store.load(job["job_id"])
        assert final["state"] == INTERRUPTED
        assert final["error"]

    def test_an_interrupted_job_is_not_retried_automatically(self, store):
        """
        Retrying on startup would mean a video that crashes the pipeline crashes
        it again on every boot, forever.
        """
        job = store.create(state=RUNNING, source_path="x", output_dir="y")
        runner = JobRunner(store, [Stage("s", lambda j: py("pass"))])
        runner.reconcile()
        runner.start()
        time.sleep(0.3)
        assert store.load(job["job_id"])["state"] == INTERRUPTED

    def test_a_queued_job_is_put_back_on_the_queue(self, store, make_job):
        # Unlike a running job, a queued one lost only its place in memory.
        job = make_job(state=QUEUED)
        runner = JobRunner(store, [Stage("s", lambda j: py("print('ran')"))])
        runner.reconcile()
        runner.start()
        assert wait_for_state(store, job["job_id"], DONE)


# -------------------- the real commands --------------------

class TestStageCommands:
    """
    The stand-in stages above cannot show that the real commands are right, so
    these check what `pipeline_stages` actually builds. Still not a substitute
    for running it: a well-formed command can still be the wrong command.
    """

    def job(self, tmp_path, **extra):
        base = {
            "job_id": "abc123",
            "source_path": str(tmp_path / "bout_abc123.mp4"),
            "output_dir": str(tmp_path / "upload_abc123"),
            "piste_result_path": str(tmp_path / "measure.json"),
            "piste_config_path": str(tmp_path / "piste.json"),
            "web_video_path": str(tmp_path / "web.mp4"),
            "piste": None,
            "confirm_piste": True,
        }
        base.update(extra)
        return base

    def stage(self, name):
        return next(s for s in pipeline_stages() if s.name == name)

    def test_detection_is_told_the_source_and_the_output(self, tmp_path):
        cmd = self.stage("detect").build(self.job(tmp_path))
        assert "run_detection.py" in cmd
        assert "--video" in cmd and "--output" in cmd
        assert cmd[cmd.index("--output") + 1] == str(tmp_path / "upload_abc123")

    def test_detection_asks_for_machine_readable_progress(self, tmp_path):
        # Without it the job runner has nothing to report, and a user watching a
        # multi-minute run is shown a spinner instead of a position.
        assert "--progress" in self.stage("detect").build(self.job(tmp_path))

    def test_the_piste_config_is_passed_when_there_is_one(self, tmp_path):
        """
        The flag that is easy to forget and expensive to omit. Rebuilding the
        reference results without it dropped the broadcast clip from 98.0 per
        cent coverage to 80.1, which looked exactly like a code regression.
        """
        config = tmp_path / "piste.json"
        config.write_text('{"polygon": [[0,0],[10,0],[10,10],[0,10]]}')
        job = self.job(tmp_path, piste={"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]})
        cmd = self.stage("detect").build(job)
        assert cmd[cmd.index("--piste-config") + 1] == str(config)

    def test_no_piste_flag_when_the_region_was_skipped(self, tmp_path):
        job = self.job(tmp_path, piste={"needed": False, "polygon": None})
        assert "--piste-config" not in self.stage("detect").build(job)

    def test_measurement_is_skipped_when_a_region_is_already_known(self, tmp_path):
        job = self.job(tmp_path, piste={"needed": True, "polygon": [[0, 0]]})
        assert self.stage("piste").build(job) is None

    def test_touch_detection_reads_the_csv_detection_writes(self, tmp_path):
        """
        The two stages are joined by a filename convention rather than by a
        return value, so a change to either end breaks the join silently.
        """
        cmd = self.stage("touches").build(self.job(tmp_path))
        csv_path = cmd[cmd.index("--csv") + 1]
        assert csv_path == str(tmp_path / "upload_abc123" / "bout_abc123_distance.csv")

    def test_no_stage_generates_a_summary(self, tmp_path):
        """
        Summary generation costs a paid API call. Keeping it out of the
        automatic path is a decision the existing API made deliberately, and one
        an added stage could reverse by accident.
        """
        for stage in pipeline_stages():
            cmd = stage.build(self.job(tmp_path))
            assert cmd is None or "generate_summary.py" not in cmd

    def test_transcode_is_skipped_when_there_is_nothing_to_convert(self, tmp_path):
        assert self.stage("transcode").build(self.job(tmp_path)) is None
