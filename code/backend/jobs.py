"""
Epee Fencing Bout Analysis - Processing Job Runner
===================================================
Runs the command-line pipeline on an uploaded video, in the background, so a
bout can be processed from a browser instead of a terminal.

WHY THIS EXISTS. Until now the pipeline was reachable only from a shell: there
was no upload, no way to start a run, and no way for anyone but the author to
use the system at all. The design chapter describes a web application and the
evaluation chapter admits the gap. This is the missing layer.

WHY SUBPROCESSES AND NOT THREADS. Each stage runs as a separate process rather
than as a function call in the API's own process, for three reasons that are
specific to this pipeline rather than general good practice.

  1. The heavy work is native code. YOLO, OpenCV and MediaPipe do their work in
     C++ extensions, and when those fail they can take the interpreter with them
     rather than raising something catchable. In a thread that ends the API and
     every job with it. In a subprocess it is an exit code the runner can
     record against one job.
  2. The models cost hundreds of megabytes of resident memory. A subprocess
     gives it all back on exit; an in-process import keeps it for the life of
     the server, which on this project is a laptop also running a browser.
  3. The pipeline stays a working command line. Every stage is invoked here by
     exactly the command a user would type, which the evaluation depends on:
     the figures in the report came from those commands, and a second in-process
     code path would be a second thing that could disagree with them.

WHY ONE JOB AT A TIME. Detection is CPU-bound and pose estimation is the
bottleneck on a machine without a GPU. Two concurrent runs do not finish in the
time one takes, they finish in rather more than twice it, while making the
machine unusable. Jobs therefore queue, and the queue position is reported so a
waiting user is told they are waiting rather than left to guess.

WHERE PROGRESS COMES FROM. Each stage prints machine-readable `PROGRESS done
total` lines, which this parses as it reads the process output. The alternative
was a callback into the pipeline, which would have meant importing it, which is
the thing this file exists to avoid.

STATE. Every job is a JSON file on disk, in the same spirit as the annotation
store: no database, nothing lost on restart, and a record a human can read when
something goes wrong. A run interrupted by a restart is marked as interrupted on
the next startup rather than being left claiming to be running.
"""

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

# Job states.
QUEUED = "queued"                  # waiting for the worker
RUNNING = "running"                # a stage is executing
AWAITING_PISTE = "awaiting_piste"  # paused for the user to confirm the piste region
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"        # the server stopped while this was running

TERMINAL_STATES = (DONE, FAILED, CANCELLED, INTERRUPTED)

# Uploads are capped rather than unbounded. The evaluation clips are 10 to 50 MB;
# this allows a long bout at broadcast quality while still refusing a file that
# was obviously not meant for this.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")

# How many lines of a failed stage's output to keep on the job record. Enough to
# carry a Python traceback, which is what a failure here almost always is.
ERROR_TAIL_LINES = 40


class JobStore:
    """
    Job records as one JSON file each.

    Reads go straight to disk rather than through a cache. Jobs are polled a few
    times a second by one browser, the files are under a kilobyte, and a cache
    would mean the worker thread and the request threads holding two views of
    the same job.
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id):
        return os.path.join(self.root, f"{job_id}.json")

    def create(self, **fields):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "state": QUEUED,
            "stage": None,
            "stages_done": [],
            "progress": {"done": 0, "total": 0, "pct": 0.0},
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "log_tail": [],
            "bout_id": None,
            "piste": None,
        }
        job.update(fields)
        self.save(job)
        return job

    def save(self, job):
        # Written to a temporary file and moved into place, so a reader that
        # arrives mid-write sees the previous record rather than half of this
        # one. Polling makes that a routine event rather than a rare one.
        with self._lock:
            path = self._path(job["job_id"])
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(job, f, indent=2)
            os.replace(tmp, path)
        return job

    def load(self, job_id):
        path = self._path(job_id)
        if not os.path.exists(path):
            return None
        with self._lock:
            try:
                with open(path) as f:
                    return json.load(f)
            except ValueError:
                return None

    def update(self, job_id, **fields):
        job = self.load(job_id)
        if job is None:
            return None
        job.update(fields)
        return self.save(job)

    def list(self):
        out = []
        for name in os.listdir(self.root):
            if name.endswith(".json") and not name.endswith(".tmp"):
                job = self.load(name[:-5])
                if job:
                    out.append(job)
        return sorted(out, key=lambda j: j["created_at"], reverse=True)

    def delete(self, job_id):
        path = self._path(job_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


class Stage:
    """
    One pipeline command.

    `build` returns the argument list to run, or None to skip the stage for this
    job. Stages are supplied to the runner rather than hardcoded in it so the
    tests can substitute cheap stand-ins: a test that had to run YOLO would take
    minutes and need real footage, and would be testing the models rather than
    the runner.
    """

    def __init__(self, name, build, label=None, on_success=None):
        self.name = name
        self.build = build
        self.label = label or name
        self.on_success = on_success


def parse_progress(line):
    """
    Read a `PROGRESS done total` line, or return None.

    Deliberately strict about the shape. An earlier version matched anything
    containing two integers, which happily read the frame counter out of
    ultralytics' own logging and reported a percentage of the wrong quantity.
    """
    parts = line.strip().split()
    if len(parts) != 3 or parts[0] != "PROGRESS":
        return None
    try:
        done, total = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if total <= 0:
        return None
    return {"done": done, "total": total,
            "pct": round(100.0 * min(done, total) / total, 1)}


class JobRunner:
    """
    A single background worker that takes jobs off a queue and runs their stages.

    The worker thread is a daemon: it holds no state that is not already on
    disk, so there is nothing to flush at shutdown, and a job caught mid-run is
    recovered by `reconcile` on the next startup rather than by an exit handler
    that a hard kill would skip anyway.
    """

    def __init__(self, store, stages, cwd=None, env=None):
        self.store = store
        self.stages = stages
        self.cwd = cwd
        self.env = env
        self._queue = queue.Queue()
        self._thread = None
        self._current = None          # job_id being processed
        self._proc = None             # the running subprocess
        self._cancelled = set()
        self._lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._work, daemon=True,
                                        name="job-runner")
        self._thread.start()

    def reconcile(self):
        """
        Repair the record after a restart.

        A job whose process died with the server is left on disk saying it is
        running, which is a claim nothing is behind. Marking those interrupted
        keeps the record honest; re-queueing them automatically would be worse,
        since a server that crashes on a particular video would then retry it
        forever.
        """
        repaired = []
        for job in self.store.list():
            if job["state"] in (RUNNING,):
                self.store.update(job["job_id"], state=INTERRUPTED,
                                  finished_at=time.time(),
                                  error="the server stopped while this job was "
                                        "running; start it again to retry")
                repaired.append(job["job_id"])
            elif job["state"] == QUEUED:
                # A queued job lost only its place in an in-memory queue, so
                # unlike a running one it can simply be put back.
                self._queue.put(job["job_id"])
        return repaired

    def submit(self, job_id):
        self._queue.put(job_id)

    def cancel(self, job_id):
        """
        Stop a job, whether it is queued or already running.

        A queued job is marked and skipped when the worker reaches it, because
        removing an item from the middle of a Queue is not something Queue
        supports. A running one has its process killed.
        """
        job = self.store.load(job_id)
        if job is None or job["state"] in TERMINAL_STATES:
            return False
        with self._lock:
            self._cancelled.add(job_id)
            proc = self._proc if self._current == job_id else None
        if proc is not None:
            self._kill(proc)
        else:
            self.store.update(job_id, state=CANCELLED, finished_at=time.time())
        return True

    def queue_position(self, job_id):
        """
        How many jobs are ahead of this one.

        Derived from the store rather than from the Queue, because the Queue
        cannot be inspected without consuming it, and a waiting user is owed a
        number rather than a spinner.
        """
        job = self.store.load(job_id)
        if job is None or job["state"] != QUEUED:
            return None
        ahead = [j for j in self.store.list()
                 if j["state"] in (QUEUED, RUNNING)
                 and j["created_at"] < job["created_at"]]
        return len(ahead)

    @staticmethod
    def _kill(proc):
        # The whole process group goes, not just the child. ffmpeg and
        # ultralytics both spawn their own workers, and killing only the parent
        # leaves those running with nothing reading their output.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    # --- the worker -----------------------------------------------------

    def _work(self):
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception as e:  # noqa: BLE001 - a worker must not die
                self.store.update(job_id, state=FAILED, finished_at=time.time(),
                                  error=f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._current = None
                    self._proc = None
                self._queue.task_done()

    def _run_job(self, job_id):
        job = self.store.load(job_id)
        if job is None:
            return
        if job["state"] in TERMINAL_STATES:
            return
        with self._lock:
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                self.store.update(job_id, state=CANCELLED,
                                  finished_at=time.time())
                return
            self._current = job_id

        self.store.update(job_id, state=RUNNING,
                          started_at=job.get("started_at") or time.time(),
                          error=None)

        for stage in self.stages:
            job = self.store.load(job_id)
            if job is None or job["state"] in TERMINAL_STATES:
                return
            # Stages already completed are skipped rather than repeated. This is
            # what lets a job pause for the user to confirm the piste region and
            # then carry on from where it stopped instead of starting over.
            if stage.name in job["stages_done"]:
                continue

            cmd = stage.build(job)
            if cmd is None:
                job["stages_done"].append(stage.name)
                self.store.save(job)
                continue

            self.store.update(job_id, stage=stage.name,
                              progress={"done": 0, "total": 0, "pct": 0.0})
            code, tail = self._run_stage(job_id, cmd)

            job = self.store.load(job_id)
            if job is None:
                return
            with self._lock:
                cancelled = job_id in self._cancelled
                if cancelled:
                    self._cancelled.discard(job_id)
            if cancelled:
                self.store.update(job_id, state=CANCELLED, stage=None,
                                  finished_at=time.time(),
                                  log_tail=tail)
                return
            if code != 0:
                self.store.update(
                    job_id, state=FAILED, stage=stage.name,
                    finished_at=time.time(), log_tail=tail,
                    error=f"the {stage.label} stage exited with code {code}")
                return

            job["stages_done"].append(stage.name)
            job["log_tail"] = tail
            self.store.save(job)

            if stage.on_success is not None:
                # A hook may end the job early. Confirming the piste region does
                # exactly that: the run pauses, the user answers, and the job is
                # resubmitted rather than being held open on a blocked thread.
                job = self.store.load(job_id)
                if stage.on_success(job, self.store) is False:
                    return

        self.store.update(job_id, state=DONE, stage=None,
                          finished_at=time.time(),
                          progress={"done": 1, "total": 1, "pct": 100.0})

    def _run_stage(self, job_id, cmd):
        """
        Run one command, streaming its output to the job's log file and reading
        progress out of it as it goes.
        """
        job = self.store.load(job_id)
        log_path = job.get("log_path")
        tail = []

        proc = subprocess.Popen(
            cmd, cwd=self.cwd, env=self.env,
            stdout=subprocess.PIPE,
            # Merged rather than kept apart: ultralytics writes to both, and a
            # failure is far easier to read when the traceback still sits in
            # sequence with the output that preceded it.
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            # Its own process group, so cancelling can take the children too.
            start_new_session=True,
        )
        with self._lock:
            self._proc = proc

        log = open(log_path, "a") if log_path else None
        try:
            if log:
                log.write(f"\n$ {' '.join(cmd)}\n")
                log.flush()
            for line in proc.stdout:
                if log:
                    log.write(line)
                    log.flush()
                tail.append(line.rstrip("\n"))
                if len(tail) > ERROR_TAIL_LINES:
                    tail.pop(0)
                prog = parse_progress(line)
                if prog is not None:
                    self.store.update(job_id, progress=prog)
        finally:
            proc.stdout.close()
            code = proc.wait()
            if log:
                log.close()
            with self._lock:
                self._proc = None
        return code, tail


# --- the real pipeline stages ------------------------------------------

def pipeline_stages(python_exe=None, prototype_dir=None):
    """
    The four stages an uploaded bout goes through.

    Detection and touch proposal run automatically. Summary generation does not,
    and its absence here is deliberate rather than an omission: it costs a paid
    API call per run, and the existing API kept it a considered command-line step
    for that reason. A job that quietly spent money on every upload would undo
    that decision by accident.
    """
    python_exe = python_exe or sys.executable

    def stem(job):
        return os.path.splitext(os.path.basename(job["source_path"]))[0]

    def build_piste(job):
        # Skipped when the user has already supplied or confirmed a region, so a
        # resumed job does not re-measure what it just measured.
        if job.get("piste") is not None:
            return None
        return [python_exe, "derive_piste.py",
                "--video", job["source_path"],
                "--progress",
                "--output-json", job["piste_result_path"]]

    def after_piste(job, store):
        """
        Read the measurement, and decide whether to pause for the user.

        Returning False stops the job here. It pauses only when there is
        something worth showing: a video with no bystanders needs no region at
        all, and stopping to ask about one would be an interruption with no
        question attached.
        """
        path = job["piste_result_path"]
        if not os.path.exists(path):
            store.update(job["job_id"], piste={"needed": False, "polygon": None,
                                               "confident": False,
                                               "reason": "measurement produced no result"})
            return True
        with open(path) as f:
            result = json.load(f)
        job["piste"] = result
        store.save(job)

        if result.get("needed") and result.get("polygon"):
            write_piste_config(result, job["piste_config_path"])
            if job.get("confirm_piste", True):
                store.update(job["job_id"], state=AWAITING_PISTE, stage=None)
                return False
        return True

    def build_detect(job):
        cmd = [python_exe, "run_detection.py",
               "--video", job["source_path"],
               "--output", job["output_dir"],
               "--progress"]
        piste = job.get("piste") or {}
        if piste.get("polygon") and os.path.exists(job["piste_config_path"]):
            cmd += ["--piste-config", job["piste_config_path"]]
        if job.get("pose_stride"):
            cmd += ["--pose-stride", str(job["pose_stride"])]
        return cmd

    def build_touches(job):
        csv_path = os.path.join(job["output_dir"], f"{stem(job)}_distance.csv")
        return [python_exe, "detect_touches.py", "--csv", csv_path]

    def build_transcode(job):
        """
        Convert the annotated render to something a browser will play.

        Done here rather than on first view. OpenCV writes MPEG-4 Part 2, which
        browsers generally refuse, so the existing API transcodes on demand
        inside a request handler. That was tolerable for four fixed clips and is
        not for uploads: it puts a multi-second ffmpeg run in the request path of
        whoever happens to open the bout first, which is the shape of thing this
        whole layer exists to remove.
        """
        src = os.path.join(job["output_dir"], f"{stem(job)}_annotated.mp4")
        if not os.path.exists(src):
            return None
        if shutil.which("ffmpeg") is None:
            return None
        return ["ffmpeg", "-y", "-i", src,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-movflags", "+faststart", "-an", job["web_video_path"]]

    return [
        Stage("piste", build_piste, label="piste measurement",
              on_success=after_piste),
        Stage("detect", build_detect, label="detection and tracking"),
        Stage("touches", build_touches, label="touch proposal"),
        Stage("transcode", build_transcode, label="video conversion"),
    ]


def write_piste_config(result, path):
    """
    Write a derived polygon where run_detection.py can load it.

    The reason the measurement gave is carried into the file as its comment. The
    hand-authored polygons record their reasoning the same way, and a derived
    file that did not would be indistinguishable from one placed by eye, which is
    the distinction this project has already paid to learn.
    """
    with open(path, "w") as f:
        json.dump({"_comment": result.get("reason", ""),
                   "polygon": result["polygon"]}, f, indent=2)
    return path
