"""
Epee Fencing Bout Analysis - Annotation API
============================================
FastAPI backend for the assisted-annotation workflow.

WHAT THIS EXISTS TO TEST. The project's central design claim is that unreliable
computer vision is acceptable because a user can repair each class of error with
a single high-level action. Until now that claim has been argued but never
exercised, which the evaluation chapter identifies as the most significant gap in
the project: the failures are demonstrated, the mechanism that resolves them is
not. This API exposes exactly the four actions the design specifies, so the claim
can be tested rather than asserted.

WHAT IT DELIBERATELY DOES NOT DO. It does not process video. Detection, tracking
and touch proposal remain command-line stages, and this reads their artefacts.
Keeping inference out of the request path means a long processing job cannot
block a review session, which is the separation the architecture calls for, and
it also means the interface can be evaluated against already-processed footage
without waiting on a pipeline run.

Run with:
    cd code/backend && python3 -m uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import glob
import os
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# The pipeline modules live alongside the prototype, and are imported rather
# than reimplemented so that the metrics the interface shows are computed by
# exactly the same code as the metrics in the evaluation.
PROTOTYPE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prototype"))
sys.path.insert(0, PROTOTYPE_DIR)

from store import (  # noqa: E402
    AnnotationStore, SCORERS, VALID_STATES,
    discover_bouts, load_proposed_touches,
)
from jobs import (  # noqa: E402
    ALLOWED_EXTENSIONS, AWAITING_PISTE, MAX_UPLOAD_BYTES, TERMINAL_STATES,
    JobRunner, JobStore, pipeline_stages, summary_stages,
    write_piste_config,
)

# results_fixed is listed first because it is the only output produced since the
# raw position columns were added, and those columns are what the reliable
# movement metrics are derived from. Without it the interface can only reach CSVs
# that force the fallback derivation, which inherits the noise floor, the
# movement cap and the banking buffer, and which disagreed with the position
# route by 3.5 m of net displacement on the club clip. The older directories stay
# discoverable so the before/after comparisons in the evaluation remain openable,
# and the interface reports which route each bout used rather than hiding it.
# results_current is first because it is the reference set: all four clips from one
# version of the pipeline, each with the piste configuration it needs, and the only
# set where movement figures are comparable across clips. RESULTS.md says which
# numbers to quote from where.
RESULTS_DIRS = [os.path.join(PROTOTYPE_DIR, d) for d in
                ("results_current", "results_pose", "results_fixed", "results_after",
                 "results_stabilised", "results_fixedscale", "results_ablation",
                 "results")]
ANNOTATION_ROOT = os.path.join(PROTOTYPE_DIR, "annotations")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Everything produced at runtime lives under one root, separate from both the
# code and the evaluation artefacts. Bouts uploaded through the browser are kept
# apart from the four clips the report's figures come from, so that "the results
# directory" continues to mean the evaluation set and a user's upload cannot be
# mistaken for one.
VAR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "var"))
UPLOAD_ROOT = os.path.join(VAR_ROOT, "uploads")
JOB_ROOT = os.path.join(VAR_ROOT, "jobs")
UPLOAD_RESULTS_ROOT = os.path.join(VAR_ROOT, "results_uploads")
for _d in (UPLOAD_ROOT, JOB_ROOT, UPLOAD_RESULTS_ROOT):
    os.makedirs(_d, exist_ok=True)

store = AnnotationStore(ANNOTATION_ROOT)
job_store = JobStore(JOB_ROOT)

_PROCESSING_STAGES = pipeline_stages()
_SUMMARY_STAGES = summary_stages()


def _stages_for(job):
    """
    Which pipeline a job runs.

    Two kinds share one worker rather than one runner each, because a second
    runner would mean a second worker thread and the one-job-at-a-time guarantee
    exists precisely so two CPU-bound runs do not fight over a machine with no
    GPU. A summary job is cheap, but it is not free and it is not worth a
    special case that could let it start while detection is mid-run.
    """
    return _SUMMARY_STAGES if job.get("kind") == "summary" else _PROCESSING_STAGES


runner = JobRunner(job_store, _stages_for, cwd=PROTOTYPE_DIR)


# The service name the key is stored under in the macOS Keychain.
KEYCHAIN_SERVICE = "anthropic-api-key"


def _load_api_key_from_keychain():
    """
    Put the provider API key into the environment at startup, reading it from the
    macOS Keychain when it is not already there.

    WHY AT STARTUP AND NOT ON DEMAND. Starting the server is a deliberate act by
    the person who owns the key, and macOS can prompt them for Keychain access
    while they are still at the keyboard. Reading a secret in response to an HTTP
    request would move that prompt to a moment nobody is watching, and would make
    a web request the thing that reaches into the Keychain. Once here, the value
    lives in this process's environment and is inherited by the summary
    subprocess, which is where it is actually needed.

    The key is never logged, never returned by any endpoint, and never written to
    a job record. Only whether one was found is reported.

    Failing is not an error. A machine without the key, or without `security`,
    simply has no summary button, and the endpoint says so.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "environment"
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-a", os.environ.get("USER", ""), "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    key = result.stdout.strip()
    if result.returncode != 0 or not key:
        return None
    os.environ["ANTHROPIC_API_KEY"] = key
    return "keychain"


@asynccontextmanager
async def lifespan(_app):
    # Jobs left mid-run by a previous process are reconciled before the worker
    # starts, so a restart cannot leave a record claiming to be running with
    # nothing behind it.
    source = _load_api_key_from_keychain()
    print(f"[startup] Provider API key: "
          + (f"loaded from the {source}, summary generation is available"
             if source else
             f"not found in the environment or in the Keychain under "
             f"'{KEYCHAIN_SERVICE}'; everything except summary generation works"))
    runner.reconcile()
    runner.start()
    yield
    # Nothing to tear down. The worker is a daemon thread holding no state that
    # is not already on disk, and a job caught mid-run is recovered by the
    # reconcile above rather than by a shutdown handler that a hard kill would
    # skip anyway.


app = FastAPI(title="Epee Bout Analysis", version="0.2.0", lifespan=lifespan)


# Discovery reads every results directory and stats every file in them, and it
# runs on every request that names a bout. That is a dozen directories now and
# grows by one per upload, so the scan is cached against the directories'
# modification times: a new bout changes the mtime of the directory holding it,
# which is exactly when the cache should be discarded and never otherwise.
_bouts_cache = {"key": None, "value": None}


def _bouts():
    # Uploaded bouts are discovered by scanning rather than from a fixed list,
    # because unlike the evaluation set their number is not known in advance.
    upload_dirs = sorted(glob.glob(os.path.join(UPLOAD_RESULTS_ROOT, "upload_*")))
    dirs = RESULTS_DIRS + upload_dirs

    key = tuple((d, os.path.getmtime(d) if os.path.isdir(d) else None)
                for d in dirs)
    if _bouts_cache["key"] == key:
        return _bouts_cache["value"]

    found = discover_bouts(dirs)
    _bouts_cache["key"] = key
    _bouts_cache["value"] = found
    return found


def _get_bout(bout_id):
    b = _bouts().get(bout_id)
    if b is None:
        raise HTTPException(404, f"unknown bout: {bout_id}")
    return b


# --- request bodies -----------------------------------------------------

class TouchDecision(BaseModel):
    state: str = Field(..., description=f"one of {VALID_STATES}")
    scorer: str | None = Field(None, description=f"one of {SCORERS}")
    time_s: float | None = Field(None, description="corrected timestamp")


class NewTouch(BaseModel):
    time_s: float
    scorer: str = "unknown"
    note: str = ""


class Segment(BaseModel):
    start_s: float
    end_s: float
    reason: str = ""


class Reanchor(BaseModel):
    time_s: float
    slot: int = Field(..., ge=0, le=1)
    x: float
    y: float


# --- read endpoints -----------------------------------------------------

def _bout_label(bout_id):
    """
    A name a person can read.

    Bout ids are derived from output directories, which is right for an
    identifier and wrong for a menu: an uploaded bout is called
    `upload_a07ab97cf679:bout_a07ab97cf679`, which says nothing about the video
    it came from. The evaluation clips keep their ids, since those ARE the names
    the report and RESULTS.md use and renaming them in the interface would break
    the correspondence.
    """
    directory = bout_id.split(":")[0]
    if not directory.startswith("upload_"):
        return bout_id
    job = job_store.load(directory[len("upload_"):])
    if not job:
        return bout_id
    name = job.get("filename") or bout_id
    if job.get("reprocess_of"):
        # Say what it is a rerun OF, because the whole point of writing a
        # reprocess to a new bout is comparing it against the original.
        return f"{name} (corrected)"
    return name


@app.get("/api/bouts")
def list_bouts():
    """Processed bouts available for review, with review progress for each."""
    out = []
    for bout_id, b in sorted(_bouts().items()):
        proposed = load_proposed_touches(b.touches_csv)
        out.append({
            "bout_id": bout_id,
            "label": _bout_label(bout_id),
            "has_touches": bool(b.touches_csv),
            "has_summary": bool(b.summary_md),
            "has_video": bool(b.video),
            "progress": store.review_progress(bout_id, proposed),
        })
    return {"bouts": out}


@app.get("/api/bouts/{bout_id}/touches")
def get_touches(bout_id: str):
    """
    Proposed touches with the user's decision on each, plus manually added ones.

    Confidence and the signals that fired are returned alongside every proposal,
    because the user is being asked to adjudicate the system's guess and needs to
    see what that guess rests on.
    """
    b = _get_bout(bout_id)
    proposed = load_proposed_touches(b.touches_csv)
    data = store.load(bout_id)
    for p in proposed:
        st = data["touch_states"].get(p["id"], {})
        p["state"] = st.get("state", "pending")
        p["scorer"] = st.get("scorer")
        p["corrected_time_s"] = st.get("time_s")
    # the timeline needs a scale, and the metrics CSV is authoritative for it
    import csv as _csv
    with open(b.metrics_csv) as f:
        rows = list(_csv.DictReader(f))
    duration_s = float(rows[-1]["time_s"]) if rows else 0.0
    return {
        "bout_id": bout_id,
        "duration_s": duration_s,
        "has_video": bool(b.video),
        "proposed": proposed,
        "added": data["added_touches"],
        "unreliable_segments": data["unreliable_segments"],
        "reanchors": data["reanchors"],
        "lunges": sorted(data["lunges"], key=lambda l: l["time_s"]),
        "progress": store.review_progress(bout_id, proposed),
    }


@app.get("/api/bouts/{bout_id}/metrics")
def get_metrics(bout_id: str):
    """
    Bout metrics, scoped to in-play segments when the user has confirmed touches.

    Both the whole-recording and in-play figures are returned. The difference
    between them is itself informative: on the club clip, reset periods are 43
    per cent of the recording and inflate movement totals by a third, so showing
    only the scoped number would hide how much the scoping mattered.
    """
    from in_play import (out_of_play_windows, in_play_mask, scope_distance,
                         closing_share, net_forward_movement)
    import numpy as np
    from generate_summary import compute_stats, load_rows

    b = _get_bout(bout_id)
    rows = load_rows(b.metrics_csv)
    try:
        whole = compute_stats(rows)
    except ValueError as e:
        # A bout where the tracker never held both fencers at once has no
        # distance samples, and every statistic here is derived from them. That
        # is a real outcome rather than a broken file: an upload shot from behind
        # the piste, or one showing a single fencer drilling, produces exactly
        # this. Raising through as a 500 told the user only that something had
        # gone wrong, when what they need is to know their footage did not track
        # and why that is not a crash. Found by the end-to-end test, on a bout
        # whose subject the detector never recognised as people at all.
        raise HTTPException(
            422, f"this bout has no usable measurements: {e}. The tracker never "
                 f"held both fencers in the same frame, so there is nothing to "
                 f"scope or aggregate. The annotated video is still viewable, "
                 f"and the usual cause is framing: both fencers have to be in "
                 f"shot and roughly side-on.")

    proposed = load_proposed_touches(b.touches_csv)
    confirmed = store.confirmed_touch_times(bout_id, proposed)
    data = store.load(bout_id)

    t = np.array([float(r["time_s"]) for r in rows])
    d = np.array([float(r["distance_raw_m"]) if r["distance_raw_m"] else np.nan
                  for r in rows])
    end_time = float(t[-1]) if len(t) else 0.0

    windows = out_of_play_windows([c["time_s"] for c in confirmed], end_time)
    # segments the user marked unreliable are excluded on the same footing as
    # resets: both are periods whose data should not reach an aggregate
    windows += [(s["start_s"], s["end_s"]) for s in data["unreliable_segments"]]
    mask = in_play_mask(t, windows)

    def movement(prefix):
        """
        In-play movement figures for one fencer.

        Net forward movement and closing share are the reliable pair, and they are
        the only movement figures returned. The cumulative push and pull totals are
        deliberately absent: B1g traced their error to the per-frame movement cap
        and measured it at 24 m on a 14 m piste, so they are wrong rather than
        approximate, and an API that returns them invites a client to display them.
        They remain in the pipeline's CSV, which is the evidence artefact.

        The scoped figure is called movement rather than displacement on purpose.
        Excluding the resets breaks the position series, so it does not telescope
        to a first-to-last difference and can exceed the whole-recording
        displacement, which is where the ground gained in a phrase is given back.
        """
        cs = closing_share(rows, prefix, mask)
        return {
            "net_forward_movement_m": round(net_forward_movement(rows, prefix, mask), 2),
            "closing_share_pct": round(100.0 * cs, 1) if cs is not None else None,
        }

    dist = scope_distance(d, mask)
    scoped = {
        "in_play_fraction": round(float(mask.mean()), 3) if len(t) else 0.0,
        "mean_distance_m": round(float(dist.mean()), 2) if len(dist) else None,
        "distance_samples": int(len(dist)),
        "fencer_1": movement("f1"),
        "fencer_2": movement("f2"),
    }
    return {
        "bout_id": bout_id,
        "confirmed_touches": confirmed,
        "whole_recording": whole,
        "in_play": scoped,
        "scoping_basis": ("confirmed touches" if confirmed else
                          "none: no touches confirmed yet, so in-play equals the "
                          "whole recording"),
        # whole["movement_basis"] says which derivation route was used. It is
        # lifted to the top level as well because it qualifies the in-play
        # figures identically, and a caller reading only the scoped block would
        # otherwise not see it.
        "movement_basis": whole["movement_basis"],
    }


# Fields in the cached stats that do not depend on which touch file was used. A
# summary is stale when any of these has moved, which means the metrics were
# recomputed after the summary was written.
_SUMMARY_INVARIANTS = ("duration_s", "frames_total", "coverage_pct",
                       "pose_method_pct", "distance_m", "time_in_zone_pct",
                       "movement_basis", "fencer_1", "fencer_2")


@app.get("/api/bouts/{bout_id}/summary")
def get_summary(bout_id: str):
    """
    The cached LLM summary for a bout, and whether it still matches the metrics.

    Reading only. Generating a summary costs an API call, so it stays a deliberate
    command-line step rather than something a button can trigger by accident.

    The staleness check earns its place. A summary is a file on disk with no link to
    the data it was written from, so re-running detection leaves a confident piece
    of prose describing numbers that no longer exist. This project has already
    shipped one summary that faithfully reported a mis-specified input, and the
    movement metrics have been redefined three times, so a summary that silently
    predates the current CSV is a real hazard rather than a hypothetical one.

    Staleness compares the stats fields that do not depend on the touch file. The
    touch list is excluded on purpose: the summary may have been generated from
    ground truth, from detector output or from an exported review, and disagreeing
    with whichever is on disk now is not the same as being out of date.
    """
    b = _get_bout(bout_id)
    base = os.path.splitext(b.metrics_csv)[0]
    # discover_bouts already resolves the summary path, so prefer it and fall back
    # to the naming convention. Deriving it twice invites the two to disagree.
    md_path = b.summary_md or f"{base}_summary.md"
    meta_path = f"{base}_summary.meta.json"
    if not os.path.exists(md_path):
        return {"exists": False,
                "hint": (f'python3 generate_summary.py --csv "{b.metrics_csv}" '
                         f'--touches <touches.csv>')}

    import json as _json
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
        except ValueError:
            meta = {}

    stale, reason = False, None
    cached = meta.get("stats")
    if cached is None:
        stale, reason = True, "no cached stats to compare against"
    else:
        from generate_summary import compute_stats, load_rows
        current = compute_stats(load_rows(b.metrics_csv))
        moved = [k for k in _SUMMARY_INVARIANTS if cached.get(k) != current.get(k)]
        if moved:
            stale = True
            reason = ("the metrics changed after this was written: "
                      + ", ".join(moved))

    with open(md_path) as f:
        markdown = f.read()
    return {"exists": True, "markdown": markdown,
            "model": meta.get("model"), "stale": stale, "stale_reason": reason,
            "generated_from": ("touch data" if (cached or {}).get("touches")
                               else "the whole recording, no touch data")}


@app.post("/api/bouts/{bout_id}/export-touches")
def export_touches(bout_id: str):
    """
    Write the user's confirmed touches to a CSV the rest of the pipeline can read.

    WHY THIS EXISTS. The loop was open at the last step. A user could confirm,
    correct, add and reject touches, and the review interface would rescope its own
    metrics accordingly, but nothing downstream could see any of it: in_play.py and
    generate_summary.py take a touch file by path, and the only files available were
    the detector's raw proposals and the hand-labelled ground truth. So the summary
    a reader actually sees was still built from unreviewed detector output, which
    undercuts the project's central claim that user correction improves the output.
    This writes the reviewed list in the same schema as the ground-truth files, so
    both downstream stages accept it unchanged.

    The file is written beside the metrics CSV under a new name. Nothing existing is
    modified, which is the same guarantee the annotation store makes: a reprocess
    must never destroy user work and user work must never require recovering a
    mutated artefact.

    Pending proposals are excluded, because a proposal nobody has looked at is not
    evidence. That makes a partially reviewed export quietly incomplete, so the
    header records how many were confirmed, added, rejected and still pending, and
    the response says whether review was finished.
    """
    b = _get_bout(bout_id)
    proposed = load_proposed_touches(b.touches_csv)
    confirmed = store.confirmed_touch_times(bout_id, proposed)
    if not confirmed:
        raise HTTPException(
            400, "nothing to export: no touches confirmed or added yet. An empty "
                 "touch file would scope every metric to nothing, which is worse "
                 "than having no file at all.")

    progress = store.review_progress(bout_id, proposed)
    base = os.path.splitext(b.metrics_csv)[0]
    out_path = f"{base}_touches_confirmed.csv"

    import csv as _csv
    with open(out_path, "w", newline="") as f:
        f.write(f"# Touches confirmed through the review interface for {bout_id}.\n")
        f.write(f"# {progress['confirmed']} detector proposals confirmed, "
                f"{progress['user_added']} added by hand, "
                f"{progress['rejected']} rejected, "
                f"{progress['proposed'] - progress['reviewed']} left pending.\n")
        if not progress["complete"]:
            f.write("# REVIEW INCOMPLETE: pending proposals are absent from this "
                    "file, so the touch count is a lower bound.\n")
        f.write("# Schema matches ground_truth/*_touches.csv so in_play.py and "
                "generate_summary.py read it unchanged. A scorer column means "
                "touch_provenance() reports these as human_confirmed, which is "
                "what they are.\n")
        w = _csv.DictWriter(f, fieldnames=["time_s", "scorer", "annulled", "notes"])
        w.writeheader()
        for t in confirmed:
            w.writerow({"time_s": round(t["time_s"], 2),
                        "scorer": t["scorer"],
                        "annulled": 0,
                        "notes": t["origin"]})
    return {"path": out_path, "touches": len(confirmed),
            "review_complete": progress["complete"],
            "next": (f'python3 generate_summary.py --csv "{b.metrics_csv}" '
                     f'--touches "{out_path}"')}


@app.post("/api/bouts/{bout_id}/export-reanchors")
def export_reanchors(bout_id: str):
    """
    Write the user's re-anchor corrections where the pipeline can read them.

    Action 4 is the only one that changes tracking rather than interpretation, so it
    cannot take effect in this interface: the tracker has already run. Until now the
    corrections were stored and nothing consumed them, which meant the interface
    offered a repair that did nothing at all. This closes that, in the same shape as
    the touch export: a file beside the metrics CSV, nothing existing modified, and
    the exact command to run returned with it.

    Corrections already marked applied are included rather than filtered out. A
    reprocess starts from the original video every time, so every correction is
    needed on every run; excluding the applied ones would silently undo them.
    """
    b = _get_bout(bout_id)
    data = store.load(bout_id)
    reanchors = data["reanchors"]
    if not reanchors:
        raise HTTPException(400, "nothing to export: no re-anchor corrections recorded")

    import json as _json
    base = os.path.splitext(b.metrics_csv)[0]
    out_path = f"{base}_reanchors.json"
    with open(out_path, "w") as f:
        _json.dump([{"time_s": a["time_s"], "slot": a["slot"],
                     "x": a["x"], "y": a["y"]} for a in reanchors], f, indent=2)

    # The rerun must start from the SOURCE video. b.video is the annotated output,
    # and running detection over a clip with boxes and text already burnt into it
    # would be detecting on top of the overlay.
    stem = os.path.basename(base).replace("_distance", "")
    source = os.path.join(PROTOTYPE_DIR, f"{stem}.mp4")
    have_source = os.path.exists(source)

    # And it must carry the same piste configuration, or the rerun changes two things
    # at once. Omitting it once cost 18 points of coverage on clip 2 and looked
    # exactly like a code regression, so the command is assembled rather than left
    # to memory.
    piste = None
    for candidate in (f"piste_{stem}.json",
                      f"piste_{stem.replace('fencing_clip', 'clip')}.json",
                      f"piste_clip{stem.replace('fencing_clip', '') or '1'}.json"):
        if os.path.exists(os.path.join(PROTOTYPE_DIR, candidate)):
            piste = candidate
            break

    cmd = (f'python3 run_detection.py --video "{source if have_source else stem + ".mp4"}" '
           f'--output results_reanchored --reanchors "{out_path}"')
    if piste:
        cmd += f' --piste-config {piste}'

    return {
        "path": out_path,
        "corrections": len(reanchors),
        "pending": sum(1 for a in reanchors if not a["applied"]),
        "source_video_found": have_source,
        "piste_config": piste,
        "next": cmd,
    }


@app.post("/api/bouts/{bout_id}/reanchors/applied")
def mark_reanchors_applied(bout_id: str):
    """
    Record the user's statement that they have reprocessed with the corrections.

    Deliberately an assertion rather than a detection. Nothing in a pipeline run
    writes back to the annotation store, and inferring a reprocess from a newer CSV
    timestamp would be a guess dressed as a fact.
    """
    _get_bout(bout_id)
    changed = store.mark_reanchors_applied(bout_id)
    return {"ok": True, "marked": changed}


class LungeIn(BaseModel):
    time_s: float = Field(..., ge=0)
    slot: int = Field(..., ge=0, le=1)
    note: str = ""


@app.post("/api/bouts/{bout_id}/lunges")
def add_lunge(bout_id: str, body: LungeIn):
    """
    Label the peak of one lunge, for evaluating the pose model.

    This is not a fifth annotation action. The four designed actions let a user
    repair the system's output; this lets a user grade it. TODO B1h found that
    pose stance features do not mark awarded touches, but touch times are a weak
    proxy for lunges in both directions, since most lunges miss and some touches
    are not lunges. These labels test the hypothesis directly.
    """
    _get_bout(bout_id)
    store.add_lunge(bout_id, body.time_s, body.slot, body.note)
    return {"ok": True}


@app.delete("/api/bouts/{bout_id}/lunges/{lunge_id}")
def remove_lunge(bout_id: str, lunge_id: str):
    _get_bout(bout_id)
    if not store.remove_lunge(bout_id, lunge_id):
        raise HTTPException(404, f"no such lunge: {lunge_id}")
    return {"ok": True}


WEB_VIDEO_DIR = os.path.join(PROTOTYPE_DIR, "web_video")


def _web_playable(src):
    """
    Return a browser-playable copy of an annotated render, transcoding once and
    caching the result.

    OpenCV's VideoWriter writes MPEG-4 Part 2 with the `mp4v` tag, which browsers
    generally refuse to decode: the element loads, reports readyState 0, and
    plays nothing. The pipeline is deliberately left alone rather than switched
    to H.264 at write time, because the `avc1` fourcc is not available in every
    OpenCV build and a pipeline that fails to write video on some machines would
    be a worse problem than a transcode here. Transcoding is done once per bout
    and cached, so the cost is paid on first view rather than on every request.
    """
    os.makedirs(WEB_VIDEO_DIR, exist_ok=True)
    stem = os.path.basename(src).replace(".mp4", "")
    parent = os.path.basename(os.path.dirname(src))
    out = os.path.join(WEB_VIDEO_DIR, f"{parent}__{stem}.h264.mp4")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    cmd = ["ffmpeg", "-y", "-i", src,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           # faststart moves the index to the front so the browser can start
           # playing and seeking before the whole file has arrived
           "-movflags", "+faststart", "-an", out]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(out):
        raise HTTPException(500, "failed to transcode the annotated video")
    return out


@app.get("/api/bouts/{bout_id}/video")
def get_video(bout_id: str):
    """
    Stream a browser-playable copy of the annotated render for this bout.

    Served with range-request support so the player can seek, which is what makes
    review practical: a user jumps to a proposed touch, watches two seconds, and
    decides. Without seeking they would have to scrub linearly through three
    minutes per decision, and the workflow's claim to save effort would not
    survive that.
    """
    b = _get_bout(bout_id)
    if not b.video:
        raise HTTPException(404, "no annotated video for this bout")
    return FileResponse(_web_playable(b.video), media_type="video/mp4")


# --- the four annotation actions ---------------------------------------

@app.post("/api/bouts/{bout_id}/touches/{touch_id}/decision")
def decide_touch(bout_id: str, touch_id: str, body: TouchDecision):
    """Action 1: confirm, correct or reject a proposed touch."""
    _get_bout(bout_id)
    try:
        store.set_touch_state(bout_id, touch_id, body.state,
                              scorer=body.scorer, time_s=body.time_s)
    except ValueError as e:
        raise HTTPException(400, str(e))
    proposed = load_proposed_touches(_get_bout(bout_id).touches_csv)
    return {"ok": True, "progress": store.review_progress(bout_id, proposed)}


@app.post("/api/bouts/{bout_id}/touches")
def add_touch(bout_id: str, body: NewTouch):
    """Action 2: add a touch the detector missed."""
    _get_bout(bout_id)
    try:
        new_id = store.add_touch(bout_id, body.time_s, body.scorer, body.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "id": new_id}


@app.delete("/api/bouts/{bout_id}/touches/{touch_id}")
def delete_added_touch(bout_id: str, touch_id: str):
    _get_bout(bout_id)
    if not store.remove_added_touch(bout_id, touch_id):
        raise HTTPException(404, f"no user-added touch {touch_id}")
    return {"ok": True}


@app.post("/api/bouts/{bout_id}/segments")
def add_segment(bout_id: str, body: Segment):
    """Action 3: mark a time range as tracking-unreliable."""
    _get_bout(bout_id)
    try:
        store.add_unreliable_segment(bout_id, body.start_s, body.end_s, body.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/bouts/{bout_id}/segments/{segment_id}")
def delete_segment(bout_id: str, segment_id: str):
    _get_bout(bout_id)
    if not store.remove_unreliable_segment(bout_id, segment_id):
        raise HTTPException(404, f"no segment {segment_id}")
    return {"ok": True}


@app.post("/api/bouts/{bout_id}/reanchor")
def add_reanchor(bout_id: str, body: Reanchor):
    """
    Action 4: tell the tracker it has a fencer wrong at this moment.

    Recorded as pending rather than applied. Unlike the other three actions this
    changes tracking rather than interpretation, so it takes effect only when the
    pipeline is rerun, and the response says so plainly rather than implying the
    correction is already in force.
    """
    _get_bout(bout_id)
    try:
        store.add_reanchor(bout_id, body.time_s, body.slot, body.x, body.y)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True,
            "note": "recorded as pending; re-run the detection pipeline to apply it"}


def _source_for_bout(bout_id, b):
    """
    The ORIGINAL video a bout was produced from, and the piste config used.

    Needed by any reprocess, and the two kinds of bout keep it in different
    places: an uploaded bout's source is recorded on its job, while an evaluation
    bout's sits in the prototype directory under the clip's stem.

    The distinction that matters is that this must never return the ANNOTATED
    render. That file has boxes, labels and a distance readout burnt into it, so
    re-running detection over it would be detecting on top of the overlay.

    Returns (source_path or None, piste_config_path or None).
    """
    directory = bout_id.split(":")[0]
    if directory.startswith("upload_"):
        job = job_store.load(directory[len("upload_"):])
        if job:
            piste = job.get("piste") or {}
            config = job.get("piste_config_path")
            source = job.get("source_path")
            # Existence is checked here as well as on the evaluation branch. The
            # job record says where the source WAS, and an upload whose files
            # have since been deleted would otherwise queue a job that fails
            # several minutes later instead of being refused immediately.
            return (source if source and os.path.exists(source) else None,
                    config if piste.get("polygon") and config
                    and os.path.exists(config) else None)
        return None, None

    base = os.path.splitext(b.metrics_csv)[0]
    stem = os.path.basename(base).replace("_distance", "")
    source = os.path.join(PROTOTYPE_DIR, f"{stem}.mp4")
    # The piste config has to come with it, or a rerun changes two things at
    # once. Omitting it cost 18 points of coverage on clip 2 and looked exactly
    # like a code regression.
    config = None
    for candidate in (f"piste_{stem}.json",
                      f"piste_{stem.replace('fencing_clip', 'clip')}.json",
                      f"piste_clip{stem.replace('fencing_clip', '') or '1'}.json"):
        path = os.path.join(PROTOTYPE_DIR, candidate)
        if os.path.exists(path):
            config = path
            break
    return (source if os.path.exists(source) else None), config


@app.post("/api/bouts/{bout_id}/reprocess")
def reprocess_bout(bout_id: str):
    """
    Re-run the pipeline on this bout's source video, applying the user's
    re-anchor corrections.

    WHY THIS EXISTS. Action 4 changes tracking rather than interpretation, so it
    can only take effect on a reprocess. Until now the interface's answer to "I
    have corrected the tracking" was a command line for the user to go and type
    in a terminal, which is precisely the arrangement this whole application
    layer exists to remove. The corrections were recorded, exported, and then
    depended on the user being someone who could run the pipeline by hand.

    The result is a NEW bout rather than an overwrite. That is the same guarantee
    the annotation store makes: a reprocess must never destroy a previous result,
    and here it also means the before and after can be opened side by side, which
    is the only way to see whether a correction helped.
    """
    b = _get_bout(bout_id)
    source, piste_config = _source_for_bout(bout_id, b)
    if not source:
        raise HTTPException(
            404, "the original video for this bout could not be found, so it "
                 "cannot be reprocessed. Only the annotated render is on disk, "
                 "and re-running detection over that would be detecting on top "
                 "of the overlay.")

    data = store.load(bout_id)
    reanchors = data["reanchors"]

    job = job_store.create(
        kind="processing",
        filename=f"reprocess of {bout_id}",
        reprocess_of=bout_id,
        confirm_piste=False,
        source_path=source,
        # The piste region is carried over rather than re-measured, so the rerun
        # changes exactly one thing: the corrections.
        piste=({"needed": True, "polygon": True,
                "decision": "carried over from the original run"}
               if piste_config else {"needed": False, "polygon": None}),
        piste_config_path=piste_config or "",
        piste_result_path="",
    )
    job_id = job["job_id"]
    output_dir = os.path.join(UPLOAD_RESULTS_ROOT, f"upload_{job_id}")
    os.makedirs(output_dir, exist_ok=True)

    reanchor_path = ""
    if reanchors:
        # Corrections already marked applied are included rather than filtered
        # out. A reprocess starts from the original video every time, so every
        # correction is needed on every run; excluding the applied ones would
        # silently undo them.
        import json as _json
        reanchor_path = os.path.join(JOB_ROOT, f"{job_id}_reanchors.json")
        with open(reanchor_path, "w") as f:
            _json.dump([{"time_s": a["time_s"], "slot": a["slot"],
                         "x": a["x"], "y": a["y"]} for a in reanchors], f, indent=2)

    stem = os.path.splitext(os.path.basename(source))[0]
    job_store.update(
        job_id,
        output_dir=output_dir,
        reanchor_path=reanchor_path,
        log_path=os.path.join(JOB_ROOT, f"{job_id}.log"),
        web_video_path=os.path.join(
            WEB_VIDEO_DIR, f"upload_{job_id}__{stem}_annotated.h264.mp4"),
        video_info=_probe_video(source),
    )
    os.makedirs(WEB_VIDEO_DIR, exist_ok=True)
    runner.submit(job_id)
    return {
        "job_id": job_id,
        "corrections": len(reanchors),
        "piste_config": os.path.basename(piste_config) if piste_config else None,
        "note": ("re-running with your corrections; the result arrives as a new "
                 "bout so you can compare it against this one"
                 if reanchors else
                 "no re-anchor corrections recorded, so this is a plain re-run"),
    }


@app.get("/api/bouts/{bout_id}/reanchor-outcomes")
def get_reanchor_outcomes(bout_id: str):
    """
    Whether the corrections applied to this bout actually changed anything.

    A re-anchor is not a force-assignment: it moves the slot's reference point
    and clears its gates for one frame, then lets ordinary matching resume. A
    correction the matcher disagrees with leaves no trace at all, which was
    confirmed on real footage when a mis-aimed correction produced output
    byte-identical to its baseline. Without this the user re-runs a job that
    takes minutes and is told nothing, and cannot tell "my correction was wrong"
    from "my correction was right and did not help".
    """
    b = _get_bout(bout_id)
    base = os.path.splitext(b.metrics_csv)[0]
    path = f"{base}_reanchor_outcomes.json"
    if not os.path.exists(path):
        return {"exists": False, "outcomes": []}
    import json as _json
    with open(path) as f:
        outcomes = _json.load(f)
    return {
        "exists": True,
        "outcomes": outcomes,
        "applied": sum(1 for o in outcomes if o["outcome"] == "applied"),
        "total": len(outcomes),
    }


class ScorerRequest(BaseModel):
    """
    Which fencer the green lamp belongs to.

    Required, with no default, because nothing in the image says it and a guess
    would be wrong half the time in a way that looks authoritative. It is one
    confirmation per bout, which the design already asks the user for in the same
    spirit as the piste region.
    """
    green_is: str = Field(..., pattern="^(left|right)$")


@app.post("/api/bouts/{bout_id}/propose-scorers")
def propose_scorers(bout_id: str, body: ScorerRequest):
    """
    Read the scoring lamps and propose who scored each confirmed touch.

    WHY THIS RUNS IN THE REQUEST. It decodes a handful of frames per touch and
    loads no models, so it costs seconds rather than the minutes a pipeline stage
    takes. The rule that inference stays out of the request path is about the
    models; this is colour thresholding.

    WHY IT ONLY LOOKS AT TOUCHES THE USER HAS CONFIRMED. The lamps fire whenever
    the circuit closes, which includes fencers testing weapons against the piste
    or each other's guards, routinely just after a touch and before coming back
    on guard. Reading them only at times a touch is already known to have
    happened sidesteps that whole class of spurious firing, and it is why this
    can never become a touch detector.

    Proposals are returned rather than applied. The user still confirms each one,
    which is the same contract as every other suggestion the system makes.
    """
    b = _get_bout(bout_id)
    source, _ = _source_for_bout(bout_id, b)
    if not source:
        raise HTTPException(
            404, "the original video for this bout could not be found, and the "
                 "lamps cannot be read from the annotated render because it is "
                 "re-encoded.")

    proposed = load_proposed_touches(b.touches_csv)
    confirmed = store.confirmed_touch_times(bout_id, proposed)
    if not confirmed:
        raise HTTPException(
            400, "no touches confirmed yet. The lamps are read only at times a "
                 "touch is already known to have happened, because they also "
                 "fire when fencers test their weapons.")

    from detect_scorer import classify, fit_thresholds, lamp_response

    times = [c["time_s"] for c in confirmed]
    responses = lamp_response(source, times)

    # Calibrate on whatever the user has already attributed by hand. With none,
    # fall back to fitting on the responses themselves, which is weaker and is
    # reported as such rather than presented as the same thing.
    labelled = [(r, c["scorer"]) for r, c in zip(responses, confirmed)
                if c.get("scorer") in ("left", "right", "double")]
    if labelled:
        th = fit_thresholds([r for r, _ in labelled],
                            [l for _, l in labelled], green_is=body.green_is)
        basis = f"calibrated on {len(labelled)} touch(es) you already attributed"
    else:
        th = fit_thresholds(responses, ["unknown"] * len(responses),
                            green_is=body.green_is)
        basis = ("no touches attributed yet, so the thresholds are guessed from "
                 "the lamp readings alone and are weaker than they would be "
                 "after you attribute two or three by hand")

    out = []
    for c, r in zip(confirmed, responses):
        pred = classify(r, th)
        out.append({"time_s": c["time_s"], "current": c.get("scorer"),
                    "proposed": pred["scorer"], "confidence": pred["confidence"],
                    "red_delta": round(r["red"], 1),
                    "green_delta": round(r["green"], 1)})
    decided = sum(1 for o in out if o["proposed"] != "unknown")
    return {"proposals": out, "basis": basis, "decided": decided,
            "total": len(out),
            "note": ("The green lamp is the reliable half. Measured across all "
                     "four evaluation clips it identified whether one named "
                     "fencer was involved in 27 touches out of 27; telling a "
                     "single touch from a double needs the red lamp, which is "
                     "contaminated by anything permanently red in shot.")}


@app.post("/api/bouts/{bout_id}/propose-lunges")
def propose_lunges(bout_id: str, slot: int = Query(0, ge=0, le=1)):
    """
    Propose lunges for one fencer, calibrated on the ones already confirmed.

    WHY IT CALIBRATES INSTEAD OF TRANSFERRING. The stance ratio is not
    view-invariant: an operating point fitted on one clip reaches F1 0.22 on
    another while firing on a quarter of all windows. Measured, clip 3 calibrates
    to 1.902 and clip 2 to 2.706, a 42 per cent difference in what counts as a
    lunge-like posture. So the threshold comes from lunges the user has confirmed
    on THIS bout, which is the correction mechanism the design already uses
    rather than a new demand on them.

    Refusing below five confirmed lunges is deliberate rather than cautious. A
    threshold fitted on two fires on a quarter of the bout, and a user who has to
    reject every proposal is worse off than one who was offered none.
    """
    b = _get_bout(bout_id)
    from detect_lunges import (MIN_CALIBRATION_LUNGES, calibrate, propose,
                               stance_ratio_series)

    times, ratios = stance_ratio_series(b.metrics_csv, slot)
    if len(times) == 0:
        raise HTTPException(
            400, "this bout has no pose stance data. It was processed before the "
                 "stance columns existed, or pose never ran on it.")

    data = store.load(bout_id)
    confirmed = sorted(l["time_s"] for l in data["lunges"] if l["slot"] == slot)
    threshold, used = calibrate(times, ratios, confirmed)
    if threshold is None:
        raise HTTPException(
            400, f"only {used} confirmed lunge(s) for Fencer {slot + 1}; "
                 f"{MIN_CALIBRATION_LUNGES} are needed to calibrate. Label a few "
                 f"more with the 1 and 2 keys and try again.")

    proposals = propose(times, ratios, threshold,
                        skip_before=confirmed[MIN_CALIBRATION_LUNGES - 1] + 2.0)
    # Anything the user has already labelled is dropped: they are being offered
    # what to look at next, not their own work back.
    fresh = [p for p in proposals
             if all(abs(p["time_s"] - c) > 0.5 for c in confirmed)]
    return {
        "slot": slot,
        "calibrated_on": used,
        "threshold": round(threshold, 3),
        "proposals": fresh,
        "note": ("Measured on the one clip with enough labels to say anything, "
                 "precision 0.73 and recall 0.76 on 25 held-out lunges. "
                 "Precision is a LOWER bound: a proposal on a real lunge you had "
                 "not labelled counts against it."),
    }


@app.get("/api/bouts/{bout_id}/profile")
def fencer_profile(bout_id: str):
    """
    A per-fencer profile of one bout, as six axes that can be drawn as a radar.

    WHY THIS IS NOT THE WITHDRAWN PUSH / PULL METRIC WEARING A NEW SHAPE. That
    metric accumulated per-frame position deltas, and a fencer re-acquired after
    a tracking dropout contributed a one-sided step that never cancelled: clip
    3's Fencer 2 accumulated +23.01 m against an endpoint difference of -0.94 m.
    Every axis here is either an instantaneous reading averaged over frames or a
    count of touches the user confirmed, so a dropout displaces a few samples out
    of thousands instead of banking itself permanently.

    WHY IT CAN REFUSE. All six axes are per-fencer and therefore assume slot
    identity held. On clip 4 it did not, and a profile drawn there would describe
    the tracker while looking exactly as convincing as a real one. The refusal
    carries the swap count so the interface can say why.
    """
    b = _get_bout(bout_id)
    from fencer_profile import build

    proposed = load_proposed_touches(b.touches_csv)
    confirmed = store.confirmed_touch_times(bout_id, proposed)
    data = store.load(bout_id)
    lunges = {}
    for l in data["lunges"]:
        lunges.setdefault(l["slot"] + 1, []).append(l["time_s"])

    result = build(b.metrics_csv, confirmed, lunges=lunges)
    # The profile rests on the user's confirmed touches, so how many there are is
    # part of reading it: three axes are undefined until some touches exist, and
    # a profile built on two touches should not be read like one built on twenty.
    result["confirmed_touches"] = len(confirmed)
    result["confirmed_lunges"] = {k: len(v) for k, v in lunges.items()}
    return result


@app.post("/api/bouts/{bout_id}/summary/generate")
def generate_summary_for_bout(bout_id: str, force: bool = False):
    """
    Generate the written summary for this bout, as a background job.

    Deliberately never automatic. It costs a paid API call per run, and a job
    that quietly spent money on every upload would reverse a decision the
    annotation API took on purpose. What has changed is only that the user
    presses a button rather than being handed a command to type: the decision is
    still theirs, the terminal is no longer required.

    The touch file is the reviewed export where one exists, so the summary
    describes the record the user confirmed rather than the detector's first
    guess. That was the point of the export, and without preferring it here the
    prose a reader sees would still come from unreviewed output.
    """
    b = _get_bout(bout_id)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            400, f"No provider API key is available, so the summary cannot be "
                 f"generated. The server looks in ANTHROPIC_API_KEY and then in "
                 f"the macOS Keychain under the service '{KEYCHAIN_SERVICE}', "
                 f"once, at startup. Add the key there and restart the server.")

    base = os.path.splitext(b.metrics_csv)[0]
    reviewed = f"{base}_touches_confirmed.csv"
    touches = reviewed if os.path.exists(reviewed) else b.touches_csv

    job = job_store.create(
        kind="summary",
        filename=f"summary for {bout_id}",
        summary_of=bout_id,
        metrics_csv=b.metrics_csv,
        touches_csv=touches,
        force=force,
        source_path=b.metrics_csv,
        output_dir=os.path.dirname(b.metrics_csv),
    )
    job_store.update(job["job_id"],
                     log_path=os.path.join(JOB_ROOT, f"{job['job_id']}.log"))
    runner.submit(job["job_id"])
    return {
        "job_id": job["job_id"],
        "touches_used": ("the touches you confirmed" if touches == reviewed
                         else "the detector's proposals, unreviewed"),
    }


# --- processing jobs ----------------------------------------------------
#
# The endpoints below are what turn the pipeline from a command line into an
# application. Everything above this point reads artefacts that someone had
# already produced in a terminal.

def _probe_video(path):
    """
    Confirm the upload is a video this pipeline can open, and measure it.

    Done in the request, deliberately, because it is the one check that must
    happen before a job is accepted: OpenCV opening the file is the same test
    the pipeline itself will apply, so failing it here turns a job that would
    die two stages later into an immediate, explainable rejection. It costs one
    file open and one frame read, and loads no models.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or width == 0 or height == 0:
        return None
    return {"width": width, "height": height, "fps": round(fps, 2),
            "frames": frames,
            "duration_s": round(frames / fps, 1) if fps else None}


def _job_view(job):
    """
    The job as the client sees it.

    Adds the things that are derived rather than stored: queue position, elapsed
    time, and the bout id, which only means anything once there is an output
    directory for it to point at.
    """
    import time as _time
    view = dict(job)
    view.pop("log_tail", None)
    view["queue_position"] = runner.queue_position(job["job_id"])
    started, finished = job.get("started_at"), job.get("finished_at")
    view["elapsed_s"] = round((finished or _time.time()) - started, 1) if started else None
    if job["state"] == "done":
        stem = os.path.splitext(os.path.basename(job["source_path"]))[0]
        view["bout_id"] = f"{os.path.basename(job['output_dir'])}:{stem}"
    if job["state"] in ("failed", "interrupted"):
        view["log_tail"] = job.get("log_tail", [])
    return view


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    confirm_piste: bool = Form(True),
    # 0 means "use the pipeline's default". The upper bound is not arbitrary
    # politeness: pose stride changes what the numbers MEAN, since a stride of 3
    # is what produced the 27 per cent pose-availability figure the report
    # discusses, and a stride of several hundred would report figures derived
    # from a handful of frames while looking like every other run.
    pose_stride: int = Form(0, ge=0, le=30),
):
    """
    Accept a bout video and queue it for processing.

    The request writes the file to disk, checks it opens, and returns. It does
    not process anything, which is the separation the architecture requires and
    the reason this layer exists at all: a three minute clip takes minutes to
    process, and a request that waited for it would time out in the proxy, the
    browser, or both, while holding a worker for the duration.

    The upload is streamed in chunks rather than read whole. A 500 MB file read
    into memory to be written straight back out is 500 MB of resident memory
    spent for nothing, on the same machine that is about to load three models.
    """
    filename = os.path.basename(video.filename or "bout.mp4")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"unsupported file type '{ext}'. Accepted: "
                 f"{', '.join(ALLOWED_EXTENSIONS)}")

    job = job_store.create(
        filename=filename,
        confirm_piste=confirm_piste,
        pose_stride=pose_stride or None,
        source_path="", output_dir="", log_path="",
        piste_result_path="", piste_config_path="", web_video_path="",
    )
    job_id = job["job_id"]

    upload_dir = os.path.join(UPLOAD_ROOT, job_id)
    os.makedirs(upload_dir, exist_ok=True)
    # The stored name comes from the job, not from the upload. A filename
    # arriving over the wire is user input, and it also becomes the bout id and
    # the stem of every output file, so a name with a space or a slash in it
    # would propagate into paths the whole pipeline then has to quote correctly.
    source_path = os.path.join(upload_dir, f"bout_{job_id}{ext}")

    written = 0
    try:
        with open(source_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"file exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        job_store.delete(job_id)
        raise

    info = _probe_video(source_path)
    if info is None:
        shutil.rmtree(upload_dir, ignore_errors=True)
        job_store.delete(job_id)
        raise HTTPException(
            400, "the file could not be opened as a video. It may be corrupt, or "
                 "in a container this build of OpenCV cannot read.")

    output_dir = os.path.join(UPLOAD_RESULTS_ROOT, f"upload_{job_id}")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"bout_{job_id}"
    job_store.update(
        job_id,
        source_path=source_path,
        output_dir=output_dir,
        size_bytes=written,
        video_info=info,
        log_path=os.path.join(JOB_ROOT, f"{job_id}.log"),
        piste_result_path=os.path.join(upload_dir, "piste_measurement.json"),
        piste_config_path=os.path.join(upload_dir, "piste.json"),
        # Written where the existing video endpoint already looks for a
        # browser-playable copy, so transcoding during the job removes the
        # on-demand ffmpeg run rather than duplicating it.
        web_video_path=os.path.join(
            WEB_VIDEO_DIR, f"upload_{job_id}__{stem}_annotated.h264.mp4"),
    )
    os.makedirs(WEB_VIDEO_DIR, exist_ok=True)
    runner.submit(job_id)
    return _job_view(job_store.load(job_id))


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [_job_view(j) for j in job_store.list()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    return _job_view(job)


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_job_log(job_id: str):
    """The full pipeline output for a job, for when the summary is not enough."""
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    path = job.get("log_path")
    if not path or not os.path.exists(path):
        return PlainTextResponse("no output recorded yet")
    with open(path) as f:
        return PlainTextResponse(f.read())


@app.get("/api/jobs/{job_id}/frame")
def get_job_frame(job_id: str):
    """
    A still from the uploaded video, for drawing the piste region over.

    Taken from a quarter of the way in rather than from frame one. Broadcast
    footage routinely opens on a title card or a crowd shot, and a first frame
    with no fencers in it is exactly the wrong picture to ask someone to confirm
    a fencer-detection boundary against.
    """
    import cv2
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    src = job.get("source_path")
    if not src or not os.path.exists(src):
        raise HTTPException(404, "no source video for this job")

    out = os.path.join(os.path.dirname(src), "frame.jpg")
    if not os.path.exists(out):
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 4)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(500, "could not read a frame from the video")
        cv2.imwrite(out, frame)
    return FileResponse(out, media_type="image/jpeg")


class PisteDecision(BaseModel):
    """
    What the user decided about the measured piste region.

    `polygon` overrides the measurement; omitting it accepts what was measured.
    `skip` runs with no region at all, which is the right answer for footage
    containing nobody but the two fencers and the wrong one for a competition.
    """
    polygon: list[list[float]] | None = None
    skip: bool = False


@app.post("/api/jobs/{job_id}/piste")
def confirm_piste(job_id: str, body: PisteDecision):
    """
    Accept, adjust or skip the measured piste region, and let the job continue.

    WHY THE JOB PAUSES HERE. The region decides which detections the tracker is
    allowed to see, and getting it wrong is not a small error: rebuilding the
    reference results without the regions dropped the broadcast clip from 98.0
    per cent coverage to 80.1. It is also the one decision in the pipeline that
    a person can make far better than the system, because they can see at a
    glance whether the band drawn on the frame contains the fencers and excludes
    the referee.

    WHAT THE USER IS BEING ASKED. To confirm a measurement, not to produce a
    guess. The distinction is the whole reason the region is measured first: a
    polygon placed by eye on this project once admitted the adjacent piste and
    raised the count of physically impossible distance readings from 53 to 252,
    while the headline coverage figure went up. The interface therefore shows
    what was measured and offers agreement, not an empty canvas.
    """
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    if job["state"] != AWAITING_PISTE:
        raise HTTPException(
            409, f"job is {job['state']}, not waiting on a piste decision")

    piste = dict(job.get("piste") or {})
    if body.skip:
        piste["polygon"] = None
        piste["needed"] = False
        piste["decision"] = "skipped by the user"
    elif body.polygon:
        if len(body.polygon) < 3:
            raise HTTPException(400, "a polygon needs at least three vertices")
        piste["polygon"] = body.polygon
        piste["decision"] = "adjusted by the user"
        write_piste_config(piste, job["piste_config_path"])
    else:
        piste["decision"] = "accepted as measured"

    job_store.update(job_id, piste=piste, state="queued")
    runner.submit(job_id)
    return _job_view(job_store.load(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    if not runner.cancel(job_id):
        raise HTTPException(409, f"job is already {job['state']}")
    return _job_view(job_store.load(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """
    Remove a job and everything it produced.

    Needed rather than tidy. Each bout keeps its source, an annotated render, a
    browser copy of that render, a metrics CSV and a plot, which is several
    times the size of the upload, and nothing else in this system ever deletes
    anything. Without this the only way to reclaim the disk is to know the
    layout and use a terminal, which is the situation this whole layer exists to
    remove.
    """
    job = job_store.load(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    if job["state"] not in TERMINAL_STATES and job["state"] != AWAITING_PISTE:
        raise HTTPException(
            409, f"job is {job['state']}; cancel it before deleting")
    for path in (os.path.join(UPLOAD_ROOT, job_id), job.get("output_dir")):
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    for path in (job.get("log_path"), job.get("web_video_path")):
        if path and os.path.exists(path):
            os.remove(path)
    job_store.delete(job_id)
    return {"ok": True}


# --- disk ---------------------------------------------------------------

def _all_results_dirs():
    return RESULTS_DIRS + sorted(
        glob.glob(os.path.join(UPLOAD_RESULTS_ROOT, "upload_*")))


@app.get("/api/storage")
def get_storage():
    """
    Where the disk went, and how much of it can go.

    Worth an endpoint rather than a note in the README because nothing else in
    this system reclaims anything, and the transcode cache grows every time a
    bout is viewed. On the development machine it reached 313 MB unnoticed. The
    point of reporting it by category is that "how much" is not the useful
    question: the answer a user needs is which of it is derived and which is
    their own footage.
    """
    from storage import stale_jobs, storage_report
    report = storage_report(WEB_VIDEO_DIR, _all_results_dirs(),
                            UPLOAD_ROOT, JOB_ROOT, UPLOAD_RESULTS_ROOT)
    report["old_jobs"] = stale_jobs(job_store, older_than_days=30)
    return report


@app.post("/api/storage/cleanup")
def clean_storage(everything: bool = Query(
        False, description="also delete transcodes that are still serviceable")):
    """
    Delete cached video that is derived, never anything that is not.

    Defaults to the two kinds that cost nothing to lose: transcodes whose source
    video is gone, and transcodes older than the source they were made from,
    which the serving code would re-encode over anyway. Clearing the live cache
    as well costs a few seconds per bout on next view and has to be asked for.

    Nothing here can reach the pipeline results, the annotated videos, the
    uploaded sources or the annotations. The annotations matter most: they are 36
    hand-marked lunges and 27 hand-labelled touches that no amount of
    reprocessing would bring back.
    """
    from storage import clean
    return clean(WEB_VIDEO_DIR, _all_results_dirs(), orphans_only=not everything)


# --- static UI ----------------------------------------------------------
#
# Two interfaces are served, and both are kept deliberately.
#
# `/` is the React application: upload, job progress and review in one place.
# `/legacy` is the original no-build-step page, which reviews already-processed
# bouts and needs nothing but Python to run. It stays because it is the fallback
# when the React build is absent, and because the two are directly comparable:
# the same workflow, the same API, one with a build step and one without.

REACT_DIR = os.path.join(STATIC_DIR, "app")

if os.path.isdir(REACT_DIR):
    # Mounted rather than routed one file at a time, because a Vite build emits
    # hashed asset names that are not known here.
    app.mount("/app", StaticFiles(directory=REACT_DIR, html=True), name="app")


@app.get("/", response_class=HTMLResponse)
def index():
    built = os.path.join(REACT_DIR, "index.html")
    if os.path.exists(built):
        return FileResponse(built)
    # Falling back rather than failing. A checkout without `npm run build` still
    # has a working interface, which matters for a project whose examiner may
    # never run npm at all.
    legacy = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(legacy):
        return FileResponse(legacy)
    return HTMLResponse(
        "<h1>No interface built</h1><p>Run <code>npm install &amp;&amp; npm run "
        "build</code> in <code>code/frontend</code>, or use the legacy page.</p>",
        status_code=404)


@app.get("/legacy", response_class=HTMLResponse)
def legacy_index():
    """
    The no-build-step interface, DELIBERATELY FROZEN at the four annotation
    actions it was written for.

    It does not have upload, job progress, the piste confirmation step, reprocess,
    who-scored or lunge proposals, and it will not be given them. Two reasons.
    It exists so that a checkout with Python and nothing else still has a working
    review interface, which matters for an examiner who may never run npm, and
    that guarantee is worth more than feature parity. And it is the comparison the
    evaluation makes: the same four actions, the same API, one interface with a
    build step and one without.

    Keeping it current would mean maintaining every feature twice, which is how
    the two would quietly diverge in behaviour rather than in scope.
    """
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>UI not built</h1>", status_code=404)
    return FileResponse(path)
