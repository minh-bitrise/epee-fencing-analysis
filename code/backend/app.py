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

import os
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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

# results_fixed is listed first because it is the only output produced since the
# raw position columns were added, and those columns are what the reliable
# movement metrics are derived from. Without it the interface can only reach CSVs
# that force the fallback derivation, which inherits the noise floor, the
# movement cap and the banking buffer, and which disagreed with the position
# route by 3.5 m of net displacement on the club clip. The older directories stay
# discoverable so the before/after comparisons in the evaluation remain openable,
# and the interface reports which route each bout used rather than hiding it.
RESULTS_DIRS = [os.path.join(PROTOTYPE_DIR, d) for d in
                ("results_pose", "results_fixed", "results_after",
                 "results_stabilised", "results_fixedscale", "results_ablation",
                 "results")]
ANNOTATION_ROOT = os.path.join(PROTOTYPE_DIR, "annotations")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Epee Bout Analysis", version="0.1.0")
store = AnnotationStore(ANNOTATION_ROOT)


def _bouts():
    return discover_bouts(RESULTS_DIRS)


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

@app.get("/api/bouts")
def list_bouts():
    """Processed bouts available for review, with review progress for each."""
    out = []
    for bout_id, b in sorted(_bouts().items()):
        proposed = load_proposed_touches(b.touches_csv)
        out.append({
            "bout_id": bout_id,
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
    whole = compute_stats(rows)

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


@app.get("/api/bouts/{bout_id}/summary")
def get_summary(bout_id: str):
    b = _get_bout(bout_id)
    if not b.summary_md:
        raise HTTPException(404, "no generated summary for this bout")
    with open(b.summary_md) as f:
        return {"bout_id": bout_id, "markdown": f.read()}


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


# --- static UI ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>UI not built</h1>", status_code=404)
    return FileResponse(path)
