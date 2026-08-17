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

RESULTS_DIRS = [os.path.join(PROTOTYPE_DIR, d) for d in
                ("results_after", "results_stabilised", "results_fixedscale",
                 "results_ablation", "results")]
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
    return {
        "bout_id": bout_id,
        "proposed": proposed,
        "added": data["added_touches"],
        "unreliable_segments": data["unreliable_segments"],
        "reanchors": data["reanchors"],
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
    from in_play import out_of_play_windows, in_play_mask, scope_cumulative, scope_distance
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

    dist = scope_distance(d, mask)
    scoped = {
        "in_play_fraction": round(float(mask.mean()), 3) if len(t) else 0.0,
        "mean_distance_m": round(float(dist.mean()), 2) if len(dist) else None,
        "distance_samples": int(len(dist)),
        "fencer_1": {
            "push_m": round(scope_cumulative([float(r["f1_advance_m"]) for r in rows], t, mask), 2),
            "pull_m": round(scope_cumulative([float(r["f1_retreat_m"]) for r in rows], t, mask), 2),
        },
        "fencer_2": {
            "push_m": round(scope_cumulative([float(r["f2_advance_m"]) for r in rows], t, mask), 2),
            "pull_m": round(scope_cumulative([float(r["f2_retreat_m"]) for r in rows], t, mask), 2),
        },
    }
    return {
        "bout_id": bout_id,
        "confirmed_touches": confirmed,
        "whole_recording": whole,
        "in_play": scoped,
        "scoping_basis": ("confirmed touches" if confirmed else
                          "none: no touches confirmed yet, so in-play equals the "
                          "whole recording"),
    }


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
