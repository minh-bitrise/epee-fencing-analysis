"""
Epee Fencing Bout Analysis - Annotation Store
==============================================
Reads the artefacts the processing pipeline produces and holds the user's
annotations alongside them.

DESIGN NOTE: why annotations are stored separately from pipeline output.
The design chapter requires that every pipeline output be persisted as editable
data rather than baked into the video, so the annotation interface can read it
back for confirmation. This module keeps that separation strict. Pipeline
artefacts (the per-frame metrics CSV, the proposed touches, the generated
summary) are treated as read-only inputs, and the user's decisions live in a
separate JSON file per bout. Nothing the user does destroys a pipeline result,
so reprocessing a bout never loses their work and their work never has to be
reconstructed from a modified artefact.

The four annotation actions the design specifies are each represented here:
confirming or correcting a proposed touch, adding a missed touch, marking a
segment as tracking-unreliable, and re-anchoring a fencer. The first three are
resolved entirely within this store. Re-anchoring is recorded but cannot take
effect until the pipeline is rerun, since it changes tracking rather than
interpretation, and that distinction is made explicit rather than hidden.
"""

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict

# Annotation state for a proposed touch.
PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
VALID_STATES = (PENDING, CONFIRMED, REJECTED)

# Who a touch is attributed to. The detector never supplies this: it reports
# when a touch happened, never who scored, so attribution always originates
# with the user.
SCORERS = ("left", "right", "double", "unknown")


@dataclass
class Bout:
    """A processed bout and everything known about it."""
    bout_id: str
    metrics_csv: str
    touches_csv: str = ""
    summary_md: str = ""
    video: str = ""

    def exists(self):
        return os.path.exists(self.metrics_csv)


def discover_bouts(results_dirs):
    """
    Find processed bouts by looking for the pipeline's `*_distance.csv` outputs.

    Bouts are identified by the CSV's basename so that the same clip processed
    into different output directories (the before/after comparisons the
    evaluation relies on) stay distinguishable rather than silently colliding.
    """
    bouts = {}
    for d in results_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith("_distance.csv"):
                continue
            base = name[: -len("_distance.csv")]
            bout_id = f"{os.path.basename(d.rstrip('/'))}:{base}"
            metrics = os.path.join(d, name)
            touches = os.path.join(d, f"{base}_distance_touches.csv")
            summary = os.path.join(d, f"{base}_distance_summary.md")
            # The annotated render is preferred over the source video: it carries
            # the boxes, labels and distance readout, so a user adjudicating a
            # proposed touch can see what the tracker saw at that moment rather
            # than having to trust it.
            video = os.path.join(d, f"{base}_annotated.mp4")
            bouts[bout_id] = Bout(
                bout_id=bout_id,
                metrics_csv=metrics,
                touches_csv=touches if os.path.exists(touches) else "",
                summary_md=summary if os.path.exists(summary) else "",
                video=video if os.path.exists(video) else "",
            )
    return bouts


def load_proposed_touches(path):
    """Read the detector's proposed touches, or an empty list if absent."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "id": f"p{i}",
            "time_s": float(r["time_s"]),
            "confidence": float(r.get("confidence", 0.0) or 0.0),
            "separation_m": float(r["separation_m"]) if r.get("separation_m") else None,
            "min_distance_m": float(r["min_distance_m"]) if r.get("min_distance_m") else None,
            "signals": r.get("signals", ""),
            "origin": "detector",
        })
    return out


def _next_id(items, prefix):
    """
    An id unique among the records currently in `items`.

    Deriving it from len() looks equivalent and is not: removing a record lowers the
    count, so the next insert reuses an id that is still LIVE. Two records then
    share one id and a delete takes both. Found in real data after a review session
    that used the remove button, which left two `l1` records and a hole where `l32`
    had been.

    One past the highest number present cannot collide with anything present, which
    is the property that prevents that bug. It is deliberately not the stronger
    "never reuse an id at all": an id freed by a removal can come round again. That
    would only matter if a caller held a stale id across a removal, and the
    interface reloads after every mutation, so it does not. Making it stronger means
    persisting a counter and migrating every existing annotation file, which is not
    worth it for a guarantee nothing needs.
    """
    used = []
    for it in items:
        raw = str(it.get("id", ""))
        if raw.startswith(prefix) and raw[len(prefix):].isdigit():
            used.append(int(raw[len(prefix):]))
    return f"{prefix}{max(used) + 1 if used else 0}"


class AnnotationStore:
    """
    Per-bout annotation state, persisted as JSON.

    Kept deliberately simple: one file per bout, rewritten on each change. A
    database is what the design specifies for the full system, but for a
    single-user review tool a file is sufficient and makes the stored state
    trivially inspectable, which matters while the workflow is still being
    evaluated.
    """

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, bout_id):
        safe = bout_id.replace("/", "_").replace(":", "__")
        return os.path.join(self.root, f"{safe}.json")

    def load(self, bout_id):
        p = self._path(bout_id)
        if not os.path.exists(p):
            return {"bout_id": bout_id, "touch_states": {}, "added_touches": [],
                    "unreliable_segments": [], "reanchors": [], "lunges": []}
        with open(p) as f:
            data = json.load(f)
        # tolerate files written by an earlier version
        data.setdefault("touch_states", {})
        data.setdefault("added_touches", [])
        data.setdefault("unreliable_segments", [])
        data.setdefault("reanchors", [])
        data.setdefault("lunges", [])
        return data

    def save(self, bout_id, data):
        with open(self._path(bout_id), "w") as f:
            json.dump(data, f, indent=2)
        return data

    # --- action 1: confirm or correct a proposed touch ------------------

    def set_touch_state(self, bout_id, touch_id, state, scorer=None, time_s=None):
        """
        Accept, reject, or correct a proposed touch.

        Correction is folded into the same action rather than being separate,
        because in practice a user adjusting a timestamp or assigning a scorer is
        confirming the touch at the same time. Requiring two interactions for
        one decision would work against the design's own aim of resolving each
        error in a single step.
        """
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of {VALID_STATES}")
        if scorer is not None and scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}")
        data = self.load(bout_id)
        entry = data["touch_states"].get(touch_id, {})
        entry["state"] = state
        if scorer is not None:
            entry["scorer"] = scorer
        if time_s is not None:
            entry["time_s"] = float(time_s)
        entry["updated"] = time.time()
        data["touch_states"][touch_id] = entry
        return self.save(bout_id, data)

    # --- action 2: add a touch the detector missed ----------------------

    def add_touch(self, bout_id, time_s, scorer="unknown", note=""):
        data = self.load(bout_id)
        if scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}")
        new_id = _next_id(data["added_touches"], "u")
        data["added_touches"].append({
            "id": new_id, "time_s": float(time_s), "scorer": scorer,
            "note": note, "origin": "user", "created": time.time(),
        })
        self.save(bout_id, data)
        return new_id

    def remove_added_touch(self, bout_id, touch_id):
        data = self.load(bout_id)
        before = len(data["added_touches"])
        data["added_touches"] = [t for t in data["added_touches"] if t["id"] != touch_id]
        self.save(bout_id, data)
        return len(data["added_touches"]) < before

    # --- action 3: mark a segment tracking-unreliable -------------------

    def add_unreliable_segment(self, bout_id, start_s, end_s, reason=""):
        """
        Exclude a time range from aggregate metrics while preserving the raw
        per-frame data, which is why this is recorded here rather than by
        deleting rows from the pipeline's CSV.
        """
        if float(end_s) <= float(start_s):
            raise ValueError("end_s must be greater than start_s")
        data = self.load(bout_id)
        data["unreliable_segments"].append({
            "id": _next_id(data["unreliable_segments"], "s"),
            "start_s": float(start_s), "end_s": float(end_s),
            "reason": reason, "created": time.time(),
        })
        return self.save(bout_id, data)

    def remove_unreliable_segment(self, bout_id, segment_id):
        data = self.load(bout_id)
        before = len(data["unreliable_segments"])
        data["unreliable_segments"] = [
            s for s in data["unreliable_segments"] if s["id"] != segment_id]
        self.save(bout_id, data)
        return len(data["unreliable_segments"]) < before

    # --- action 4: re-anchor a fencer -----------------------------------

    def add_reanchor(self, bout_id, time_s, slot, x, y):
        """
        Record that at time_s the tracker had fencer `slot` wrong, and that the
        correct position is (x, y).

        Unlike the other three actions this cannot take effect immediately: it
        changes tracking rather than interpretation, so the pipeline must be
        rerun for it to matter. The record is stored with `applied: False` so the
        interface can show the user that their correction is pending rather than
        implying it has already been honoured.
        """
        if slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        data = self.load(bout_id)
        data["reanchors"].append({
            "id": _next_id(data["reanchors"], "a"),
            "time_s": float(time_s), "slot": int(slot),
            "x": float(x), "y": float(y),
            "applied": False, "created": time.time(),
        })
        return self.save(bout_id, data)

    # --- lunge labelling (evaluation only, not one of the four actions) ---

    def add_lunge(self, bout_id, time_s, slot, note=""):
        """
        Record the peak of one lunge: the moment of maximum extension.

        WHY ONE TIMESTAMP AND NOT TWO. TODO B1h originally called for start and
        peak, but the hypothesis under test only needs the peak: does a
        pose-derived stance feature reach an extreme when a fencer is at full
        extension? A start time would let the rise be measured too, and it doubles
        the labelling cost per lunge, so it is left out until the simpler question
        is answered.

        WHY `scored` IS NOT STORED. It is derivable. A lunge that scored is one
        whose peak sits shortly before an already-labelled touch, and the touch
        labels exist. Asking for it again would add a decision per lunge and
        introduce a second chance to disagree with the touch file.

        These labels exist to evaluate the pose model, not to correct it, so this
        is deliberately NOT presented as a fifth annotation action. The four
        designed actions repair the system's output; this one grades it.
        """
        if slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        if float(time_s) < 0:
            raise ValueError("time_s must not be negative")
        data = self.load(bout_id)
        data["lunges"].append({
            "id": _next_id(data["lunges"], "l"),
            "time_s": float(time_s), "slot": int(slot),
            "note": note, "created": time.time(),
        })
        return self.save(bout_id, data)

    def remove_lunge(self, bout_id, lunge_id):
        data = self.load(bout_id)
        before = len(data["lunges"])
        data["lunges"] = [l for l in data["lunges"] if l["id"] != lunge_id]
        self.save(bout_id, data)
        return len(data["lunges"]) < before

    # --- derived view ---------------------------------------------------

    def confirmed_touch_times(self, bout_id, proposed):
        """
        The user-verified touch list: confirmed proposals at their corrected
        times, plus manually added touches. Pending proposals are excluded,
        because a proposal the user has not looked at is not evidence.
        """
        data = self.load(bout_id)
        times = []
        for p in proposed:
            st = data["touch_states"].get(p["id"], {})
            if st.get("state") == CONFIRMED:
                times.append({
                    "time_s": st.get("time_s", p["time_s"]),
                    "scorer": st.get("scorer", "unknown"),
                    "origin": "detector-confirmed",
                })
        for t in data["added_touches"]:
            times.append({"time_s": t["time_s"], "scorer": t["scorer"],
                          "origin": "user-added"})
        times.sort(key=lambda t: t["time_s"])
        return times

    def review_progress(self, bout_id, proposed):
        """
        How much of the review is done, which is the number the interface should
        surface: a user needs to know whether the metrics they are looking at
        rest on a complete pass or a partial one.
        """
        data = self.load(bout_id)
        states = data["touch_states"]
        reviewed = sum(1 for p in proposed
                       if states.get(p["id"], {}).get("state") in (CONFIRMED, REJECTED))
        return {
            "proposed": len(proposed),
            "reviewed": reviewed,
            "confirmed": sum(1 for p in proposed
                             if states.get(p["id"], {}).get("state") == CONFIRMED),
            "rejected": sum(1 for p in proposed
                            if states.get(p["id"], {}).get("state") == REJECTED),
            "user_added": len(data["added_touches"]),
            "unreliable_segments": len(data["unreliable_segments"]),
            "pending_reanchors": sum(1 for a in data["reanchors"] if not a["applied"]),
            "complete": len(proposed) == reviewed,
        }
