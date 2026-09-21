"""
Read the artefacts the pipeline produces, and hold the user's annotations
alongside them.

Pipeline output is treated as read-only input and the user's decisions live in a
separate JSON file per bout, so nothing the user does can corrupt the artefacts
and a re-run cannot silently discard their work.
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
    """An id unique among the records currently in `items`.

    Deriving it from len() looks equivalent and is not: removing a record
    lowers the count, so the next insert reuses an id that is still LIVE.
    """
    used = []
    for it in items:
        raw = str(it.get("id", ""))
        if raw.startswith(prefix) and raw[len(prefix):].isdigit():
            used.append(int(raw[len(prefix):]))
    return f"{prefix}{max(used) + 1 if used else 0}"


class AnnotationStore:
    """Per-bout annotation state, persisted as JSON.

    Kept deliberately simple: one file per bout, rewritten on each change.
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
                    "unreliable_segments": [], "reanchors": [], "lunges": [],
                    "sessions": [], "green_is": None}
        with open(p) as f:
            data = json.load(f)
        # tolerate files written by an earlier version
        data.setdefault("touch_states", {})
        data.setdefault("added_touches", [])
        data.setdefault("unreliable_segments", [])
        data.setdefault("reanchors", [])
        data.setdefault("lunges", [])
        data.setdefault("sessions", [])
        data.setdefault("green_is", None)
        return data

    def save(self, bout_id, data):
        with open(self._path(bout_id), "w") as f:
            json.dump(data, f, indent=2)
        return data

    def set_green_is(self, bout_id, side):
        """Remember which side the green lamp belongs to on this bout.

        It is a fact about the recording, not about a session: nothing in the
        image reveals it, it never changes for a given bout, and every
        attributed touch depends on it.
        """
        if side not in ("left", "right"):
            raise ValueError(f"green_is must be left or right, got {side!r}")
        data = self.load(bout_id)
        data["green_is"] = side
        return self.save(bout_id, data)

    # --- action 1: confirm or correct a proposed touch ------------------

    def set_touch_state(self, bout_id, touch_id, state, scorer=None, time_s=None):
        """Accept, reject, or correct a proposed touch.

        Correction is folded into the same action rather than being separate,
        because in practice a user adjusting a timestamp or assigning a scorer
        is confirming the touch at the same time.
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
        # A touch before the recording starts is not a touch. It is accepted
        # silently otherwise, sorts to the front of every list, and scopes the
        # first in-play window to a negative span.
        if float(time_s) < 0:
            raise ValueError("time_s cannot be negative")
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

    # --- review sessions: the effort measurement ------------------------

    VALID_MODES = ("assisted", "manual")

    def add_session(self, bout_id, mode, elapsed_s, decisions,
                    touches_after=0, lunges_after=0, note=""):
        """Record how long one pass over a bout took, and in which mode.

        The project's central claim is that confirming proposals costs less
        effort than labelling from scratch, and there is currently no
        measurement of effort anywhere in it.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        if float(elapsed_s) <= 0:
            raise ValueError("elapsed_s must be positive")
        if int(decisions) < 0:
            raise ValueError("decisions cannot be negative")
        data = self.load(bout_id)
        new_id = _next_id(data["sessions"], "s")
        data["sessions"].append({
            "id": new_id,
            "mode": mode,
            "elapsed_s": round(float(elapsed_s), 1),
            "decisions": int(decisions),
            # Seconds per decision is the comparable figure, since the two modes
            # do not produce the same NUMBER of decisions: assisted review answers
            # one question per proposal, manual logging creates one entry per
            # touch the user finds. Dividing here rather than in the interface
            # keeps every consumer using the same definition.
            "seconds_per_decision": (round(float(elapsed_s) / int(decisions), 1)
                                     if int(decisions) else None),
            "touches_after": int(touches_after),
            "lunges_after": int(lunges_after),
            "note": note,
            "created": time.time(),
        })
        self.save(bout_id, data)
        return new_id

    def session_comparison(self, bout_id):
        """
        The two modes side by side, or an explanation of what is still missing.

        Returns a dict rather than a bare number because a single mean would hide
        the thing that decides whether the comparison means anything: how many
        runs it rests on. One run each is an anecdote and should read as one.
        """
        sessions = self.load(bout_id)["sessions"]
        out = {"sessions": sessions}
        for mode in self.VALID_MODES:
            runs = [s for s in sessions if s["mode"] == mode]
            rated = [s["seconds_per_decision"] for s in runs
                     if s["seconds_per_decision"] is not None]
            out[mode] = {
                "runs": len(runs),
                "total_decisions": sum(s["decisions"] for s in runs),
                "mean_seconds_per_decision": (round(sum(rated) / len(rated), 1)
                                              if rated else None),
            }
        a, m = out["assisted"], out["manual"]
        if a["runs"] and m["runs"]:
            if a["mean_seconds_per_decision"] and m["mean_seconds_per_decision"]:
                out["speedup"] = round(m["mean_seconds_per_decision"]
                                       / a["mean_seconds_per_decision"], 2)
            # Said plainly rather than left for the reader to notice. A ratio
            # from one run each is an anecdote, and it will be read as a result
            # unless it is labelled as not being one.
            out["strength"] = ("one run in each mode: an illustration, not a "
                               "measurement" if a["runs"] == 1 and m["runs"] == 1
                               else f"{a['runs']} assisted and {m['runs']} manual runs")
        else:
            missing = [k for k in self.VALID_MODES if not out[k]["runs"]]
            out["speedup"] = None
            out["strength"] = ("no comparison yet: nothing recorded in "
                               + " or ".join(missing) + " mode")
        return out

    # --- action 3: mark a segment tracking-unreliable -------------------

    def add_unreliable_segment(self, bout_id, start_s, end_s, reason=""):
        """
        Exclude a time range from aggregate metrics while preserving the raw
        per-frame data, which is why this is recorded here rather than by
        deleting rows from the pipeline's CSV.
        """
        if float(start_s) < 0:
            raise ValueError("start_s cannot be negative")
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
        """Record that at time_s the tracker had fencer `slot` wrong, and that the
        correct position is (x, y).

        Unlike the other three actions this cannot take effect immediately: it
        changes tracking rather than interpretation, so the pipeline must be
        rerun for it to matter.
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

    def mark_reanchors_applied(self, bout_id):
        """Record that the pipeline has been rerun with the corrections in force.

        This is the user asserting a reprocess happened, not the system
        observing one, and the distinction is deliberate.
        """
        data = self.load(bout_id)
        changed = 0
        for a in data["reanchors"]:
            if not a["applied"]:
                a["applied"] = True
                changed += 1
        self.save(bout_id, data)
        return changed

    # --- lunge labelling (evaluation only, not one of the four actions) ---

    def add_lunge(self, bout_id, time_s, slot, note=""):
        """Record the peak of one lunge: the moment of maximum extension.

        TODO B1h originally called for start and peak, but the hypothesis under
        test only needs the peak: does a pose-derived stance feature reach an
        extreme when a fencer is at full extension?
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
