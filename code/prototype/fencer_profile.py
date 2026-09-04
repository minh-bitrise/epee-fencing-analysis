"""
Epee Fencing Bout Analysis - Fencer Profile
===========================================
A per-fencer summary of how someone fenced in one bout, as a small set of axes
that can be drawn as a radar and read at a glance.

WHY THIS IS SAFE TO BUILD AND THE PUSH / PULL TOTALS WERE NOT. The withdrawn
push / pull figures were ACCUMULATIONS: a sum of per-frame position deltas. The
report's appendix E.2 traces exactly why that failed. A fencer re-acquired at a
new position after a tracking dropout contributes a one-sided step that never
cancels, so clip 3's Fencer 2 accumulated +23.01 m of movement against an
endpoint difference of -0.94 m. No clamp repairs it, because the truncated steps
reach 5.4 m in a single frame.

Every axis here is instead either an INSTANTANEOUS reading averaged over frames
(mean position, interquartile range of position, distance at a touch) or a COUNT
of touches the user has already confirmed. Neither accumulates error: a dropout
displaces a handful of samples out of thousands and is absorbed by the mean,
where the accumulator banked it permanently. That distinction is the whole reason
this module exists rather than reviving the withdrawn metric.

WHY THE AXES COMPARE THE TWO FENCERS RATHER THAN A POPULATION. A radar needs a
scale, and the honest scale is not available: four clips is not a population, and
normalising against them would invent a "typical epeeist" out of eight
fencer-bouts from one club and one broadcast. So each axis is scored as this
fencer's share of the pair's total, where 50 is parity. That answers the question
a coach actually asks - which of these two did more of this - and it is the only
question this much data can support. The raw measured value is carried alongside
every score so nothing is hidden behind the shape.

WHY IT REFUSES TO DRAW ON SWAPPED FOOTAGE. Every axis is per-fencer, so all of
them assume slot identity held. On clip 4 it did not: 14 side swaps, with Fencer
1 on the left in only 26.9 per cent of frames. A profile computed there would
describe the tracker rather than either fencer, and would look entirely
plausible. `count_side_swaps` already detects this exactly, so the profile is
withheld rather than qualified: a radar with a footnote still gets read as a
radar.
"""
import argparse
import csv
import json
import statistics

from generate_summary import count_side_swaps
from in_play import in_play_mask, out_of_play_windows

# A touch is attributed to the position the fencers were at when it was awarded,
# read from the frame nearest that time rather than interpolated. The touch times
# themselves are only labelled to the nearest second, so sub-frame precision here
# would be false precision.
TOUCH_MATCH_S = 0.5


def load_rows(csv_path):
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


def _positions(rows, mask=None):
    """
    Both fencers' positions, in metres, for frames where both were tracked.

    Returns (times, f1_positions, f2_positions), all the same length.
    """
    t, p1, p2 = [], [], []
    for i, r in enumerate(rows):
        if mask is not None and i < len(mask) and not mask[i]:
            continue
        if not r.get("f1_pos_m") or not r.get("f2_pos_m"):
            continue
        t.append(float(r["time_s"]))
        p1.append(float(r["f1_pos_m"]))
        p2.append(float(r["f2_pos_m"]))
    return t, p1, p2


def _iqr(values):
    """
    How much of the strip a fencer used, as the middle 50 per cent of their
    positions. The interquartile range rather than the full range because the
    full range is set by the two most extreme samples in the bout, which is
    precisely where a tracking dropout puts its worst reading.
    """
    if len(values) < 4:
        return 0.0
    s = sorted(values)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return q3 - q1


def territory(p_own, p_other):
    """
    How far up the strip a fencer lived, in metres from their own end.

    Measured from the observed extent of the bout rather than an assumed piste
    length, because the metre scale is derived per clip and the fencers never
    reach both ends of a fourteen-metre piste in a clip of this length. Both
    fencers are measured from their OWN end, so the two numbers are comparable
    and a larger one always means further forward.
    """
    lo = min(min(p_own), min(p_other))
    hi = max(max(p_own), max(p_other))
    own_mean = statistics.mean(p_own)
    # Whoever sits nearer the low end of the strip owns that end.
    return own_mean - lo if own_mean < statistics.mean(p_other) else hi - own_mean


def distance_at(rows, time_s):
    """Inter-fencer distance at the frame nearest a touch, or None if untracked."""
    best, best_gap = None, TOUCH_MATCH_S
    for r in rows:
        if not r.get("distance_smooth_m"):
            continue
        gap = abs(float(r["time_s"]) - time_s)
        if gap <= best_gap:
            best, best_gap = float(r["distance_smooth_m"]), gap
    return best


def longest_streak(scorers, who):
    """
    Longest run of consecutive touches scored by one fencer, or None when there
    are no scored touches at all. Doubles break a run.

    The None matters more than it looks. Returning 0 for a bout with no confirmed
    touches draws a spoke at parity, and two fencers apparently tied on "best
    run" is a finding about them rather than a statement that nothing has been
    reviewed yet. Caught by looking at the panel on a real bout, not by a test:
    every axis was individually correct and the picture still said something
    false.
    """
    if not scorers:
        return None
    best = run = 0
    for s in scorers:
        run = run + 1 if s == who else 0
        best = max(best, run)
    return best


def touch_axes(rows, touches):
    """
    The three axes that come from confirmed touches rather than from tracking.

    Doubles count as half a touch to each fencer, which is what they are: in epee
    both lights mean both scored.
    """
    scorers = [t.get("scorer") for t in touches if t.get("scorer")]
    out = {}
    for slot, side in ((1, "left"), (2, "right")):
        own = [t for t in touches if t.get("scorer") == side]
        doubles = [t for t in touches if t.get("scorer") == "double"]
        credited = len(own) + 0.5 * len(doubles)
        ranges = [d for d in (distance_at(rows, t["time_s"]) for t in own)
                  if d is not None]
        out[slot] = {
            "touches_scored": len(own),
            "doubles": len(doubles),
            "scoring_share_pct": (round(100.0 * credited / len(touches), 1)
                                  if touches else None),
            "scoring_range_m": (round(statistics.mean(ranges), 2)
                                if ranges else None),
            "longest_streak": longest_streak(scorers, side),
        }
    return out


def _share(a, b):
    """
    One fencer's value as a share of the pair's total, on a 0 to 100 scale where
    50 is parity. Returns None when neither fencer has a value, so an axis with
    no data is drawn as absent rather than as a zero, which would read as a
    measured weakness.
    """
    if a is None or b is None:
        return None
    total = a + b
    if total <= 0:
        return 50.0 if a == b else None
    return round(100.0 * a / total, 1)


def build(csv_path, touches, lunges=None, reset_s=None):
    """
    A profile for both fencers, or a refusal explaining why one cannot be drawn.

    `touches` is the user's confirmed list, `lunges` an optional mapping of slot
    to a list of confirmed lunge times.
    """
    rows = load_rows(csv_path)
    if not rows:
        return {"available": False,
                "reason": "the bout has no tracking data to profile"}

    swaps, left_share, _ = count_side_swaps(rows)
    if swaps > 0:
        return {
            "available": False,
            "swaps": swaps,
            "left_share_pct": round(100.0 * left_share, 1) if left_share else None,
            "reason": (
                f"the tracker exchanged the two fencers {swaps} times on this "
                "bout, so every per-fencer figure would describe the tracking "
                "rather than either fencer. Two fencers do not cross on a piste, "
                "so this is a tracking failure and not a fencing event."),
        }

    duration_s = float(rows[-1]["time_s"]) if rows else 0.0
    times = [float(r["time_s"]) for r in rows]
    touch_times = [t["time_s"] for t in touches]
    # Scoped to active fencing where touches are known: walkbacks to the en garde
    # line are not fencing, and they are where a fencer's position is furthest
    # from anything tactical.
    if touch_times:
        windows = out_of_play_windows(touch_times, duration_s,
                                      **({"reset_s": reset_s} if reset_s else {}))
        mask = in_play_mask(times, windows)
        scoping = "active fencing only, resets between touches excluded"
    else:
        mask = None
        scoping = "the whole recording: no confirmed touches to scope resets by"

    _, p1, p2 = _positions(rows, mask)
    if len(p1) < 10:
        return {"available": False,
                "reason": "too few frames with both fencers tracked to profile"}

    ta = touch_axes(rows, touches)
    lunges = lunges or {}
    minutes = duration_s / 60.0 if duration_s else 0.0

    raw = {}
    for slot, own, other in ((1, p1, p2), (2, p2, p1)):
        n_lunges = len(lunges.get(slot) or lunges.get(str(slot)) or [])
        raw[slot] = {
            "territory_m": round(territory(own, other), 2),
            "range_used_m": round(_iqr(own), 2),
            "lunge_rate_per_min": (round(n_lunges / minutes, 2)
                                   if minutes and n_lunges else None),
            **ta[slot],
        }

    # The unit travels with the axis rather than being inferred in the
    # interface: a share and a distance are both two-decimal numbers, and
    # "58.30" beside "1.67" invites reading them on the same scale.
    axes = [
        ("territory_m", "Territory", "m",
         "metres up the strip from their own end"),
        ("range_used_m", "Ground used", "m",
         "middle 50% of their positions, in metres"),
        ("scoring_share_pct", "Scoring", "%",
         "share of touches, doubles counted half"),
        ("scoring_range_m", "Scoring range", "m",
         "mean distance when they scored"),
        ("lunge_rate_per_min", "Lunges", "/min", "confirmed lunges per minute"),
        ("longest_streak", "Best run", "", "longest run of consecutive touches"),
    ]

    return {
        "available": True,
        "scoping": scoping,
        "duration_s": round(duration_s, 1),
        "frames_used": len(p1),
        "mean_distance_m": round(statistics.mean(
            [float(r["distance_smooth_m"]) for r in rows
             if r.get("distance_smooth_m")]), 2),
        "axes": [
            {"key": k, "label": label, "unit": unit, "explains": explains,
             "fencer_1": {"value": raw[1][k], "score": _share(raw[1][k], raw[2][k])},
             "fencer_2": {"value": raw[2][k], "score": _share(raw[2][k], raw[1][k])}}
            for k, label, unit, explains in axes
        ],
        "note": ("Each axis scores one fencer against the other in this bout, "
                 "where 50 is parity. It is not a comparison against other "
                 "fencers: four recordings is not a population to normalise "
                 "against."),
    }


def load_touches(path):
    """Touch times with scorers, from a ground-truth CSV or a review annotation."""
    if path.endswith(".json"):
        data = json.load(open(path))
        rows = data.get("touches") or data.get("proposed") or []
        return [{"time_s": float(r.get("corrected_time_s") or r["time_s"]),
                 "scorer": r.get("scorer")}
                for r in rows if r.get("state") in (None, "confirmed")]
    out = []
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        if r.get("time_s"):
            out.append({"time_s": float(r["time_s"]), "scorer": r.get("scorer")})
    return out


def main():
    p = argparse.ArgumentParser(description="Profile both fencers in one bout")
    p.add_argument("--metrics", required=True, help="the bout's distance CSV")
    p.add_argument("--touches", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    result = build(args.metrics, load_touches(args.touches))
    if not result["available"]:
        print(f"no profile: {result['reason']}")
    else:
        print(f"{'axis':<16}{'fencer 1':>22}{'fencer 2':>22}")
        for a in result["axes"]:
            f1, f2 = a["fencer_1"], a["fencer_2"]
            fmt = lambda x: (f"{x['value']} ({x['score']:.0f})"
                             if x["value"] is not None and x["score"] is not None
                             else "-")
            print(f"{a['label']:<16}{fmt(f1):>22}{fmt(f2):>22}")
        print(f"\nscoped to {result['scoping']}")
        print(result["note"])

    if args.output:
        json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
