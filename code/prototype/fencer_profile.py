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

# The smallest own-position spread measured on real foot-fencing footage is 0.66 m
# (clip 1's Fencer 1); the largest is 2.78 m. Bounding-box jitter alone produces
# far less than either. This sits well below the smallest real value and well
# above jitter.
#
# PROVISIONAL, AND ASYMMETRIC EVIDENCE. It is derived from four clips of foot
# fencing and from NO footage of the case it exists to catch, because none was
# available. It is calibrated from one side of the boundary only, and should be
# re-derived if wheelchair footage is ever obtained.
MIN_FOOTWORK_IQR_M = 0.25

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


def footwork_range(p1, p2):
    """
    How much ground the more mobile of the two fencers covered, as the
    interquartile range of their own position.

    Used to decide whether footwork-based measurement means anything on this
    bout at all. The larger of the two is taken rather than the mean, because one
    fencer holding still while the other moves is ordinary foot fencing, whereas
    NEITHER moving is the case this exists to detect.
    """
    return max(_iqr(p1), _iqr(p2))


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


def score_progression(touches, duration_s):
    """
    How the score moved through the bout: lead changes, and how long each fencer
    spent ahead.

    WHY TIME AND NOT JUST TOUCHES. A 5-4 bout where one fencer led throughout and
    a 5-4 bout that changed hands four times are the same scoreline and different
    bouts, and the second is the one a coach wants to talk about. Time leading is
    the cheap way to tell them apart.

    Doubles advance both scores, so they can end a lead without either fencer
    scoring past the other. That is correct epee behaviour and not a special
    case: at 4-4 a double is 5-5.

    Everything here is derived from the touch list the user confirmed. Nothing is
    inferred from tracking, so this section is unaffected by the swap check that
    withholds the per-fencer axes.
    """
    if not touches:
        return {"available": False,
                "reason": "no confirmed touches to build a scoreline from"}

    ordered = sorted(touches, key=lambda t: t["time_s"])
    left = right = 0
    leader = None
    # The last fencer to have HELD the lead, which is not the same as the current
    # leader. A lead almost always changes hands by passing through level, so
    # comparing only against the current leader counts no change at all: the
    # sequence is 1, None, 2, and neither step is a swap between two fencers.
    # Measured on clip 3 this reported zero lead changes for a bout where one
    # fencer led for 87 seconds and the other for 38.
    last_holder = None
    changes = 0
    leading_s = {1: 0.0, 2: 0.0}
    prev_t = 0.0
    timeline = []

    for t in ordered:
        # Credit the stretch that just ended to whoever was ahead during it,
        # before the score changes.
        if leader:
            leading_s[leader] += t["time_s"] - prev_t
        prev_t = t["time_s"]

        scorer = t.get("scorer")
        if scorer == "left":
            left += 1
        elif scorer == "right":
            right += 1
        elif scorer == "double":
            left += 1
            right += 1
        # An unattributed touch advances neither score. It is not a zero-zero
        # event, it is an unknown one, and guessing would put a fabricated
        # scoreline in front of the user.

        new_leader = 1 if left > right else (2 if right > left else None)
        if new_leader is not None:
            if last_holder is not None and new_leader != last_holder:
                changes += 1
            last_holder = new_leader
        leader = new_leader
        timeline.append({"time_s": t["time_s"], "left": left, "right": right,
                         "scorer": scorer})

    if leader:
        leading_s[leader] += max(0.0, duration_s - prev_t)

    total = sum(leading_s.values())
    return {
        "available": True,
        "final": {"fencer_1": left, "fencer_2": right},
        "lead_changes": changes,
        "time_leading_s": {1: round(leading_s[1], 1), 2: round(leading_s[2], 1)},
        "time_leading_pct": {
            1: round(100.0 * leading_s[1] / total, 1) if total else None,
            2: round(100.0 * leading_s[2] / total, 1) if total else None,
        },
        "level_s": round(max(0.0, duration_s - total), 1),
        "unattributed": sum(1 for t in ordered
                            if t.get("scorer") not in ("left", "right", "double")),
        "timeline": timeline,
    }


# Thirds rather than the five zones a manual logger would offer. The metre scale
# is derived per clip and the fencers never reach both ends of a fourteen-metre
# piste in a clip this length, so the boundaries between five bands would be
# finer than the measurement behind them.
ZONE_NAMES = ("their own third", "the middle", "the far third")


def piste_zones(rows, touches):
    """
    Where along the strip each fencer's touches were scored.

    Uses the observed extent of the bout rather than an assumed piste length, for
    the same reason `territory` does: the scale is derived per clip and the
    fencers do not visit both ends.
    """
    pairs = [(float(r["f1_pos_m"]), float(r["f2_pos_m"])) for r in rows
             if r.get("f1_pos_m") and r.get("f2_pos_m")]
    if not pairs or not touches:
        return {"available": False,
                "reason": "no tracked positions or no confirmed touches"}

    lo = min(min(a, b) for a, b in pairs)
    hi = max(max(a, b) for a, b in pairs)
    span = hi - lo
    if span <= 0:
        return {"available": False, "reason": "the fencers never moved apart"}

    counts = {1: [0, 0, 0], 2: [0, 0, 0]}
    for t in touches:
        side = t.get("scorer")
        slot = 1 if side == "left" else (2 if side == "right" else None)
        if slot is None:
            continue
        pos = position_at(rows, t["time_s"], slot)
        if pos is None:
            continue
        # Measured from the scoring fencer's OWN end, so "the far third" means
        # the same thing for both of them.
        frac = (pos - lo) / span if slot == 1 else (hi - pos) / span
        counts[slot][min(2, max(0, int(frac * 3)))] += 1

    return {"available": True, "zones": ZONE_NAMES,
            "fencer_1": counts[1], "fencer_2": counts[2]}


def position_at(rows, time_s, slot):
    """One fencer's position at the frame nearest a touch, or None."""
    col = "f1_pos_m" if slot == 1 else "f2_pos_m"
    best, best_gap = None, TOUCH_MATCH_S
    for r in rows:
        if not r.get(col):
            continue
        gap = abs(float(r["time_s"]) - time_s)
        if gap <= best_gap:
            best, best_gap = float(r[col]), gap
    return best


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

    duration_for_score = float(rows[-1]["time_s"]) if rows else 0.0
    # The scoreline is derived entirely from the touches the user confirmed, so
    # it survives a tracking failure that withholds every per-fencer axis. A bout
    # the tracker could not follow still has a score, and refusing to show it
    # would be withholding something this system did not get wrong.
    scoreline = score_progression(touches, duration_for_score)

    swaps, left_share, _ = count_side_swaps(rows)
    if swaps > 0:
        return {
            "available": False,
            "score": scoreline,
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
        return {"available": False, "score": scoreline,
                "reason": "too few frames with both fencers tracked to profile"}

    # WHY THIS REFUSES RATHER THAN REPORTING NEAR-ZERO EVERYTHING. Every axis
    # here, and the touch and lunge detectors upstream of it, assume fencing done
    # on the feet: distance opens and closes because the fencers move, a touch is
    # a local minimum in that distance, a lunge is an ankle separation. Wheelchair
    # fencing is fenced from frames bolted to the floor, so none of that holds.
    #
    # Pointed at such a bout the system does not fail visibly, which is worse
    # than failing. It returns a distance series that barely moves, proposes no
    # touches, and draws a profile whose axes all sit at parity, and every one of
    # those outputs looks like a legitimate reading of a cagey bout. Producing
    # confident nonsense for a population the system was never designed for is a
    # more serious exclusion than declining to answer.
    #
    # The same check catches two failures that are not about Para fencing at all:
    # footage framed so tightly that the piste collapses in the measured space,
    # and a tracker that has locked onto two people who are not fencing.
    footwork = footwork_range(p1, p2)
    if footwork < MIN_FOOTWORK_IQR_M:
        return {
            "available": False,
            "score": scoreline,
            "footwork_iqr_m": round(footwork, 2),
            "reason": (
                f"neither fencer's position varies by more than "
                f"{footwork:.2f} m across this bout, against {MIN_FOOTWORK_IQR_M} m "
                "of movement that footwork-based measurement needs. Every figure "
                "here assumes fencing done on the feet, so on wheelchair fencing, "
                "on footage framed too tightly to show the strip, or where the "
                "tracker has locked onto the wrong people, these axes would read "
                "as a cagey bout rather than as a measurement that does not "
                "apply."),
        }

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
        "score": scoreline,
        "zones": piste_zones(rows, touches),
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
