"""
Epee Fencing Bout Analysis - Tempo and Exchange Metrics
=======================================================
Derives the timing metrics the introduction promises from a bout's touch list.

WHY THIS COULD NOT EXIST BEFORE. Chapter 1 argues the project's value rests on
metrics that cannot be measured by eye, and lists tempo among them: time between
touches, the duration of individual exchanges, and how a fencer's behaviour
changes as a bout progresses. Every one of those needs to know when touches
happen. Until touch detection existed the pipeline could produce distance and
displacement but nothing temporal, which is why the generated summaries were
accurate and thin.

WHAT AN EXCHANGE IS TAKEN TO BE. A phrase ends at a touch. It begins when play
resumes after the previous reset, which is estimated as the touch time plus the
reset duration measured in in_play.py from the distance profile aligned on
labelled touches. The first exchange begins at the start of the recording. This
is an approximation of the referee's "allez", which is not observable in the
data, and the approximation is stated rather than hidden: an exchange duration
here is the interval available for fencing, not necessarily the interval spent
fencing.

WHAT IS DELIBERATELY NOT DERIVED. Nothing here attributes a touch to a fencer.
The detector reports when a touch occurred and never who scored, and testing
showed geometry cannot supply that: in epee there is no right-of-way, so a
counter-attack scores as readily as the attack, and "whichever fencer advanced
more" predicts the scorer on 53 per cent of single-scorer touches against a 50
per cent chance baseline. Per-fencer tempo metrics therefore appear only when the
user has supplied scorers.

Usage:
    python3 tempo.py --csv results_after/fencing_clip3_distance.csv \
                     --touches ground_truth/fencing_clip3_touches.csv
"""

import argparse
import csv
import statistics

from in_play import DEFAULT_RESET_S, load_touch_times


def load_touches_with_scorer(path):
    """
    Read touches, keeping the scorer where one is present.

    Detector output has no scorer column, so those rows come back with None and
    every per-scorer metric is then omitted rather than guessed.
    """
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    out = []
    for r in rows:
        out.append({
            "time_s": float(r["time_s"]),
            "scorer": r.get("scorer") or None,
        })
    out.sort(key=lambda t: t["time_s"])
    return out


def exchanges(touches, duration_s, reset_s=DEFAULT_RESET_S):
    """
    Split a bout into exchanges, one per touch.

    Each exchange runs from when play resumed after the previous touch to the
    touch that ends it. Returns a list of dicts with start, end and duration.
    """
    out = []
    prev_end = 0.0
    for t in touches:
        start = min(prev_end, t["time_s"])
        if t["time_s"] <= start:
            # a touch inside the previous reset window; record a zero-length
            # exchange rather than a negative one, and flag it
            out.append({"start_s": round(start, 2), "end_s": round(t["time_s"], 2),
                        "duration_s": 0.0, "scorer": t["scorer"], "overlapping": True})
        else:
            out.append({"start_s": round(start, 2), "end_s": round(t["time_s"], 2),
                        "duration_s": round(t["time_s"] - start, 2),
                        "scorer": t["scorer"], "overlapping": False})
        prev_end = min(t["time_s"] + reset_s, duration_s)
    return out


def _spread(values):
    if not values:
        return {}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
    }


def compute_tempo(touches, duration_s, reset_s=DEFAULT_RESET_S):
    """
    Tempo statistics for a bout. Returns a dict suitable for the LLM payload.
    """
    if not touches:
        return {"touches": 0, "note": "no touches supplied, so no tempo metrics"}

    times = [t["time_s"] for t in touches]
    gaps = [b - a for a, b in zip(times, times[1:])]
    ex = exchanges(touches, duration_s, reset_s)
    durations = [e["duration_s"] for e in ex if not e["overlapping"]]

    # Does the bout speed up or slow down? Compared as first against second half
    # by touch count rather than by time, so an uneven touch distribution does
    # not make one window meaninglessly sparse.
    half = len(gaps) // 2
    first, second = gaps[:half], gaps[half:]
    trend = None
    if first and second:
        d = statistics.mean(second) - statistics.mean(first)
        trend = {
            "first_half_mean_gap_s": round(statistics.mean(first), 2),
            "second_half_mean_gap_s": round(statistics.mean(second), 2),
            "change_s": round(d, 2),
            "direction": "slowing" if d > 0.5 else ("quickening" if d < -0.5 else "steady"),
        }

    result = {
        "touches": len(touches),
        "duration_s": round(duration_s, 1),
        "touches_per_minute": round(60.0 * len(touches) / duration_s, 2) if duration_s else None,
        "time_between_touches_s": _spread(gaps),
        "exchange_duration_s": _spread(durations),
        "reset_assumed_s": reset_s,
        "tempo_trend": trend,
        "exchanges": ex,
    }

    # Per-scorer figures only when the user has supplied scorers, since the
    # detector cannot infer them (see the module docstring).
    scorers = [t["scorer"] for t in touches if t["scorer"]]
    if scorers:
        counts = {}
        for s in scorers:
            counts[s] = counts.get(s, 0) + 1
        result["scorer_counts"] = counts
        # longest run of consecutive touches by the same scorer: a simple proxy
        # for momentum that a coach can check against the video
        best_who, best_run, run_who, run = None, 0, None, 0
        for s in scorers:
            if s == run_who:
                run += 1
            else:
                run_who, run = s, 1
            if run > best_run:
                best_who, best_run = run_who, run
        result["longest_streak"] = {"scorer": best_who, "touches": best_run}
    else:
        result["scorer_counts"] = None
        result["note"] = ("touch source has no scorer column, so per-scorer tempo "
                          "metrics are omitted rather than inferred")
    return result


def bout_duration(metrics_csv):
    rows = list(csv.DictReader(open(metrics_csv)))
    return float(rows[-1]["time_s"]) if rows else 0.0


def main():
    p = argparse.ArgumentParser(description="Tempo and exchange metrics for a bout")
    p.add_argument("--csv", required=True, help="Per-frame metrics CSV")
    p.add_argument("--touches", required=True,
                   help="Touch times: ground truth or detector output")
    p.add_argument("--reset", type=float, default=DEFAULT_RESET_S)
    args = p.parse_args()

    duration = bout_duration(args.csv)
    touches = load_touches_with_scorer(args.touches)
    r = compute_tempo(touches, duration, reset_s=args.reset)

    print(f"{r['touches']} touches over {r['duration_s']}s "
          f"({r['touches_per_minute']} per minute)\n")
    for key, label in [("time_between_touches_s", "time between touches (s)"),
                       ("exchange_duration_s", "exchange duration (s)")]:
        s = r.get(key) or {}
        if s:
            print(f"  {label:<28} mean {s['mean']:>5}  median {s['median']:>5}  "
                  f"min {s['min']:>5}  max {s['max']:>5}  std {s['std']:>5}")
    if r.get("tempo_trend"):
        t = r["tempo_trend"]
        print(f"\n  tempo: {t['direction']} "
              f"({t['first_half_mean_gap_s']}s -> {t['second_half_mean_gap_s']}s between touches)")
    if r.get("scorer_counts"):
        print(f"  scorers: {r['scorer_counts']}")
        print(f"  longest streak: {r['longest_streak']['touches']} "
              f"by {r['longest_streak']['scorer']}")
    elif r.get("note"):
        print(f"\n  note: {r['note']}")


if __name__ == "__main__":
    main()
