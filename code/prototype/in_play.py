"""
Epee Fencing Bout Analysis - In-Play Segment Scoping
=====================================================
Restricts derived metrics to periods of actual fencing, excluding the resets
that follow each touch.

WHY THIS MATTERS. Every metric the pipeline produces currently spans the whole
recording, so a fencer's push and pull totals include the walk back to the
guard line after each touch, and average distance is diluted by the seconds
spent standing at en garde waiting for "allez". The preliminary report
identified this as the single largest gap between the prototype and the
designed system. It could not be closed before touch events existed.

HOW THE RESET DURATION WAS DERIVED. Not guessed. Aligning the distance signal
on the fourteen hand-labelled touches of clip 3 and averaging gives a clear
profile:

    -4.0s to -1.0s   ~2.5 m    phrase in progress
    +0.0s to +0.5s   1.77 -> 1.40 m   the touch itself
    +2.0s to +3.0s   ~3.4 m    peak separation, the referee's halt
    +4.0s to +7.0s   2.95 -> 2.37 m   walking back, coming en garde
    +7.5s onward     ~2.6 m    next phrase under way

The out-of-play window therefore runs from the touch to roughly six seconds
after it. DEFAULT_RESET_S is set from that measurement, and the window is
truncated when the next touch arrives sooner, which happens: observed
touch-to-touch gaps on clip 3 range from 5 to 20 seconds.

WHAT THIS IS NOT. The reset duration is a population average applied
uniformly. Real resets vary with the referee, with whether a point is
disputed, and with equipment problems. In the designed system the user
confirms segment boundaries directly, and this heuristic exists to give them
a sensible starting point rather than to be correct on its own.

Usage:
    python3 in_play.py --csv results_after/fencing_clip3_distance.csv \
                       --touches ground_truth/fencing_clip3_touches.csv
"""

import argparse
import csv

import numpy as np

# Measured from the aligned distance profile described above: separation peaks
# 2-3 s after a touch and returns to the phrase baseline by about 7 s.
DEFAULT_RESET_S = 6.0


# --- segment computation ------------------------------------------------

def out_of_play_windows(touch_times, end_time, reset_s=DEFAULT_RESET_S):
    """
    Windows to exclude, one per touch, as a list of (start, end) in seconds.

    Each window runs from the touch for reset_s, truncated by the next touch
    (bouts can produce touches five seconds apart) and by the end of the
    recording. Returns windows sorted by start time; touch_times need not be
    sorted on input.
    """
    ts = sorted(float(x) for x in touch_times)
    out = []
    for i, touch in enumerate(ts):
        stop = touch + reset_s
        if i + 1 < len(ts):
            stop = min(stop, ts[i + 1])
        stop = min(stop, end_time)
        if stop > touch:
            out.append((touch, stop))
    return out


def in_play_mask(t, windows):
    """Boolean array, True where the frame time falls outside every window."""
    t = np.asarray(t, dtype=float)
    mask = np.ones(len(t), dtype=bool)
    for start, stop in windows:
        mask &= ~((t >= start) & (t < stop))
    return mask


def in_play_fraction(t, windows):
    """Share of frames classed as in play, in [0, 1]."""
    if len(t) == 0:
        return 0.0
    return float(in_play_mask(t, windows).mean())


# --- metric rescoping ---------------------------------------------------

def scope_distance(d, mask):
    """Distance samples inside in-play frames, with gaps dropped."""
    d = np.asarray(d, dtype=float)
    sel = d[mask]
    return sel[~np.isnan(sel)]


def scope_cumulative(series, t, mask):
    """
    Re-total a cumulative series over in-play frames only.

    The CSV stores push and pull as running totals, so the in-play total is
    not a slice of it: differencing first and summing only the in-play deltas
    is required. A delta is attributed to a frame if that frame is in play,
    which drops movement occurring during a reset instead of carrying it over.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < 2:
        return 0.0
    deltas = np.diff(series, prepend=series[0])
    return float(deltas[mask].sum())


def load_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_touch_times(path):
    """
    Read touch timestamps from either a ground-truth file or a detector
    output. Comment lines are skipped so hand-labelled files can carry notes.
    """
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [float(r["time_s"]) for r in rows]


def touch_provenance(path):
    """
    Say whether a touch file holds human-confirmed touches or detector output.

    Inferred from the columns rather than the filename: a hand-labelled file
    carries `scorer` and `annulled`, whereas the detector emits `confidence`
    and `signals`. Filenames are unreliable for this, and it matters because
    the summary stage must tell the reader whether a touch count is confirmed
    or approximate.
    """
    with open(path) as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        cols = set(reader.fieldnames or [])
    if {"scorer"} & cols:
        return "human_confirmed"
    if {"confidence", "signals"} & cols:
        return "automatic_detector"
    return "unknown"


def scope_metrics(rows, touch_times, reset_s=DEFAULT_RESET_S):
    """
    Recompute the headline metrics over in-play frames only, returning both
    the scoped values and the whole-recording values for comparison.
    """
    t = np.array([float(r["time_s"]) for r in rows])
    d = np.array([float(r["distance_raw_m"]) if r["distance_raw_m"] else np.nan
                  for r in rows])
    end_time = float(t[-1]) if len(t) else 0.0

    windows = out_of_play_windows(touch_times, end_time, reset_s)
    mask = in_play_mask(t, windows)

    def summarise(sel_mask):
        dist = scope_distance(d, sel_mask)
        return {
            "frames": int(sel_mask.sum()),
            "distance_samples": int(len(dist)),
            "mean_distance_m": round(float(dist.mean()), 2) if len(dist) else None,
            "f1_push_m": round(scope_cumulative(
                [float(r["f1_advance_m"]) for r in rows], t, sel_mask), 2),
            "f1_pull_m": round(scope_cumulative(
                [float(r["f1_retreat_m"]) for r in rows], t, sel_mask), 2),
            "f2_push_m": round(scope_cumulative(
                [float(r["f2_advance_m"]) for r in rows], t, sel_mask), 2),
            "f2_pull_m": round(scope_cumulative(
                [float(r["f2_retreat_m"]) for r in rows], t, sel_mask), 2),
        }

    return {
        "touches": len(touch_times),
        "reset_s": reset_s,
        "out_of_play_windows": len(windows),
        "in_play_fraction": round(float(mask.mean()), 3),
        "in_play": summarise(mask),
        "whole_recording": summarise(np.ones(len(t), dtype=bool)),
    }


# --- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Scope bout metrics to in-play segments using touch events")
    p.add_argument("--csv", required=True, help="Per-frame metrics CSV")
    p.add_argument("--touches", required=True,
                   help="Touch times: ground truth or detector output")
    p.add_argument("--reset", type=float, default=DEFAULT_RESET_S,
                   help=f"Seconds of reset excluded after each touch "
                        f"(default {DEFAULT_RESET_S}, measured)")
    args = p.parse_args()

    rows = load_rows(args.csv)
    touches = load_touch_times(args.touches)
    r = scope_metrics(rows, touches, reset_s=args.reset)

    print(f"{r['touches']} touches, {r['out_of_play_windows']} reset windows "
          f"of up to {r['reset_s']:.1f}s")
    print(f"in play: {100 * r['in_play_fraction']:.1f}% of frames\n")
    ip, wr = r["in_play"], r["whole_recording"]
    print(f"{'metric':<22}{'whole recording':>18}{'in play only':>16}{'change':>10}")
    for key, label in [("mean_distance_m", "mean distance (m)"),
                       ("f1_push_m", "Fencer 1 push (m)"),
                       ("f1_pull_m", "Fencer 1 pull (m)"),
                       ("f2_push_m", "Fencer 2 push (m)"),
                       ("f2_pull_m", "Fencer 2 pull (m)")]:
        a, b = wr[key], ip[key]
        if a is None or b is None:
            continue
        pct = f"{100 * (b - a) / a:+.0f}%" if a else "n/a"
        print(f"{label:<22}{a:>18.2f}{b:>16.2f}{pct:>10}")


if __name__ == "__main__":
    main()
