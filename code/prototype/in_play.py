"""
Restrict derived metrics to periods of actual fencing, excluding the resets that
follow each touch.

Without it every metric spans the whole recording, so movement totals include the
walk back to the guard line and average distance is diluted by time spent at en
garde. The reset duration is not guessed: aligning the distance signal on clip
3's fourteen labelled touches and averaging gives the profile the constants here
are read from.
"""

import argparse
import csv

import numpy as np

# Measured from the aligned distance profile described above: separation peaks
# 2-3 s after a touch and returns to the phrase baseline by about 7 s.
DEFAULT_RESET_S = 6.0


# --- segment computation ------------------------------------------------

def out_of_play_windows(touch_times, end_time, reset_s=DEFAULT_RESET_S):
    """Windows to exclude, one per touch, as a list of (start, end) in seconds.

    Each window runs from the touch for reset_s, truncated by the next touch
    (bouts can produce touches five seconds apart) and by the end of the
    recording.
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


def _direction(rows):
    """
    Which way is "toward the opponent" for each fencer, from the first frame in
    which both positions are known. +1 means increasing x.
    """
    for r in rows:
        if r.get("f1_pos_m") and r.get("f2_pos_m"):
            f1, f2 = float(r["f1_pos_m"]), float(r["f2_pos_m"])
            return (1.0, -1.0) if f2 > f1 else (-1.0, 1.0)
    return (1.0, -1.0)


def signed_movement(rows, fencer):
    """Per-frame movement toward the opponent for one fencer, in metres.

    Read from the raw position columns when the CSV has them, and otherwise
    recovered by differencing the cumulative advance and retreat totals.
    """
    key = f"{fencer}_pos_m"
    if rows and key in rows[0]:
        idx = 0 if fencer == "f1" else 1
        sign = _direction(rows)[idx]
        pos = np.array([float(r[key]) if r[key] else np.nan for r in rows])
        # carry the last known position across gaps, so a missing frame
        # contributes no movement rather than a spurious jump
        for i in range(1, len(pos)):
            if np.isnan(pos[i]):
                pos[i] = pos[i - 1]
        if np.isnan(pos[0]):
            first = np.flatnonzero(~np.isnan(pos))
            pos[:first[0]] = pos[first[0]] if len(first) else 0.0
        return np.diff(pos, prepend=pos[0]) * sign

    adv = np.array([float(r[f"{fencer}_advance_m"]) for r in rows])
    ret = np.array([float(r[f"{fencer}_retreat_m"]) for r in rows])
    return (np.diff(adv, prepend=adv[0]) - np.diff(ret, prepend=ret[0]))


def net_forward_movement(rows, fencer, mask):
    """Signed movement toward the opponent summed over the masked frames, in
    metres.

    Two readings, and the distinction matters enough to name carefully.
    """
    return float(signed_movement(rows, fencer)[mask].sum())


# retained under the old name so existing callers keep working
net_displacement = net_forward_movement


def closing_share(rows, fencer, mask):
    """Proportion of moving frames spent reducing the distance, in [0, 1] or None.

    The well-defined way to say who pressed forward more.
    """
    d = signed_movement(rows, fencer)[mask]
    moving = np.abs(d) > 1e-9
    if not moving.any():
        return None
    return float((d[moving] > 0).mean())


def scope_cumulative(series, t, mask):
    """Re-total a cumulative series over in-play frames only.

    The CSV stores push and pull as running totals, so the in-play total is not
    a slice of it: differencing first and summing only the in-play deltas is
    required.
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
    """Say whether a touch file holds human-confirmed touches or detector output.

    Inferred from the columns rather than the filename: a hand-labelled file
    carries `scorer` and `annulled`, whereas the detector emits `confidence`
    and `signals`.
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
        f1_cs = closing_share(rows, "f1", sel_mask)
        f2_cs = closing_share(rows, "f2", sel_mask)
        return {
            "frames": int(sel_mask.sum()),
            "distance_samples": int(len(dist)),
            "mean_distance_m": round(float(dist.mean()), 2) if len(dist) else None,
            # the reliable movement figures; see the module docstrings
            "f1_net_m": round(net_displacement(rows, "f1", sel_mask), 2),
            "f2_net_m": round(net_displacement(rows, "f2", sel_mask), 2),
            "f1_closing_share": round(f1_cs, 3) if f1_cs is not None else None,
            "f2_closing_share": round(f2_cs, 3) if f2_cs is not None else None,
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
    print(f"{'metric':<26}{'whole recording':>18}{'in play only':>16}{'change':>10}")
    for key, label in [("mean_distance_m", "mean distance (m)"),
                       ("f1_net_m", "F1 fwd movement (m)"),
                       ("f2_net_m", "F2 fwd movement (m)"),
                       ("f1_closing_share", "Fencer 1 closing share"),
                       ("f2_closing_share", "Fencer 2 closing share"),
                       ("f1_push_m", "F1 push (m) [indicative]"),
                       ("f1_pull_m", "F1 pull (m) [indicative]"),
                       ("f2_push_m", "F2 push (m) [indicative]"),
                       ("f2_pull_m", "F2 pull (m) [indicative]")]:
        a, b = wr[key], ip[key]
        if a is None or b is None:
            continue
        pct = f"{100 * (b - a) / a:+.0f}%" if a else "n/a"
        print(f"{label:<26}{a:>18.2f}{b:>16.2f}{pct:>10}")


if __name__ == "__main__":
    main()
