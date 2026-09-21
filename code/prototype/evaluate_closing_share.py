"""
Put a confidence interval on closing share, and measure what size of tendency it
can detect on three minutes of footage.

The reasoning behind the metric is sound and says nothing about resolution: the
values cluster near 50 per cent, which is also what a metric dominated by noise
looks like. A block bootstrap is used because frames are strongly autocorrelated,
so resampling them independently would understate the interval.
"""

import argparse
import csv
import sys

import numpy as np

import in_play

# How long a direction of travel persists, in seconds. A fencer's step, lunge or
# retreat is committed for roughly this long, which is the unit that resampling
# has to preserve. Converted to frames per clip from its own sample spacing, so
# it means the same span of play on a 29 fps clip and a 60 fps one.
BLOCK_S = 0.33

BOOTSTRAP_N = 2000
MOVING_EPS = 1e-9


def closing_share(signed):
    """Share of moving frames spent closing, in [0, 1], or None."""
    moving = np.abs(signed) > MOVING_EPS
    if not moving.any():
        return None
    return float((signed[moving] > 0).mean())


def block_interval(signed, block_frames, n=BOOTSTRAP_N, rng=None):
    """A 95 per cent interval for closing share, resampling contiguous blocks.

    Returns (low, high). The interval is mildly anti-conservative: measured
    against series with no bias at all, it excludes 50 per cent about 11 per
    cent of the time rather than the nominal 5.
    """
    rng = rng or np.random.default_rng(0)
    moving = signed[np.abs(signed) > MOVING_EPS]
    if len(moving) < 2 * block_frames:
        return None, None
    n_blocks = max(1, len(moving) // block_frames)
    shares = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(moving) - block_frames, n_blocks)
        sample = np.concatenate([moving[s:s + block_frames] for s in starts])
        shares[i] = (sample > 0).mean()
    lo, hi = np.percentile(shares, [2.5, 97.5])
    return float(lo), float(hi)


def block_frames_for(rows):
    """BLOCK_S in frames, from the series' own sample spacing."""
    times = [float(r["time_s"]) for r in rows if r.get("time_s")]
    if len(times) < 3:
        return 10
    dt = float(np.median(np.diff(times)))
    return max(3, int(round(BLOCK_S / dt))) if dt > 0 else 10


def assess(csv_path, rng=None):
    """Closing share with an interval, for both fencers of one bout."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    block = block_frames_for(rows)
    out = {}
    for fencer in ("f1", "f2"):
        signed = in_play.signed_movement(rows, fencer)
        share = closing_share(signed)
        lo, hi = block_interval(signed, block, rng=rng)
        out[fencer] = {
            "share": share, "low": lo, "high": hi, "block_frames": block,
            "distinguishable_from_chance": (
                None if lo is None else not (lo <= 0.5 <= hi)),
        }
    return out


def power(rng=None, reps=120, frames=5000, block=10):
    """
    How often a known bias is detected, by size of bias.

    This is what makes a null result readable. Without it, "the interval includes
    50 per cent" could mean the fencers were balanced or that the method could
    not have seen a difference either way.
    """
    rng = rng or np.random.default_rng(1)
    results = []
    for p in (0.50, 0.52, 0.55, 0.60, 0.65):
        hits = 0
        for _ in range(reps):
            runs = rng.random(frames // block) < p
            signed = np.repeat(np.where(runs, 1.0, -1.0), block)
            signed = signed * rng.uniform(0.001, 0.02, len(signed))
            lo, hi = block_interval(signed, block, n=400, rng=rng)
            if lo is not None and not (lo <= 0.5 <= hi):
                hits += 1
        results.append({"true_share": p, "detected_pct": 100.0 * hits / reps})
    return results


def main():
    p = argparse.ArgumentParser(
        description="Confidence intervals and resolution for closing share")
    p.add_argument("--results", default="results_current",
                   help="directory of *_distance.csv to assess")
    p.add_argument("--power", action="store_true",
                   help="run the power analysis instead")
    args = p.parse_args()

    if args.power:
        print("Detection rate by size of true tendency, 5000 frames each")
        print("(about three minutes). The first row is the FALSE POSITIVE rate.\n")
        print(f"{'true share':<14}{'detected':>10}")
        print("-" * 24)
        for r in power():
            print(f"{r['true_share']:<14.2f}{r['detected_pct']:>9.0f}%")
        return 0

    import glob
    import os
    print(f"{'clip':<20}{'fencer':<8}{'share':>8}{'95% interval':>20}"
          f"{'differs from 50':>17}")
    print("-" * 73)
    for path in sorted(glob.glob(os.path.join(args.results, "*_distance.csv"))):
        name = os.path.basename(path)[: -len("_distance.csv")]
        for fencer, r in assess(path).items():
            if r["share"] is None or r["low"] is None:
                print(f"{name:<20}{fencer:<8}{'no movement recorded':>8}")
                continue
            span = f"{100*r['low']:.1f} to {100*r['high']:.1f}%"
            verdict = "yes" if r["distinguishable_from_chance"] else "NO"
            print(f"{name:<20}{fencer:<8}{100*r['share']:>7.1f}%{span:>20}"
                  f"{verdict:>17}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
