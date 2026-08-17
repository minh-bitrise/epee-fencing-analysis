"""
Epee Fencing Bout Analysis - Touch Candidate Detection
=======================================================
Proposes probable touch events in a bout video for the user to confirm or
reject. This is the "event-segment proposal" stage of the designed pipeline.

DESIGN NOTE - why this proposes rather than decides.
The scoring machine registers a valid electrical contact; the referee awards
a point. These differ whenever a touch is annulled, for corps-a-corps, for
covering target, or for any other non-valid action. In those cases the hit
occurred, the machine fired, and no point was awarded, and the information
that distinguishes those outcomes is absent from the video and audio
entirely - it is a refereeing judgement, not a physical event. No model can
recover it from the recording. This stage therefore cannot be correct in
principle, only useful, which is why its output is a ranked list of
candidates for human confirmation rather than a verdict.

Because the user confirms, recall matters more than precision: a missed
touch is invisible to the user, whereas a false candidate costs one click.
The detector is deliberately permissive and reports a confidence score.

WHY MULTIPLE SIGNALS.
Audio alone fails on three false positives specific to how fencing is
filmed and practised:
  - beeps from adjacent pistes, since several bouts run within earshot at a
    competition;
  - deliberate weapon testing, where a fencer strikes the floor to check the
    blade registers, producing an identical beep;
  - blade contact from parries and beats, which is a metallic transient that
    resembles a short buzzer spectrally.
Each feature below is included because it discriminates one of these. A
feature that rules nothing out does not earn a place.

Usage:
    python3 detect_touches.py --video fencing_clip3.mp4 \
                              --csv results_after/fencing_clip3_distance.csv

Outputs (next to the CSV):
    <base>_touches.csv   ranked touch candidates with per-feature evidence
"""

import argparse
import csv
import math
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, stft

# --- audio config -------------------------------------------------------

# The scoring-machine buzzer is a narrow-band tone. Its exact pitch differs
# between machine models and venues, so the band is calibrated per clip by
# searching this range rather than hardcoding one frequency.
BUZZER_SEARCH_LO_HZ = 1200
BUZZER_SEARCH_HI_HZ = 5000
BUZZER_BAND_WIDTH_HZ = 500

AUDIO_SR = 22050          # plenty for a sub-5 kHz tone
ENVELOPE_WIN_S = 0.02     # 20 ms envelope frames

# A buzzer sounds for a few hundred milliseconds; blade contact is a click.
MIN_BUZZER_S = 0.05
# Events closer together than this are treated as one event.
EVENT_MERGE_GAP_S = 0.30

# --- corroboration config ----------------------------------------------

# How far either side of an audio event to look for supporting evidence.
CORROBORATE_WIN_S = 0.6
# A touch is scored through a lunge, so front-foot separation at the hit is
# roughly 2.0-2.6 m. Allow generous slack: this gates out adjacent-piste
# beeps, which is a coarse discrimination, not a precise one.
TOUCH_MAX_DIST_M = 3.2
# After a touch the referee calls halt and the fencers walk back to their
# guard lines, so they separate. Blade contact mid-exchange leaves them at
# engagement distance. Measured as mean distance in a window after the event
# minus mean distance just before it.
#
# NOTE: this replaced an earlier "did they stop moving" feature built on the
# push/pull totals. That feature was worthless in practice: measured against
# ground truth it separated real touches from false positives by a factor of
# 1.04, i.e. not at all. Two reasons. First, push/pull is inflated by camera
# panning (see the clip 3 discussion in the report), so on hand-held footage
# it never goes quiet even when the fencers do. Second, and more
# importantly, the premise was wrong: "halt" in fencing does not mean the
# fencers become still, it means the phrase ends and they return to their
# lines, which is movement. Distance is also largely immune to panning,
# since both fencers shift together in the frame, so this replacement is
# robust on exactly the footage where the old feature failed.
SEPARATION_BACK_S = 0.5           # window before the event
SEPARATION_FWD_S = (0.8, 2.5)     # window after, skipping the reaction lag
SEPARATION_MIN_M = 0.2            # keeps ~65% of touches, rejects ~82% of FPs
# Weapon testing comes in bursts; an event with this many neighbours within
# CLUSTER_WIN_S is treated as likely testing rather than scoring.
CLUSTER_WIN_S = 3.0
CLUSTER_MAX_NEIGHBOURS = 2


# --- audio --------------------------------------------------------------

def extract_audio(video_path, sr=AUDIO_SR):
    """Decode the video's audio to a mono numpy array via ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr),
           "-vn", tmp.name]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}")
    got_sr, x = wavfile.read(tmp.name)
    os.unlink(tmp.name)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return got_sr, x.astype(np.float32) / 32768.0


def find_buzzer_band(x, sr):
    """
    Locate the scoring buzzer's frequency band for this recording.

    Looks for the band whose energy is most *peaked in time* rather than
    loudest overall: a buzzer is quiet on average but extremely loud in a
    few short bursts, whereas crowd noise and speech are broadband and
    persistent. Kurtosis of the band envelope captures exactly that shape.
    Returns (lo_hz, hi_hz).
    """
    f, _, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536)
    mag = np.abs(Z)
    best, best_score = None, -np.inf
    lo = BUZZER_SEARCH_LO_HZ
    while lo + BUZZER_BAND_WIDTH_HZ <= BUZZER_SEARCH_HI_HZ:
        hi = lo + BUZZER_BAND_WIDTH_HZ
        sel = (f >= lo) & (f < hi)
        if sel.any():
            env = mag[sel].mean(axis=0)
            med = float(np.median(env)) + 1e-12
            peak = float(np.percentile(env, 99.5))
            # Peakiness (peak relative to the band's own quiet level) times
            # the absolute peak. Both terms are needed. Peakiness alone picks
            # bands holding only faint spectral leakage, because a tiny burst
            # against near-silence has a huge ratio; absolute peak alone picks
            # whatever is loudest, which on crowd noise is not the buzzer.
            score = (peak / med) * peak
            if score > best_score:
                best_score, best = score, (lo, hi)
        lo += BUZZER_BAND_WIDTH_HZ // 2          # overlapping search steps
    return best if best is not None else (2900, 3400)


def _envelope(sig, sr, win_s=ENVELOPE_WIN_S):
    """RMS envelope of a signal at win_s resolution."""
    win = max(1, int(win_s * sr))
    n = len(sig) // win
    if n == 0:
        return np.zeros(0), win
    frames = sig[:n * win].reshape(n, win)
    return np.sqrt((frames ** 2).mean(axis=1)), win


def detect_audio_events(x, sr, band, percentile=99.0):
    """
    Find candidate buzzer events.

    Scores each envelope frame by in-band loudness multiplied by the share
    of total energy sitting in that band, so a loud broadband noise such as
    a shout or a clap does not score highly. Returns a list of dicts with
    time, duration and a normalised audio strength in [0, 1].
    """
    lo, hi = band
    sos = butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    banded = sosfiltfilt(sos, x)

    e_band, win = _envelope(banded, sr)
    e_full, _ = _envelope(x, sr)
    if len(e_band) == 0:
        return []
    ratio = e_band / (e_full + 1e-9)
    score = e_band * ratio

    thr = np.percentile(score, percentile)
    hot = np.where(score > thr)[0]
    if len(hot) == 0:
        return []

    merge_frames = max(1, int(EVENT_MERGE_GAP_S / ENVELOPE_WIN_S))
    runs, start, prev = [], hot[0], hot[0]
    for i in hot[1:]:
        if i - prev > merge_frames:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))

    peak = score.max()
    events = []
    for a, b in runs:
        dur = (b - a + 1) * ENVELOPE_WIN_S
        events.append({
            "time_s": round((a * win) / sr, 3),
            "duration_s": round(dur, 3),
            "audio_strength": round(float(score[a:b + 1].max() / peak), 3),
            "band_share": round(float(ratio[a:b + 1].max()), 3),
            "sustained": dur >= MIN_BUZZER_S,
        })
    return events


# --- corroboration from tracked motion ----------------------------------

def load_motion(csv_path):
    """Read the per-frame metrics CSV into parallel numpy arrays."""
    rows = list(csv.DictReader(open(csv_path)))
    t = np.array([float(r["time_s"]) for r in rows])
    d = np.array([float(r["distance_smooth_m"]) if r["distance_smooth_m"]
                  else np.nan for r in rows])
    travel = np.array([
        float(r["f1_advance_m"]) + float(r["f1_retreat_m"]) +
        float(r["f2_advance_m"]) + float(r["f2_retreat_m"])
        for r in rows
    ])
    return t, d, travel


def min_distance_near(t, d, ts, win=CORROBORATE_WIN_S):
    """Closest the fencers came within +/- win of ts, or None."""
    seg = d[(t > ts - win) & (t < ts + win)]
    seg = seg[~np.isnan(seg)]
    return float(seg.min()) if len(seg) else None


def was_closing(t, d, ts, win=CORROBORATE_WIN_S):
    """
    True if distance was decreasing in the run-up to ts. An approach is
    evidence of an attack; ambient noise has no such structure.
    """
    before = d[(t > ts - win) & (t <= ts)]
    before = before[~np.isnan(before)]
    if len(before) < 4:
        return False
    half = len(before) // 2
    return float(before[half:].mean()) < float(before[:half].mean())


def separation_after(t, d, ts, back=SEPARATION_BACK_S, fwd=SEPARATION_FWD_S):
    """
    Metres by which the fencers separated after ts: mean distance in the
    window after the event minus mean distance just before it.

    Positive values indicate the pair moved apart, which is what a referee's
    halt produces as the fencers return to their guard lines. Returns None
    when either window lacks enough tracked frames to be meaningful.
    """
    before = d[(t > ts - back) & (t <= ts)]
    after = d[(t > ts + fwd[0]) & (t < ts + fwd[1])]
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]
    if len(before) < 3 or len(after) < 5:
        return None
    return float(after.mean() - before.mean())


def neighbour_count(events, idx, win=CLUSTER_WIN_S):
    """How many other audio events fall within win seconds of this one."""
    ts = events[idx]["time_s"]
    return sum(1 for j, e in enumerate(events)
               if j != idx and abs(e["time_s"] - ts) <= win)


# --- scoring ------------------------------------------------------------

# Weights are deliberately simple and hand-set rather than learned: there is
# no labelled training data yet, and a transparent linear score is easier to
# justify and to debug than a fitted one. Revisit once ground truth exists.
# Separation carries the largest weight because it is the only feature
# measured to discriminate strongly against ground truth: real touches
# separate by a median +0.47 m afterwards, false positives by -0.03 m.
WEIGHTS = {
    "sustained":   0.20,   # buzzer rather than blade click
    "in_distance": 0.20,   # rules out adjacent-piste beeps
    "closing":     0.10,   # rules out ambient noise
    "separated":   0.40,   # rules out blade contact during an exchange
    "isolated":    0.10,   # rules out weapon-testing bursts
}


def score_candidates(events, t, d, travel=None):
    """
    Attach corroborating evidence and a confidence score to each event.

    `travel` is accepted but unused: it fed the withdrawn halt feature and
    is kept in the signature so existing callers do not break.
    """
    out = []
    for i, e in enumerate(events):
        ts = e["time_s"]
        md = min_distance_near(t, d, ts)
        sep = separation_after(t, d, ts)
        feats = {
            "sustained":   bool(e["sustained"]),
            "in_distance": md is not None and md <= TOUCH_MAX_DIST_M,
            "closing":     was_closing(t, d, ts),
            "separated":   sep is not None and sep >= SEPARATION_MIN_M,
            "isolated":    neighbour_count(events, i) <= CLUSTER_MAX_NEIGHBOURS,
        }
        conf = sum(w for k, w in WEIGHTS.items() if feats[k])
        # audio strength modulates but never dominates: a weak-but-corroborated
        # event should outrank a loud isolated bang.
        conf = 0.8 * conf + 0.2 * e["audio_strength"]
        out.append({
            "time_s": ts,
            "duration_s": e["duration_s"],
            "confidence": round(float(conf), 3),
            "min_distance_m": round(md, 2) if md is not None else "",
            "separation_m": round(sep, 2) if sep is not None else "",
            "audio_strength": e["audio_strength"],
            **{k: int(v) for k, v in feats.items()},
            "signals": "+".join(k for k, v in feats.items() if v) or "none",
        })
    out.sort(key=lambda r: -r["confidence"])
    return out


FIELDS = ["time_s", "duration_s", "confidence", "min_distance_m",
          "separation_m", "audio_strength", "sustained", "in_distance",
          "closing", "separated", "isolated", "signals"]


def write_candidates(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# --- main ---------------------------------------------------------------

# Defaults chosen against hand-labelled ground truth on clip 3 (14 touches):
# precision 0.79, recall 0.79, and 6 user corrections versus 14 manual
# entries. CAVEAT: fitted on a single clip, so these are provisional and need
# a second labelled bout before they can be trusted as general settings.
DEFAULT_PERCENTILE = 95.0
DEFAULT_MIN_CONFIDENCE = 0.70


def run(video_path, csv_path, out_path=None, percentile=DEFAULT_PERCENTILE,
        min_confidence=DEFAULT_MIN_CONFIDENCE):
    base = os.path.splitext(csv_path)[0]
    out_path = out_path or f"{base}_touches.csv"

    print(f"Extracting audio from {video_path}...")
    sr, x = extract_audio(video_path)
    print(f"  {len(x)/sr:.1f}s at {sr} Hz")

    band = find_buzzer_band(x, sr)
    print(f"Calibrated buzzer band: {band[0]}-{band[1]} Hz")

    events = detect_audio_events(x, sr, band, percentile=percentile)
    print(f"  {len(events)} audio events above the {percentile:.1f}th percentile")

    t, d, travel = load_motion(csv_path)
    scored = score_candidates(events, t, d)
    rows = [r for r in scored if r["confidence"] >= min_confidence]
    write_candidates(rows, out_path)

    print(f"  {len(scored)} scored, {len(rows)} above confidence "
          f"{min_confidence:.2f}")
    print(f"\nProposed touches (for user confirmation):")
    print(f"  {'time':>8}{'conf':>7}{'dist':>7}{'sep':>7}  signals")
    for r in sorted(rows, key=lambda r: r["time_s"]):
        print(f"  {r['time_s']:>7.1f}s{r['confidence']:>7.2f}"
              f"{str(r['min_distance_m']):>7}{str(r['separation_m']):>7}  {r['signals']}")
    print(f"\n  saved -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Propose touch candidates from bout audio plus tracked motion")
    p.add_argument("--video", required=True, help="Bout video (for its audio track)")
    p.add_argument("--csv", required=True,
                   help="Per-frame metrics CSV from run_detection.py")
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                   help=f"Audio detection threshold percentile; lower is more "
                        f"permissive (default {DEFAULT_PERCENTILE})")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                   help=f"Drop candidates below this confidence "
                        f"(default {DEFAULT_MIN_CONFIDENCE})")
    args = p.parse_args()
    run(args.video, args.csv, out_path=args.output, percentile=args.percentile,
        min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
