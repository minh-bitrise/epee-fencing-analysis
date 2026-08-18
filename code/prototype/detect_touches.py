"""
Epee Fencing Bout Analysis - Touch Candidate Detection
=======================================================
Proposes probable touch events in a bout for the user to confirm or reject.
This is the "event-segment proposal" stage of the designed pipeline.

WHY THIS PROPOSES RATHER THAN DECIDES.
A scoring machine registers a valid electrical contact; a referee awards a
point. These differ whenever a touch is annulled, for corps-a-corps, for
covering target, or for another non-valid action, and in those cases the hit
occurred, the machine fired, and no point was awarded. The information
separating those outcomes is absent from the video and audio entirely, because
it is a refereeing judgement rather than a physical event. No model can recover
it. This stage therefore cannot be correct in principle, only useful, which is
why it emits ranked candidates for confirmation rather than a verdict. Because
the user confirms, recall matters more than precision: a missed touch is
invisible to them, whereas a false candidate costs one click.

HOW IT WORKS: GEOMETRY, NOT AUDIO.
A touch has a purely geometric signature. The fencers must close to scoring
distance, and afterwards the referee halts the action and they walk back to
their guard lines, so they separate. The detector therefore looks for a local
minimum in inter-fencer distance followed by sustained separation. Both come
from the existing tracking pipeline, so no additional model is required, and
distance is robust to camera panning because a pan shifts both fencers together.

WHY NOT AUDIO (a negative result worth keeping).
The first implementation of this stage detected the scoring-machine buzzer by
band-pass energy, following Mo (2022), who uses audio for fencing analysis. It
scored precision 0.79 and recall 0.79 on the clip it was tuned on, and collapsed
to precision 0.21 on a held-back clip, needing 15 corrections against a manual
baseline of 4. Diagnosis established that the approach cannot be rescued by
retuning:

  - The noise floor differs by an order of magnitude between recordings. A quiet
    club hall gives a touch-to-noise ratio of 114x to 3000x; a broadcast with
    crowd and commentary gives 6x to 32x. Five threshold rules were tried
    (percentile, median multiple, fraction of maximum, median plus MAD) and each
    either floods the noisier clip or finds nothing in it.
  - A percentile threshold flags a fixed fraction of frames rather than a number
    of events, so candidate count follows recording length instead of how much
    happened.
  - Searching every band with the ground truth in hand, the best achievable
    separation between the weakest real touch and strong background was still
    below 1.0 on both clips. Spectral tonality was worse. On the club clip the
    loudest tonal component at each touch sat at a different frequency every
    time, which indicates the buzzer is not reliably present in the recording at
    all and that what was being detected was blade contact and exchange noise.

Geometry alone outperforms audio plus geometry and, unlike it, generalises:

                     audio + geometry        geometry alone
  clip 3 (tuned on)  F1 0.79, 6 corrections  F1 0.86, 4 corrections
  clip 2 (held back) F1 0.35, 15 corrections F1 0.86, 1 correction

The audio path is retained behind --use-audio for footage with a genuinely
audible buzzer, but it is off by default because on the evidence above it does
active harm: it generates candidates that geometry then has to filter out.

Usage:
    python3 detect_touches.py --csv results_after/fencing_clip3_distance.csv
    python3 detect_touches.py --csv ... --video ... --use-audio   # not advised

Outputs (next to the CSV):
    <base>_touches.csv   ranked touch candidates with per-feature evidence
"""

import argparse
import csv
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, stft

# --- geometric detection config -----------------------------------------
#
# Validated on two independently labelled clips (18 touches total), a quiet
# club recording and a broadcast, giving F1 0.86 on both with these values.

# A candidate must be a local distance minimum, closer than the surrounding
# window by at least this margin. Rejects the small oscillations of ordinary
# blade play, which do not represent an attempt to reach the target.
MIN_PROMINENCE_M = 0.4

# Half-width of the window used to establish a local minimum, in frames. About
# one second at 29 fps, which is the timescale of a fencing phrase.
LOCAL_MIN_WIN_FRAMES = 25

# Required separation after the event: mean distance in the window after minus
# mean distance just before. This is the strongest single feature. Measured on
# labelled data, real touches separate by a median +0.47 m afterwards while
# false positives give -0.03 m, because a referee's halt sends the fencers back
# to their guard lines.
MIN_SEPARATION_M = 0.8
SEPARATION_BACK_S = 0.5           # window before the event
SEPARATION_FWD_S = (0.8, 2.5)     # window after, skipping the reaction lag

# Candidates closer together than this are the same event; keep the stronger.
MERGE_GAP_S = 2.0

# Separation at which confidence saturates, for scaling the score into [0, 1].
CONFIDENCE_SAT_M = 3.0

# --- audio config (optional, see the module docstring) -------------------

BUZZER_SEARCH_LO_HZ = 1200
BUZZER_SEARCH_HI_HZ = 5000
BUZZER_BAND_WIDTH_HZ = 500
AUDIO_SR = 22050
ENVELOPE_WIN_S = 0.02
MIN_BUZZER_S = 0.05
EVENT_MERGE_GAP_S = 0.30
AUDIO_SUPPORT_WIN_S = 1.0         # how near an audio event must be to support


# --- motion data --------------------------------------------------------

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


def separation_after(t, d, ts, back=SEPARATION_BACK_S, fwd=SEPARATION_FWD_S):
    """
    Metres by which the fencers separated after ts: mean distance in the window
    after the event minus mean distance just before it.

    Positive values indicate the pair moved apart, which is what a referee's
    halt produces as they return to their guard lines. Returns None when either
    window lacks enough tracked frames to be meaningful.
    """
    before = d[(t > ts - back) & (t <= ts)]
    after = d[(t > ts + fwd[0]) & (t < ts + fwd[1])]
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]
    if len(before) < 3 or len(after) < 5:
        return None
    return float(after.mean() - before.mean())


def min_distance_near(t, d, ts, win=1.0):
    """Closest the fencers came within +/- win of ts, or None."""
    seg = d[(t > ts - win) & (t < ts + win)]
    seg = seg[~np.isnan(seg)]
    return float(seg.min()) if len(seg) else None


def local_minima(t, d, prominence=MIN_PROMINENCE_M, win=LOCAL_MIN_WIN_FRAMES):
    """
    Times at which inter-fencer distance is a prominent local minimum.

    Prominence is required against the maximum on each side rather than the
    mean, so a candidate has to be a genuine approach out of and back into
    wider distance, not a dip inside continuous close play.
    """
    ok = ~np.isnan(d)
    tt, dd = t[ok], d[ok]
    out = []
    for i in range(win, len(dd) - win):
        left = dd[i - win:i].max()
        right = dd[i + 1:i + win + 1].max()
        if dd[i] < left - prominence and dd[i] < right - prominence:
            out.append((float(tt[i]), float(dd[i])))
    return out


# --- optional audio -----------------------------------------------------

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
    Locate the most burst-like narrow band, a candidate for a scoring buzzer.

    Scores each band by peakiness (peak over the band's own quiet level) times
    the absolute peak. Both terms are needed: peakiness alone prefers bands
    holding only faint spectral leakage, because a tiny burst against
    near-silence has a huge ratio, while absolute peak alone prefers whatever is
    loudest, which on crowd noise is not the buzzer.
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
            score = (peak / med) * peak
            if score > best_score:
                best_score, best = score, (lo, hi)
        lo += BUZZER_BAND_WIDTH_HZ // 2
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
    Find candidate buzzer events in a band. See the module docstring for why
    this does not generalise across recording conditions.
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
            "sustained": dur >= MIN_BUZZER_S,
        })
    return events


# --- candidate assembly -------------------------------------------------

def propose_candidates(t, d, audio_events=None,
                       min_prominence=MIN_PROMINENCE_M,
                       min_separation=MIN_SEPARATION_M,
                       merge_gap=MERGE_GAP_S):
    """
    Propose touch candidates from tracked geometry, optionally noting audio
    support. Returns rows sorted by time.
    """
    raw = []
    for ts, dist in local_minima(t, d, prominence=min_prominence):
        sep = separation_after(t, d, ts)
        if sep is None or sep < min_separation:
            continue
        raw.append({"time_s": ts, "min_distance_m": dist, "separation_m": sep})

    # collapse candidates belonging to the same event, keeping the strongest
    raw.sort(key=lambda r: r["time_s"])
    merged = []
    for r in raw:
        if merged and r["time_s"] - merged[-1]["time_s"] < merge_gap:
            if r["separation_m"] > merged[-1]["separation_m"]:
                merged[-1] = r
        else:
            merged.append(r)

    rows = []
    for r in merged:
        audio_support = 0
        if audio_events:
            audio_support = int(any(
                abs(e["time_s"] - r["time_s"]) <= AUDIO_SUPPORT_WIN_S
                for e in audio_events))
        conf = min(r["separation_m"], CONFIDENCE_SAT_M) / CONFIDENCE_SAT_M
        signals = ["approach", "separated"]
        if audio_support:
            signals.append("audio")
        rows.append({
            "time_s": round(r["time_s"], 2),
            "confidence": round(float(conf), 3),
            "min_distance_m": round(r["min_distance_m"], 2),
            "separation_m": round(r["separation_m"], 2),
            "audio_support": audio_support,
            "signals": "+".join(signals),
        })
    return rows


FIELDS = ["time_s", "confidence", "min_distance_m", "separation_m",
          "audio_support", "signals"]


def write_candidates(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# --- main ---------------------------------------------------------------

def run(csv_path, video_path=None, out_path=None, use_audio=False,
        min_separation=MIN_SEPARATION_M, percentile=99.0):
    base = os.path.splitext(csv_path)[0]
    out_path = out_path or f"{base}_touches.csv"

    t, d, _ = load_motion(csv_path)

    audio_events = None
    if use_audio:
        if not video_path:
            raise SystemExit("--use-audio requires --video")
        print("Extracting audio (note: the audio path does not generalise "
              "across recordings, see the module docstring)...")
        sr, x = extract_audio(video_path)
        band = find_buzzer_band(x, sr)
        audio_events = detect_audio_events(x, sr, band, percentile=percentile)
        print(f"  band {band[0]}-{band[1]} Hz, {len(audio_events)} audio events")

    rows = propose_candidates(t, d, audio_events=audio_events,
                              min_separation=min_separation)
    write_candidates(rows, out_path)

    print(f"\n{len(rows)} proposed touches (for user confirmation):")
    print(f"  {'time':>8}{'conf':>7}{'closest':>9}{'separation':>12}  signals")
    for r in rows:
        print(f"  {r['time_s']:>7.1f}s{r['confidence']:>7.2f}"
              f"{r['min_distance_m']:>8.2f}m{r['separation_m']:>11.2f}m"
              f"  {r['signals']}")
    print(f"\n  saved -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="Propose touch candidates from tracked fencer geometry")
    p.add_argument("--csv", required=True,
                   help="Per-frame metrics CSV from run_detection.py")
    p.add_argument("--video", default=None,
                   help="Bout video, only needed with --use-audio")
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--use-audio", action="store_true",
                   help="Additionally note buzzer support. Off by default: on "
                        "the evidence in the module docstring the audio path "
                        "does not generalise and does active harm.")
    p.add_argument("--min-separation", type=float, default=MIN_SEPARATION_M,
                   help=f"Required post-event separation in metres "
                        f"(default {MIN_SEPARATION_M}, validated on two clips)")
    p.add_argument("--percentile", type=float, default=99.0,
                   help="Audio detection threshold percentile, with --use-audio")
    args = p.parse_args()
    run(args.csv, video_path=args.video, out_path=args.output,
        use_audio=args.use_audio, min_separation=args.min_separation,
        percentile=args.percentile)


if __name__ == "__main__":
    main()
