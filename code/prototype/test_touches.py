"""
Unit tests for the touch-candidate detector.

These exercise the geometric detection path, which is the primary one, plus the
audio helpers retained behind --use-audio. Note the limitation this implies, the
same one recorded for the tracker tests: a suite built on synthetic inputs can
show that an implementation matches its specification, but cannot show that the
specification is right for real footage. The audio path passed its tests
throughout and still failed to generalise across recordings; only hand-labelled
ground truth revealed that.

Run with:
    python3 -m pytest test_touches.py -v
"""

import math

import numpy as np
import pytest

from detect_touches import (
    CONFIDENCE_SAT_M,
    MIN_PROMINENCE_M,
    MIN_SEPARATION_M,
    FIELDS,
    _envelope,
    detect_audio_events,
    find_buzzer_band,
    load_motion,
    local_minima,
    min_distance_near,
    propose_candidates,
    separation_after,
    write_candidates,
)


# -------------------- synthetic signal helpers --------------------

def tone_burst(sr, total_s, bursts, freq, amp=0.5, noise=0.02, seed=0):
    """Broadband noise with narrow-band tone bursts at (start_s, duration_s)."""
    rng = np.random.default_rng(seed)
    n = int(total_s * sr)
    x = rng.normal(0, noise, n).astype(np.float32)
    t = np.arange(n) / sr
    for start, dur in bursts:
        a, b = int(start * sr), min(int((start + dur) * sr), n)
        x[a:b] += amp * np.sin(2 * np.pi * freq * t[a:b]).astype(np.float32)
    return x


def bout_series(events, duration=60.0, fps=29.0, baseline=2.6,
                approach_to=1.4, reset_to=3.4):
    """
    Synthetic distance track. Each event time gets an approach to `approach_to`,
    then a separation to `reset_to`, then a return to baseline - the profile
    measured from real labelled touches.
    """
    t = np.arange(0.0, duration, 1.0 / fps)
    d = np.full_like(t, baseline)
    for ts in events:
        approach = (t > ts - 1.0) & (t <= ts)
        d[approach] = np.linspace(baseline, approach_to, approach.sum())
        at = (t > ts) & (t <= ts + 0.4)
        d[at] = approach_to
        rise = (t > ts + 0.4) & (t <= ts + 2.0)
        d[rise] = np.linspace(approach_to, reset_to, rise.sum())
        hold = (t > ts + 2.0) & (t <= ts + 3.5)
        d[hold] = reset_to
        fall = (t > ts + 3.5) & (t <= ts + 6.0)
        d[fall] = np.linspace(reset_to, baseline, fall.sum())
    return t, d


# -------------------- separation --------------------

class TestSeparationAfter:
    """
    After a touch the referee halts the action and the fencers return to their
    guard lines, so they move apart. Blade contact during an exchange leaves
    them at engagement distance. Measured on labelled data, real touches
    separate by a median +0.47 m and false positives by -0.03 m, which is why
    this is the detector's strongest feature.
    """

    def _step(self, before_d, after_d):
        t = np.arange(0, 12, 0.05)
        return t, np.where(t <= 5.0, before_d, after_d).astype(float)

    def test_detects_a_reset(self):
        t, d = self._step(1.8, 3.0)
        assert separation_after(t, d, 5.0) > MIN_SEPARATION_M

    def test_no_change_during_an_exchange(self):
        t, d = self._step(2.2, 2.2)
        assert abs(separation_after(t, d, 5.0)) < 0.05

    def test_closing_further_gives_a_negative_value(self):
        t, d = self._step(3.0, 1.5)
        assert separation_after(t, d, 5.0) < 0

    def test_none_without_enough_data_after(self):
        t = np.arange(0, 5.2, 0.05)
        assert separation_after(t, np.full_like(t, 2.0), 5.0) is None

    def test_none_without_enough_data_before(self):
        t = np.arange(4.95, 12, 0.05)
        assert separation_after(t, np.full_like(t, 2.0), 5.0) is None

    def test_ignores_nan_frames(self):
        t = np.arange(0, 12, 0.05)
        d = np.where(t <= 5.0, 1.8, 3.0).astype(float)
        d[::3] = np.nan
        assert separation_after(t, d, 5.0) > MIN_SEPARATION_M


# -------------------- local minima --------------------

class TestLocalMinima:
    def test_finds_a_prominent_dip(self):
        t, d = bout_series([20.0], duration=40.0)
        mins = local_minima(t, d)
        assert mins, "expected at least one local minimum"
        assert any(abs(ts - 20.0) < 1.0 for ts, _ in mins)

    def test_ignores_shallow_oscillation(self):
        """Ordinary blade play wobbles; that is not an attempt to reach target."""
        t = np.arange(0.0, 40.0, 1 / 29.0)
        d = 2.6 + 0.1 * np.sin(2 * np.pi * t / 3.0)   # amplitude below prominence
        assert local_minima(d=d, t=t) == []

    def test_prominence_threshold_is_respected(self):
        t, d = bout_series([20.0], duration=40.0, baseline=2.6, approach_to=2.4)
        # a 0.2 m dip is below the 0.4 m default
        assert local_minima(t, d, prominence=MIN_PROMINENCE_M) == []
        assert local_minima(t, d, prominence=0.1) != []

    def test_handles_nan_gaps(self):
        t, d = bout_series([20.0], duration=40.0)
        d = d.copy()
        d[::5] = np.nan
        assert any(abs(ts - 20.0) < 1.5 for ts, _ in local_minima(t, d))

    def test_short_series_yields_nothing(self):
        t = np.arange(0.0, 0.5, 1 / 29.0)
        assert local_minima(t, np.full_like(t, 2.0)) == []


# -------------------- min distance --------------------

class TestMinDistanceNear:
    def _series(self):
        t = np.arange(0, 10, 0.1)
        d = np.full_like(t, 3.0)
        d[(t > 4.9) & (t < 5.1)] = 1.2
        return t, d

    def test_finds_the_local_minimum(self):
        t, d = self._series()
        assert math.isclose(min_distance_near(t, d, 5.0), 1.2)

    def test_returns_none_outside_the_series(self):
        t, d = self._series()
        assert min_distance_near(t, d, 100.0) is None

    def test_ignores_nan(self):
        t = np.arange(0, 10, 0.1)
        d = np.full_like(t, np.nan)
        d[50] = 2.0
        assert math.isclose(min_distance_near(t, d, 5.0), 2.0)


# -------------------- candidate proposal --------------------

class TestProposeCandidates:
    def test_finds_each_synthetic_touch(self):
        events = [10.0, 25.0, 40.0]
        t, d = bout_series(events, duration=60.0)
        rows = propose_candidates(t, d)
        assert len(rows) == len(events)
        for ts in events:
            assert any(abs(r["time_s"] - ts) < 1.0 for r in rows)

    def test_rejects_an_approach_with_no_reset(self):
        """Closing and staying close is an exchange, not a scored touch."""
        t = np.arange(0.0, 40.0, 1 / 29.0)
        d = np.full_like(t, 2.6)
        d[(t > 19.0) & (t < 21.0)] = 1.4       # closes and stays there
        d[t >= 21.0] = 1.4
        assert propose_candidates(t, d) == []

    def test_merges_candidates_from_one_event(self):
        t, d = bout_series([20.0], duration=40.0)
        rows = propose_candidates(t, d)
        assert len(rows) == 1

    def test_confidence_scales_with_separation_and_is_bounded(self):
        t, d = bout_series([20.0], duration=40.0, approach_to=1.4, reset_to=3.4)
        rows = propose_candidates(t, d)
        assert 0.0 < rows[0]["confidence"] <= 1.0

    def test_bigger_separation_scores_higher(self):
        t1, d1 = bout_series([20.0], duration=40.0, reset_to=3.0)
        t2, d2 = bout_series([20.0], duration=40.0, reset_to=4.0)
        c1 = propose_candidates(t1, d1)[0]["confidence"]
        c2 = propose_candidates(t2, d2)[0]["confidence"]
        assert c2 > c1

    def test_confidence_saturates(self):
        t, d = bout_series([20.0], duration=40.0, approach_to=1.0,
                           reset_to=1.0 + CONFIDENCE_SAT_M * 2)
        assert propose_candidates(t, d)[0]["confidence"] == 1.0

    def test_output_is_sorted_by_time(self):
        t, d = bout_series([40.0, 10.0, 25.0], duration=60.0)
        rows = propose_candidates(t, d)
        assert [r["time_s"] for r in rows] == sorted(r["time_s"] for r in rows)

    def test_raising_min_separation_rejects_weak_events(self):
        # a clear approach (so the local minimum is found) but only a small
        # reset afterwards, which is the signature of an exchange that did not
        # end in a halt
        t, d = bout_series([20.0], duration=40.0, approach_to=1.4, reset_to=2.2)
        assert propose_candidates(t, d, min_separation=0.2) != []
        assert propose_candidates(t, d, min_separation=1.5) == []

    def test_flat_track_yields_nothing(self):
        t = np.arange(0.0, 40.0, 1 / 29.0)
        assert propose_candidates(t, np.full_like(t, 2.6)) == []

    def test_signals_name_the_geometric_evidence(self):
        t, d = bout_series([20.0], duration=40.0)
        rows = propose_candidates(t, d)
        assert "approach" in rows[0]["signals"]
        assert "separated" in rows[0]["signals"]
        assert rows[0]["audio_support"] == 0

    def test_audio_support_is_noted_when_supplied(self):
        t, d = bout_series([20.0], duration=40.0)
        rows = propose_candidates(t, d, audio_events=[{"time_s": 20.2}])
        assert rows[0]["audio_support"] == 1
        assert "audio" in rows[0]["signals"]

    def test_distant_audio_does_not_count_as_support(self):
        t, d = bout_series([20.0], duration=40.0)
        rows = propose_candidates(t, d, audio_events=[{"time_s": 35.0}])
        assert rows[0]["audio_support"] == 0


# -------------------- output --------------------

class TestWriteCandidates:
    def test_writes_all_declared_fields(self, tmp_path):
        t, d = bout_series([20.0], duration=40.0)
        rows = propose_candidates(t, d)
        out = tmp_path / "touches.csv"
        write_candidates(rows, str(out))
        lines = out.read_text().splitlines()
        assert lines[0].split(",") == FIELDS
        assert len(lines) == len(rows) + 1


# -------------------- motion loading --------------------

class TestLoadMotion:
    def test_reads_the_run_detection_schema(self, tmp_path):
        p = tmp_path / "m.csv"
        p.write_text(
            "frame,time_s,distance_raw_m,distance_smooth_m,method,"
            "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m\n"
            "0,0.0,2.0,2.0,pose,1.0,0.5,2.0,1.0\n"
            "1,0.5,,,,1.5,0.5,2.0,1.0\n"
        )
        t, d, travel = load_motion(str(p))
        assert list(t) == [0.0, 0.5]
        assert d[0] == 2.0 and np.isnan(d[1])
        assert travel[0] == 4.5 and travel[1] == 5.0


# -------------------- audio helpers (optional path) --------------------

class TestEnvelope:
    def test_silence_gives_zero(self):
        env, _ = _envelope(np.zeros(22050), 22050)
        assert len(env) > 0 and np.allclose(env, 0.0)

    def test_constant_amplitude(self):
        env, _ = _envelope(np.full(22050, 0.5, dtype=np.float32), 22050)
        assert np.allclose(env, 0.5, atol=1e-6)

    def test_window_sets_frame_count(self):
        env, win = _envelope(np.zeros(1000), 1000, win_s=0.1)
        assert win == 100 and len(env) == 10

    def test_shorter_than_one_window(self):
        env, _ = _envelope(np.zeros(10), 22050)
        assert len(env) == 0


class TestFindBuzzerBand:
    def test_locates_a_bursty_tone(self):
        sr, freq = 22050, 3200
        x = tone_burst(sr, 20.0, [(2.0, 0.3), (8.0, 0.3), (15.0, 0.3)], freq)
        lo, hi = find_buzzer_band(x, sr)
        assert lo <= freq <= hi

    def test_stays_within_the_search_range(self):
        rng = np.random.default_rng(1)
        lo, hi = find_buzzer_band(rng.normal(0, 0.05, 22050 * 5).astype(np.float32), 22050)
        assert lo >= 1200 and hi <= 5000 and hi > lo

    def test_prefers_bursty_over_merely_loud(self):
        """
        Peakiness alone would pick a band holding only faint leakage, because a
        tiny burst against near-silence has a huge ratio. Absolute peak alone
        would pick whatever is loudest. The score needs both terms.
        """
        sr, dur = 22050, 20.0
        cont = tone_burst(sr, dur, [(0.0, dur)], 1500, amp=0.8, seed=2)
        burst = tone_burst(sr, dur, [(3.0, 0.25), (9.0, 0.25), (16.0, 0.25)],
                           4200, amp=0.4, noise=0.0, seed=3)
        lo, hi = find_buzzer_band(cont + burst, sr)
        assert lo <= 4200 <= hi


class TestDetectAudioEvents:
    def test_finds_the_bursts(self):
        sr = 22050
        bursts = [(2.0, 0.3), (8.0, 0.3), (15.0, 0.3)]
        ev = detect_audio_events(tone_burst(sr, 20.0, bursts, 3200), sr,
                                 (2950, 3450), percentile=98.0)
        assert 3 <= len(ev) <= 5
        found = sorted(e["time_s"] for e in ev)
        for start, _ in bursts:
            assert any(abs(f - start) < 0.4 for f in found)

    def test_silence_yields_nothing(self):
        assert detect_audio_events(np.zeros(22050 * 3), 22050, (2950, 3450)) == []

    def test_strength_is_normalised(self):
        sr = 22050
        ev = detect_audio_events(tone_burst(sr, 20.0, [(2.0, 0.3), (8.0, 0.3)], 3200),
                                 sr, (2950, 3450), percentile=98.0)
        assert ev and all(0.0 <= e["audio_strength"] <= 1.0 for e in ev)

    def test_sustained_flag_marks_long_bursts(self):
        sr = 22050
        ev = detect_audio_events(tone_burst(sr, 20.0, [(3.0, 0.40)], 3200), sr,
                                 (2950, 3450), percentile=98.0)
        assert any(e["sustained"] for e in ev)
