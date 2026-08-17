"""
Unit tests for the touch-candidate detector.

These test the pure signal-processing and scoring behaviour on synthetic
inputs, without decoding a real video. Note the limitation this implies,
which is the same one recorded for the tracker tests: a suite built on
synthetic inputs can show that the implementation matches the
specification, but cannot show that the specification is right for real
footage. The genuine evaluation needs hand-labelled ground truth.

Run with:
    python3 -m pytest test_touches.py -v
"""

import math

import numpy as np
import pytest

from detect_touches import (
    _envelope,
    detect_audio_events,
    find_buzzer_band,
    load_motion,
    min_distance_near,
    neighbour_count,
    score_candidates,
    separation_after,
    was_closing,
    write_candidates,
    FIELDS,
    WEIGHTS,
    TOUCH_MAX_DIST_M,
    SEPARATION_MIN_M,
)


# -------------------- synthetic signal helpers --------------------

def tone_burst(sr, total_s, bursts, freq, amp=0.5, noise=0.02, seed=0):
    """
    Build a signal of broadband noise with narrow-band tone bursts at the
    given (start_s, duration_s) pairs. Stands in for a scoring buzzer over
    crowd noise.
    """
    rng = np.random.default_rng(seed)
    n = int(total_s * sr)
    x = rng.normal(0, noise, n).astype(np.float32)
    t = np.arange(n) / sr
    for start, dur in bursts:
        a, b = int(start * sr), int((start + dur) * sr)
        b = min(b, n)
        x[a:b] += amp * np.sin(2 * np.pi * freq * t[a:b]).astype(np.float32)
    return x


# -------------------- envelope --------------------

class TestEnvelope:
    def test_silence_gives_zero_envelope(self):
        env, win = _envelope(np.zeros(22050), 22050, win_s=0.02)
        assert len(env) > 0
        assert np.allclose(env, 0.0)

    def test_constant_amplitude_gives_constant_envelope(self):
        sr = 22050
        x = np.full(sr, 0.5, dtype=np.float32)
        env, _ = _envelope(x, sr, win_s=0.02)
        assert np.allclose(env, 0.5, atol=1e-6)

    def test_window_length_sets_frame_count(self):
        sr = 1000
        env, win = _envelope(np.zeros(1000), sr, win_s=0.1)
        assert win == 100
        assert len(env) == 10

    def test_signal_shorter_than_one_window(self):
        env, _ = _envelope(np.zeros(10), 22050, win_s=0.02)
        assert len(env) == 0


# -------------------- band calibration --------------------

class TestFindBuzzerBand:
    def test_locates_the_band_containing_a_bursty_tone(self):
        sr = 22050
        freq = 3200
        x = tone_burst(sr, 20.0, [(2.0, 0.3), (8.0, 0.3), (15.0, 0.3)], freq)
        lo, hi = find_buzzer_band(x, sr)
        assert lo <= freq <= hi, f"expected band to contain {freq}, got {lo}-{hi}"

    def test_returns_a_band_within_the_search_range(self):
        sr = 22050
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.05, sr * 5).astype(np.float32)
        lo, hi = find_buzzer_band(x, sr)
        assert lo >= 1200 and hi <= 5000
        assert hi > lo

    def test_prefers_bursty_over_merely_loud(self):
        """
        A continuous loud tone is not a buzzer. Burstiness, not loudness,
        should decide, so a persistent tone should lose to an intermittent one.
        """
        sr = 22050
        dur = 20.0
        # continuous, loud, low band
        cont = tone_burst(sr, dur, [(0.0, dur)], 1500, amp=0.8, seed=2)
        # intermittent, quieter, high band
        burst = tone_burst(sr, dur, [(3.0, 0.25), (9.0, 0.25), (16.0, 0.25)],
                           4200, amp=0.4, noise=0.0, seed=3)
        lo, hi = find_buzzer_band(cont + burst, sr)
        assert lo <= 4200 <= hi, f"expected the bursty band, got {lo}-{hi}"


# -------------------- audio event detection --------------------

class TestDetectAudioEvents:
    def test_finds_the_expected_number_of_bursts(self):
        sr = 22050
        bursts = [(2.0, 0.3), (8.0, 0.3), (15.0, 0.3)]
        x = tone_burst(sr, 20.0, bursts, 3200)
        ev = detect_audio_events(x, sr, (2950, 3450), percentile=98.0)
        assert 3 <= len(ev) <= 5, f"expected ~3 events, got {len(ev)}"

    def test_event_times_match_the_bursts(self):
        sr = 22050
        bursts = [(2.0, 0.3), (8.0, 0.3), (15.0, 0.3)]
        x = tone_burst(sr, 20.0, bursts, 3200)
        ev = detect_audio_events(x, sr, (2950, 3450), percentile=98.0)
        found = sorted(e["time_s"] for e in ev)
        for start, _ in bursts:
            assert any(abs(f - start) < 0.4 for f in found), \
                f"no event near {start}s in {found}"

    def test_sustained_flag_separates_long_from_short(self):
        sr = 22050
        # one long burst (buzzer-like) and one very short click (blade-like)
        x = tone_burst(sr, 20.0, [(3.0, 0.40), (12.0, 0.01)], 3200)
        ev = detect_audio_events(x, sr, (2950, 3450), percentile=98.0)
        assert any(e["sustained"] for e in ev), "long burst should be sustained"

    def test_silence_yields_no_events(self):
        ev = detect_audio_events(np.zeros(22050 * 3), 22050, (2950, 3450))
        assert ev == []

    def test_strength_is_normalised(self):
        sr = 22050
        x = tone_burst(sr, 20.0, [(2.0, 0.3), (8.0, 0.3)], 3200)
        ev = detect_audio_events(x, sr, (2950, 3450), percentile=98.0)
        assert ev
        for e in ev:
            assert 0.0 <= e["audio_strength"] <= 1.0


# -------------------- motion corroboration --------------------

class TestMinDistanceNear:
    def _series(self):
        t = np.arange(0, 10, 0.1)
        d = np.full_like(t, 3.0)
        d[(t > 4.9) & (t < 5.1)] = 1.2      # a brief close moment at t=5
        return t, d

    def test_finds_the_local_minimum(self):
        t, d = self._series()
        assert math.isclose(min_distance_near(t, d, 5.0, win=0.6), 1.2)

    def test_returns_none_outside_the_series(self):
        t, d = self._series()
        assert min_distance_near(t, d, 100.0) is None

    def test_ignores_nan_gaps(self):
        t = np.arange(0, 10, 0.1)
        d = np.full_like(t, np.nan)
        d[50] = 2.0
        assert math.isclose(min_distance_near(t, d, 5.0, win=0.6), 2.0)

    def test_all_nan_window_returns_none(self):
        t = np.arange(0, 10, 0.1)
        d = np.full_like(t, np.nan)
        assert min_distance_near(t, d, 5.0) is None


class TestWasClosing:
    def test_detects_an_approach(self):
        t = np.arange(0, 10, 0.1)
        d = 5.0 - 0.3 * t            # steadily closing
        assert was_closing(t, d, 5.0) is True

    def test_rejects_a_retreat(self):
        t = np.arange(0, 10, 0.1)
        d = 1.0 + 0.3 * t            # steadily opening
        assert was_closing(t, d, 5.0) is False

    def test_insufficient_data_is_not_closing(self):
        t = np.array([4.9, 5.0])
        d = np.array([3.0, 2.0])
        assert was_closing(t, d, 5.0) is False


class TestSeparationAfter:
    """
    After a touch the referee calls halt and the fencers return to their
    guard lines, so they move apart. Blade contact during an exchange leaves
    them at engagement distance. Measured against ground truth on clip 3,
    real touches separated by a median +0.47 m and false positives by
    -0.03 m, which is why this carries the largest weight in the score.
    """

    def _series(self, before_d, after_d):
        t = np.arange(0, 12, 0.05)
        d = np.where(t <= 5.0, before_d, after_d).astype(float)
        return t, d

    def test_detects_a_reset(self):
        t, d = self._series(before_d=1.8, after_d=3.0)
        sep = separation_after(t, d, 5.0)
        assert sep is not None and sep > SEPARATION_MIN_M

    def test_no_change_during_an_exchange(self):
        t, d = self._series(before_d=2.2, after_d=2.2)
        sep = separation_after(t, d, 5.0)
        assert sep is not None and abs(sep) < 0.05

    def test_closing_further_gives_a_negative_value(self):
        t, d = self._series(before_d=3.0, after_d=1.5)
        assert separation_after(t, d, 5.0) < 0

    def test_returns_none_without_enough_data_after(self):
        t = np.arange(0, 5.2, 0.05)
        d = np.full_like(t, 2.0)
        assert separation_after(t, d, 5.0) is None

    def test_returns_none_without_enough_data_before(self):
        t = np.arange(4.95, 12, 0.05)
        d = np.full_like(t, 2.0)
        assert separation_after(t, d, 5.0) is None

    def test_ignores_nan_frames(self):
        t = np.arange(0, 12, 0.05)
        d = np.where(t <= 5.0, 1.8, 3.0).astype(float)
        d[::3] = np.nan               # drop a third of frames
        sep = separation_after(t, d, 5.0)
        assert sep is not None and sep > SEPARATION_MIN_M


class TestNeighbourCount:
    def test_counts_events_within_the_window(self):
        events = [{"time_s": 10.0}, {"time_s": 10.5}, {"time_s": 11.0},
                  {"time_s": 40.0}]
        assert neighbour_count(events, 0, win=3.0) == 2
        assert neighbour_count(events, 3, win=3.0) == 0

    def test_does_not_count_itself(self):
        assert neighbour_count([{"time_s": 5.0}], 0) == 0


# -------------------- scoring --------------------

class TestScoreCandidates:
    def _motion(self, close_at=None, halted_from=None):
        t = np.arange(0, 30, 0.05)
        d = np.full_like(t, 5.0)
        if close_at is not None:
            # approach then a close moment, so both closing and in_distance hold
            m = (t > close_at - 1.0) & (t <= close_at)
            d[m] = np.linspace(4.0, 1.5, m.sum())
            d[(t > close_at) & (t < close_at + 0.3)] = 1.5
        travel = t * 2.0
        if halted_from is not None:
            travel = np.where(t < halted_from, t * 2.0, halted_from * 2.0)
        return t, d, travel

    def test_fully_corroborated_event_scores_high(self):
        ev = [{"time_s": 10.0, "duration_s": 0.3, "audio_strength": 1.0,
               "band_share": 0.9, "sustained": True}]
        t, d, travel = self._motion(close_at=10.0, halted_from=10.0)
        rows = score_candidates(ev, t, d, travel)
        assert rows[0]["confidence"] > 0.9
        assert rows[0]["signals"].count("+") == 4      # all five features

    def test_uncorroborated_event_scores_low(self):
        ev = [{"time_s": 10.0, "duration_s": 0.01, "audio_strength": 0.1,
               "band_share": 0.2, "sustained": False}]
        t, d, travel = self._motion()                  # never close, never halts
        rows = score_candidates(ev, t, d, travel)
        assert rows[0]["confidence"] < 0.3
        assert rows[0]["in_distance"] == 0

    def test_distant_event_fails_the_distance_gate(self):
        """This is what rules out beeps from an adjacent piste."""
        ev = [{"time_s": 10.0, "duration_s": 0.3, "audio_strength": 1.0,
               "band_share": 0.9, "sustained": True}]
        t = np.arange(0, 30, 0.05)
        d = np.full_like(t, TOUCH_MAX_DIST_M + 2.0)    # fencers far apart
        rows = score_candidates(ev, t, d, t * 2.0)
        assert rows[0]["in_distance"] == 0

    def test_clustered_events_lose_the_isolated_flag(self):
        """This is what rules out deliberate weapon testing, which bursts."""
        ev = [{"time_s": 10.0 + i * 0.4, "duration_s": 0.2,
               "audio_strength": 0.8, "band_share": 0.8, "sustained": True}
              for i in range(5)]
        t, d, travel = self._motion(close_at=10.0)
        rows = score_candidates(ev, t, d, travel)
        assert all(r["isolated"] == 0 for r in rows)

    def test_output_is_sorted_by_confidence(self):
        ev = [
            {"time_s": 5.0, "duration_s": 0.01, "audio_strength": 0.1,
             "band_share": 0.1, "sustained": False},
            {"time_s": 10.0, "duration_s": 0.3, "audio_strength": 1.0,
             "band_share": 0.9, "sustained": True},
        ]
        t, d, travel = self._motion(close_at=10.0, halted_from=10.0)
        rows = score_candidates(ev, t, d, travel)
        confs = [r["confidence"] for r in rows]
        assert confs == sorted(confs, reverse=True)

    def test_weights_sum_to_one(self):
        """Keeps the confidence score interpretable as a 0-1 value."""
        assert math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-9)

    def test_empty_input_gives_empty_output(self):
        t, d, travel = self._motion()
        assert score_candidates([], t, d, travel) == []


# -------------------- output --------------------

class TestWriteCandidates:
    def test_writes_all_declared_fields(self, tmp_path):
        ev = [{"time_s": 1.0, "duration_s": 0.2, "audio_strength": 0.5,
               "band_share": 0.5, "sustained": True}]
        t = np.arange(0, 5, 0.05)
        d = np.full_like(t, 2.0)
        rows = score_candidates(ev, t, d, t * 2.0)
        out = tmp_path / "touches.csv"
        write_candidates(rows, str(out))
        text = out.read_text()
        header = text.splitlines()[0].split(",")
        assert header == FIELDS
        assert len(text.splitlines()) == 2


# -------------------- round trip against a real CSV schema --------------------

class TestLoadMotion:
    def test_reads_the_run_detection_schema(self, tmp_path):
        csv_path = tmp_path / "m.csv"
        csv_path.write_text(
            "frame,time_s,distance_raw_m,distance_smooth_m,method,"
            "f1_advance_m,f1_retreat_m,f2_advance_m,f2_retreat_m\n"
            "0,0.0,2.0,2.0,pose,1.0,0.5,2.0,1.0\n"
            "1,0.5,,,,1.5,0.5,2.0,1.0\n"
        )
        t, d, travel = load_motion(str(csv_path))
        assert list(t) == [0.0, 0.5]
        assert d[0] == 2.0 and np.isnan(d[1])
        assert travel[0] == 4.5      # 1.0 + 0.5 + 2.0 + 1.0
        assert travel[1] == 5.0      # 1.5 + 0.5 + 2.0 + 1.0
