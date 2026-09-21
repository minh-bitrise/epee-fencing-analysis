"""Tests for touch_features.py and train_touch.py.

What can go wrong here is not that the code crashes; it is that it produces a
good number for a bad reason. Most of these tests exist to pin one specific way
of manufacturing a result: leaking the test clip into the threshold, labelling
so generously that the task becomes trivial, comparing the baseline under a
different protocol, or averaging per-clip scores so a three-touch clip outvotes
a fourteen-touch one.
"""

import csv

import numpy as np
import pytest

import touch_features as tf
import train_touch as tt


def write_metrics(tmp_path, rows, name="fencing_clip3_distance.csv"):
    cols = ["frame", "time_s", "distance_raw_m", "distance_smooth_m",
            "distance_bbox_m", "method", "f1_stance_m", "f2_stance_m",
            "f1_hip_height_m", "f2_hip_height_m", "f1_pos_m", "f2_pos_m"]
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return str(p)


def bout(touch_times, n=1200, dt=0.1):
    """A synthetic bout: wide distance with a sharp dip at each touch time."""
    rows = []
    for i in range(n):
        t = round(i * dt, 2)
        d = 3.0
        for g in touch_times:
            if abs(t - g) < 0.4:
                d = min(d, 0.5 + 4 * abs(t - g))
        rows.append({"frame": i, "time_s": t, "distance_raw_m": d,
                     "distance_smooth_m": d, "distance_bbox_m": d,
                     "method": "pose", "f1_pos_m": 2.0, "f2_pos_m": 2.0 + d,
                     "f1_stance_m": 0.6, "f2_stance_m": 0.7,
                     "f1_hip_height_m": 0.9, "f2_hip_height_m": 0.9})
    return rows


def truth_file(tmp_path, times, name="gt.csv"):
    p = tmp_path / name
    p.write_text("# synthetic\ntime_s,scorer,annulled,notes\n"
                 + "".join(f"{t},left,0,\"\"\n" for t in times))
    return str(p)


# --- labelling ---------------------------------------------------------

class TestLabelling:
    def test_one_positive_per_touch_not_one_per_nearby_candidate(self):
        """The defect this class exists for. Labelling everything inside the
        two-second tolerance gave 1,326 positives for 27 touches, about fifty
        per touch, and turned "is this the touch" into "is this near a touch"."""
        times = np.arange(0, 20, 0.2)
        y = tf.label(times, [10.0])
        assert (y == 1).sum() == 1
        assert (y == -1).sum() > 1
        assert times[np.argmax(y == 1)] == pytest.approx(10.0)

    def test_near_misses_are_ignored_not_called_negative(self):
        """A candidate half a second from a real touch is genuinely ambiguous.
        Training on it as a negative teaches the model something false."""
        times = np.array([9.5, 10.0, 15.0])
        y = tf.label(times, [10.0])
        assert y[0] == -1 and y[1] == 1 and y[2] == 0

    def test_two_touches_get_two_positives_even_when_close(self):
        times = np.array([10.0, 11.5])
        y = tf.label(times, [10.0, 11.5])
        assert (y == 1).sum() == 2

    def test_a_touch_with_no_candidate_labels_nothing(self):
        """Unreachable touches must not silently borrow a distant candidate."""
        times = np.array([0.0, 50.0])
        y = tf.label(times, [25.0])
        assert (y == 1).sum() == 0

    def test_ceiling_reports_what_is_reachable(self):
        times = np.array([10.0, 50.0])
        assert tf.ceiling(times, [10.0, 25.0]) == pytest.approx(0.5)


# --- features ----------------------------------------------------------

class TestFeatures:
    def test_every_named_feature_is_produced(self, tmp_path):
        m = write_metrics(tmp_path, bout([20.0, 60.0]))
        g = truth_file(tmp_path, [20, 60])
        X, y, times, truth = tf.build(m, g)
        assert X.shape[1] == len(tf.FEATURE_NAMES)
        assert times.size == X.shape[0] == y.size

    def test_a_real_dip_scores_a_positive(self, tmp_path):
        m = write_metrics(tmp_path, bout([20.0]))
        g = truth_file(tmp_path, [20])
        _, y, _, _ = tf.build(m, g)
        assert (y == 1).sum() == 1

    def test_percentile_features_are_scale_free(self):
        """The metre scale is derived per clip, so a model trained on three
        clips and tested on a fourth meets a shifted scale. The percentile
        version of a feature must not move when the whole clip is rescaled."""
        pool = np.array([1.0, 2.0, 3.0, 4.0])
        assert tf._pct(3.0, pool) == tf._pct(30.0, pool * 10)

    def test_separation_before_distinguishes_a_dip_in_close_play(self, tmp_path):
        """A touch is approached from distance; a dip inside continuous close
        play is approached from close range. Without this feature the geometry
        cannot tell them apart."""
        rows = bout([20.0])
        for r in rows:                      # press the whole bout in close
            if 15.0 < r["time_s"] < 19.0:
                r["distance_smooth_m"] = 0.9
        m = write_metrics(tmp_path, rows)
        g = truth_file(tmp_path, [20])
        X, _, _, _ = tf.build(m, g)
        col = X[:, tf.FEATURE_NAMES.index("separation_before_m")]
        assert np.isfinite(col).any()


# --- the protocol ------------------------------------------------------

class TestProtocol:
    def test_merge_keeps_the_strongest_of_a_cluster(self):
        times = np.array([10.0, 10.5, 30.0])
        scores = np.array([0.4, 0.9, 0.7])
        out = tt.merge(times, scores)
        assert len(out) == 2
        assert out[0] == (10.0, 0.4) or out[0] == (10.5, 0.9)
        assert max(s for _, s in out[:1]) == 0.9 or out[0][1] == 0.9

    def test_merge_leaves_well_separated_events_alone(self):
        times = np.array([10.0, 30.0, 50.0])
        scores = np.array([0.5, 0.5, 0.5])
        assert len(tt.merge(times, scores)) == 3

    def test_the_rule_baseline_uses_the_systems_own_constants(self):
        """Re-typing the thresholds here would let the baseline drift away from
        the detector it is supposed to represent."""
        assert tt.RULE_PROMINENCE_M == tf.dt.MIN_PROMINENCE_M
        assert tt.RULE_SEPARATION_M == tf.dt.MIN_SEPARATION_M

    def test_the_rule_rejects_below_its_thresholds(self):
        names = tf.FEATURE_NAMES
        X = np.zeros((2, len(names)))
        X[0, names.index("prominence_m")] = 1.0
        X[0, names.index("separation_after_m")] = 2.0
        X[1, names.index("prominence_m")] = 0.2      # below 0.4
        X[1, names.index("separation_after_m")] = 2.0
        s = tt.rule_scores(X, names)
        assert s[0] > 0 and s[1] < 0

    def test_pooled_is_micro_averaged_not_a_mean_of_f1(self):
        """Clip 3 holds fourteen touches and clip 1 holds three. Averaging their
        F1 scores would let a good result on three offset a bad one on fourteen."""
        rows = [{"tp": 14, "fp": 0, "fn": 0}, {"tp": 0, "fp": 3, "fn": 3}]
        p = tt.pooled(rows)
        assert p["tp"] == 14 and p["fp"] == 3 and p["fn"] == 3
        naive = (1.0 + 0.0) / 2
        assert p["f1"] > naive

    def test_training_split_drops_the_ignore_class(self):
        data = {"a": {"X": np.arange(9.0).reshape(3, 3),
                      "y": np.array([1, -1, 0])}}
        X, y = tt.training_split(data, ["a"])
        assert X.shape[0] == 2
        assert set(y.tolist()) == {0, 1}

    def test_threshold_selection_never_sees_the_test_clip(self, monkeypatch):
        """The protocol's load-bearing claim, asserted rather than trusted.
        Choosing the operating point on the held-out clip is the error the
        audio detector was diagnosed with and it would be invisible in the
        output.

        An earlier version of this test spied on `fit_predict`, which
        pick_threshold stopped calling when the inner loop was rewritten to
        fold over recordings.
        """
        tag = {c: float(i + 1) for i, c in enumerate("abcd")}
        data = {}
        for c, v in tag.items():
            X = np.full((40, len(tf.FEATURE_NAMES)), v)
            data[c] = {"X": X, "y": np.array([1] + [0] * 39),
                       "times": np.arange(40.0) * 3, "truth": [0.0]}

        scored, trained = [], []

        class Spy:
            def fit(self, X, y):
                trained.extend(np.unique(X[:, 0]).tolist())
                return self

            def predict_proba(self, X):
                scored.extend(np.unique(X[:, 0]).tolist())
                return np.column_stack([np.zeros(len(X)), np.linspace(0, 1, len(X))])

        monkeypatch.setattr(tt, "models", lambda: {"logistic": Spy()})
        tt.pick_threshold("logistic", data, ["a", "b", "c"])

        assert tag["d"] not in scored, "the held-out clip was scored"
        assert tag["d"] not in trained, "the held-out clip was trained on"
        assert sorted(set(scored)) == [tag["a"], tag["b"], tag["c"]]

    def test_a_recording_is_never_on_both_sides_of_a_fold(self, monkeypatch):
        """7a and 7b are two windows of one video. Held out separately, a model
        trains on one and is tested on the other, which is testing it on its own
        training data and reports a better number with nothing to show for it."""
        # Three recordings, not two: holding one out must still leave two for
        # the inner fold that picks the threshold, or the code correctly refuses
        # and this tests nothing.
        names = ["fencing_clip", "fencing_clip3", "fencing_clip7a", "fencing_clip7b"]
        data = {}
        for i, c in enumerate(names):
            data[c] = {"X": np.full((30, len(tf.FEATURE_NAMES)), float(i + 1)),
                       "y": np.array([1] + [0] * 29),
                       "times": np.arange(30.0) * 3, "truth": [0.0]}

        pairs = []

        class Spy:
            def fit(self, X, y):
                self.trained = set(np.unique(X[:, 0]).tolist())
                return self

            def predict_proba(self, X):
                pairs.append((self.trained, set(np.unique(X[:, 0]).tolist())))
                return np.column_stack([np.zeros(len(X)), np.linspace(0, 1, len(X))])

        monkeypatch.setattr(tt, "models", lambda: {"logistic": Spy()})
        tt.evaluate(data, "logistic")
        assert pairs
        for trained, scored in pairs:
            assert not (trained & scored), "a clip was trained on and scored"
        # 7a and 7b carry tags 3.0 and 4.0: either both are trained on or
        # neither is, never one while the other is being scored.
        for trained, _ in pairs:
            assert (3.0 in trained) == (4.0 in trained)

    def test_threshold_selection_refuses_a_single_training_clip(self):
        """It cannot be done, and the earlier version did it anyway by falling
        back to a fixed 0.5. That made the one-clip point on the learning curve
        the only one whose threshold was untuned, and it scored HIGHER than two
        clips as a result: the protocol changing, read as the model learning
        less from more data. Refusing is the only safe behaviour."""
        data = {"a": {"X": np.zeros((5, len(tf.FEATURE_NAMES))),
                      "y": np.array([1, 0, 0, 0, 0]),
                      "times": np.arange(5.0), "truth": [0.0]}}
        with pytest.raises(ValueError, match="two training RECORDINGS"):
            tt.pick_threshold("logistic", data, ["a"])
