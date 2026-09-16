"""Tests for evaluate_pose.py.

The point of this module is a comparison, so most of what can go wrong is a
comparison that is not fair: frames where only one estimator is defined, a rank
statistic pointing the wrong way, or an interval built by resampling the wrong
unit. Each of those has a test here.
"""

import csv
import math
import os

import numpy as np
import pytest

import evaluate_pose as ep


def write_csv(path, rows):
    cols = ["frame", "time_s", "distance_raw_m", "distance_smooth_m",
            "distance_bbox_m", "method"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def row(i, t, raw, bbox, method="pose"):
    return {"frame": i, "time_s": t, "distance_raw_m": raw,
            "distance_smooth_m": raw, "distance_bbox_m": bbox, "method": method}


# --- pairing -----------------------------------------------------------

def test_fallback_frames_are_excluded(tmp_path):
    """A bbox frame holds the same number in both columns by construction.

    Including those frames would pull every measured difference towards zero
    with frames that cannot differ, which understates and overstates nothing in
    particular but makes the figure meaningless.
    """
    p = tmp_path / "fencing_clip3_distance.csv"
    write_csv(p, [row(0, 0.0, 2.0, 2.0, "bbox"),
                  row(1, 0.1, 2.5, 2.0, "pose")])
    t, pose, bbox = ep.paired_series(ep.read_rows(p))
    assert t.size == 1
    assert pose[0] == 2.5 and bbox[0] == 2.0


def test_missing_bbox_column_yields_nothing(tmp_path):
    """Results from before the column existed must produce no comparison.

    Silently returning an empty comparison is right; silently comparing against
    a default would be a fabricated result.
    """
    p = tmp_path / "old_distance.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "distance_raw_m", "method"])
        w.writeheader()
        w.writerow({"frame": 0, "time_s": 0.0, "distance_raw_m": 2.0, "method": "pose"})
    t, _, _ = ep.paired_series(ep.read_rows(p))
    assert t.size == 0


# --- labels ------------------------------------------------------------

def test_annulled_touches_are_not_labels(tmp_path):
    """An annulled hit registered on the apparatus but scored no point."""
    p = tmp_path / "gt.csv"
    p.write_text("# comment\ntime_s,scorer,annulled,notes\n"
                 "10,left,0,\"\"\n20,right,1,\"annulled\"\n")
    assert ep.read_labels(p) == [10.0]


def test_comment_lines_are_skipped(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("# one\n# two\ntime_s,scorer,annulled,notes\n5,left,0,\"\"\n")
    assert ep.read_labels(p) == [5.0]


# --- windows -----------------------------------------------------------

def test_window_without_frames_is_dropped():
    """An interval the tracker lost both fencers over is an absence.

    Scoring it would credit whichever estimator happened to be undefined there.
    """
    t = np.array([0.0, 0.5, 8.0])
    y, pm, bm = ep.windows(t, np.array([2.0, 2.0, 2.0]),
                           np.array([2.0, 2.0, 2.0]), [])
    assert y.size == 2   # 0-2 s and 8-10 s; the four windows between are empty


def test_label_tolerance_reaches_the_neighbouring_window():
    """Labels are recorded to the nearest second, so a touch at 2.0 s may have
    happened at 1.6 s. Both adjacent windows count as touch windows."""
    t = np.arange(0.0, 6.0, 0.5)
    d = np.full(t.size, 2.0)
    y, _, _ = ep.windows(t, d, d, [2.0])
    assert y[0] and y[1]


def test_window_score_is_the_minimum():
    t = np.array([0.0, 0.5, 1.0])
    y, pm, bm = ep.windows(t, np.array([3.0, 0.4, 2.0]),
                           np.array([3.0, 1.1, 2.0]), [])
    assert pm[0] == pytest.approx(0.4)
    assert bm[0] == pytest.approx(1.1)


# --- the rank statistic ------------------------------------------------

def test_auc_treats_lower_as_positive():
    """Distance is the score and a touch is the CLOSE case, so the statistic
    has to be oriented the other way round from the usual convention. A version
    of this that pointed the standard way scored a perfect detector at 0."""
    y = np.array([True, True, False, False])
    perfect = np.array([0.1, 0.2, 3.0, 4.0])
    assert ep.auc(perfect, y) == pytest.approx(1.0)
    assert ep.auc(-perfect, y) == pytest.approx(0.0)


def test_auc_ties_score_a_half():
    y = np.array([True, False])
    assert ep.auc(np.array([1.0, 1.0]), y) == pytest.approx(0.5)


def test_auc_undefined_without_both_classes():
    y = np.array([True, True])
    assert ep.auc(np.array([1.0, 2.0]), y) is None


def test_auc_matches_the_pairwise_definition():
    """Checked against the definition it implements, counting pairs directly,
    because the rank shortcut is where an off-by-one hides."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.random(12) < 0.4
        if y.all() or not y.any():
            continue
        s = np.round(rng.random(12), 1)   # rounding forces ties
        wins = sum((sp < sn) + 0.5 * (sp == sn)
                   for sp in s[y] for sn in s[~y])
        assert ep.auc(s, y) == pytest.approx(wins / (y.sum() * (~y).sum()))


# --- the interval ------------------------------------------------------

def test_identical_estimators_give_an_interval_containing_zero():
    rng = np.random.default_rng(1)
    y = rng.random(60) < 0.2
    s = rng.random(60)
    ci = ep.paired_auc_ci([(y, s, s.copy())], n=300)
    assert ci is not None and ci[0] <= 0 <= ci[1]
    assert ci == pytest.approx((0.0, 0.0), abs=1e-9)


def test_a_clearly_better_estimator_gives_a_positive_interval():
    y = np.array([True] * 15 + [False] * 45)
    good = np.where(y, 0.3, 3.0)
    # Deliberately NOT linspace: the positives are the first 15 entries, so an
    # ascending score would rank them lowest and be a perfect detector too.
    bad = np.tile([0.5, 2.0, 1.0, 2.5], 15)   # unrelated to the labels
    ci = ep.paired_auc_ci([(y, good, bad)], n=300)
    assert ci is not None and ci[0] > 0


def test_clips_are_held_fixed_not_resampled():
    """Four clips cannot support a resample over clips. If they were being
    resampled, a run where one clip's windows are all negative would sometimes
    drop every positive and return a degenerate interval."""
    y1 = np.array([True] * 5 + [False] * 15)
    y2 = np.zeros(20, bool)
    s1, s2 = np.where(y1, 0.2, 2.0), np.full(20, 2.0)
    ci = ep.paired_auc_ci([(y1, s1, s1.copy()), (y2, s2, s2.copy())], n=200)
    assert ci is not None


# --- end to end --------------------------------------------------------

def test_analyse_skips_clips_without_ground_truth(tmp_path):
    """A results directory holds whatever was last run through it. Only clips
    with hand labels can be part of a measurement against labels."""
    d = tmp_path / "res"
    d.mkdir()
    write_csv(d / "some_other_clip_distance.csv", [row(0, 0.0, 2.0, 2.1)])
    res = ep.analyse(str(d), gt_dir=str(tmp_path))
    assert res["clips"] == []


def test_analyse_reports_per_clip_and_pooled(tmp_path):
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "fencing_clip3_touches.csv").write_text(
        "time_s,scorer,annulled,notes\n4,left,0,\"\"\n")
    d = tmp_path / "res"
    d.mkdir()
    rows = []
    for i in range(40):
        t = i * 0.25
        close = abs(t - 4.0) < 1.0
        rows.append(row(i, t, 0.4 if close else 2.5, 0.5 if close else 2.6))
    write_csv(d / "fencing_clip3_distance.csv", rows)
    res = ep.analyse(str(d), gt_dir=str(tmp_path))
    c = res["clips"][0]
    assert c["clip"] == "fencing_clip3"
    assert c["n_paired"] == 40
    assert c["median_diff_m"] == pytest.approx(-0.1)
    assert res["pooled"]["auc_pose"] == pytest.approx(1.0)
    assert res["pooled"]["auc_bbox"] == pytest.approx(1.0)
    assert res["pooled"]["auc_diff"] == pytest.approx(0.0)


def test_main_explains_an_empty_directory(tmp_path, capsys):
    assert ep.main(["--results", str(tmp_path), "--gt-dir", str(tmp_path)]) == 1
    assert "distance_bbox_m" in capsys.readouterr().out
