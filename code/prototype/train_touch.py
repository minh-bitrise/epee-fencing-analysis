"""
Epee Fencing Bout Analysis - A Learned Touch Proposer, Against the Hand Rule
============================================================================
Train a classifier to make the accept-or-reject decision that two hand-set
constants currently make, and compare the two under one protocol.

WHY THIS EXISTS. `detect_touches.py` accepts a candidate when the distance
minimum has at least 0.4 m of prominence and the fencers then separate by at
least 0.8 m. Both numbers were set against clip 3. The rule works; the project
cannot say whether it works because the sport has that structure or because
those constants suit these four recordings. Putting a trained model against it
answers that, and the answer is informative either way: if the model wins, the
constants were leaving signal on the table, and if it loses, a rule derived from
how fencing works beat a model given three minutes of footage per clip, which is
a result about how much data this problem needs rather than an absence of one.

THE PROTOCOL, AND WHY EACH PART OF IT IS NOT NEGOTIABLE.

LEAVE ONE CLIP OUT, never a random split. Candidates a second apart describe
overlapping stretches of the same play, so a random split puts near-duplicates
on both sides of it and reports a number that measures memorisation. The unit
that generalisation is claimed over is the RECORDING, so the recording is the
unit that must be held out. This is also the error the audio detector was
diagnosed with, in a different form: it was tuned and read on the same clip.

THE THRESHOLD IS CHOSEN INSIDE THE TRAINING FOLDS. A score has to become a
decision somewhere, and choosing where by looking at the held-out clip is the
single easiest way to manufacture a good result. Each outer fold therefore runs
its own inner leave-one-clip-out over the three training clips, pools those
out-of-fold predictions, and takes the threshold that maximises F1 on them. The
test clip is not consulted until the number is final.

THE BASELINE IS EVALUATED IDENTICALLY. The rule is applied to the same candidate
set, merged by the same rule, and scored by the same function with the same
tolerance. Comparing a learned model against published numbers produced by a
different pipeline would be comparing two protocols and calling it a model
comparison.

WHAT IS NOT LEARNED. Candidate generation stays a domain rule: the model ranks
local minima, it does not find touches in video. The report should say so
plainly. See `touch_features.py`.

Run with:
    python3 train_touch.py --results results_current
    python3 train_touch.py --results results_current --curve --ablation
"""

import argparse
import itertools
import json
import sys

import numpy as np

import evaluate_touches as et
import touch_features as tf

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The rule's own constants, as published in detect_touches. Imported as values
# rather than re-typed so the baseline cannot drift away from the system.
RULE_PROMINENCE_M = tf.dt.MIN_PROMINENCE_M
RULE_SEPARATION_M = tf.dt.MIN_SEPARATION_M
MERGE_GAP_S = tf.dt.MERGE_GAP_S
TOLERANCE_S = tf.MATCH_TOLERANCE_S

SEED = 17

FEATURE_GROUPS = {
    "distance":   ["min_distance_m", "min_distance_pct"],
    "prominence": ["prominence_m", "prominence_pct"],
    "separation": ["separation_after_m", "separation_after_pct",
                   "separation_before_m"],
    "rates":      ["closing_rate_ms", "opening_rate_ms"],
    "dwell":      ["dwell_close_s"],
    "pose":       ["stance_max_m", "hip_height_min_m"],
    "context":    ["piste_position_pct", "time_since_prev_s", "bout_fraction"],
}


def models():
    """The two candidates, and why only these two.

    A small sample is the whole problem here: 27 positives across four
    recordings. Logistic regression is the most constrained model that can use
    these features at all, and gradient boosting the least constrained one still
    defensible at this size. Anything with more capacity, a temporal network in
    particular, would be fitting more parameters than there are positive
    examples and could not be reported honestly whatever it scored.
    """
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED)),
        # Handles NaN natively, so the pose features stay usable on frames where
        # pose did not run rather than being imputed into existence.
        "boosted": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.1,
            min_samples_leaf=10, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED),
    }


def merge(times, scores, gap=MERGE_GAP_S):
    """Collapse candidates belonging to one event, keeping the strongest.

    The same step the rule performs, applied to the model's scores, so neither
    side is credited for proposing the same touch several times.
    """
    order = np.argsort(times)
    out = []
    for i in order:
        if out and times[i] - out[-1][0] < gap:
            if scores[i] > out[-1][1]:
                out[-1] = (times[i], scores[i])
        else:
            out.append((times[i], scores[i]))
    return out


def score_proposals(times, scores, thresh, truth):
    """Threshold, merge, then score against the labels the rule is scored on."""
    keep = scores >= thresh
    merged = merge(times[keep], scores[keep]) if keep.any() else []
    cands = [{"time_s": t} for t, _ in merged]
    return et.score([{"time_s": g} for g in truth], cands, TOLERANCE_S)


def rule_scores(X, names):
    """The hand rule expressed as a score over the same candidates.

    Accepted candidates are ranked by separation, which is what the rule's own
    confidence does; rejected ones score below any threshold the sweep reaches.
    Expressing it this way rather than as a yes-or-no lets the baseline be drawn
    on the same precision-recall axes as the model instead of as a single point
    that cannot be compared across operating points.
    """
    prom = X[:, names.index("prominence_m")]
    sep = X[:, names.index("separation_after_m")]
    ok = (prom >= RULE_PROMINENCE_M) & (sep >= RULE_SEPARATION_M)
    ok &= np.isfinite(prom) & np.isfinite(sep)
    s = np.where(ok, np.nan_to_num(sep, nan=0.0), -1.0)
    return s


def fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def sweep(times, scores, truth, grid=None):
    """Best threshold and its F1 over a grid of operating points."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.5, 0.0
    if grid is None:
        grid = np.unique(np.quantile(finite, np.linspace(0.5, 1.0, 60)))
    best = (grid[0], -1.0)
    for th in grid:
        f1 = score_proposals(times, scores, th, truth)["f1"]
        if f1 > best[1]:
            best = (th, f1)
    return best


def training_split(data, clips):
    """Stack the training clips, dropping the ignore class."""
    Xs, ys = [], []
    for c in clips:
        X, y = data[c]["X"], data[c]["y"]
        keep = y >= 0
        Xs.append(X[keep]); ys.append(y[keep])
    return np.vstack(Xs), np.concatenate(ys)


def pick_threshold(model_name, data, train_clips):
    """Choose an operating point using ONLY the training clips.

    Inner leave-one-clip-out over the three training clips, pooling the
    out-of-fold scores before the sweep. Pooling matters: sweeping each inner
    fold separately and averaging the chosen thresholds would pick a value that
    was never evaluated on anything.
    """
    if len(train_clips) < 2:
        raise ValueError(
            "threshold selection needs at least two training clips: with one "
            "there is no inner fold to sweep on, and falling back to a fixed "
            "default would make that point incomparable with the others")
    times, scores, truth_all = [], [], []
    offset = 0.0
    for held in train_clips:
        rest = [c for c in train_clips if c != held]
        if not rest:
            continue
        Xtr, ytr = training_split(data, rest)
        d = data[held]
        s = fit_predict(models()[model_name], Xtr, ytr, d["X"])
        # Shift each inner clip onto its own stretch of a shared timeline so
        # merging and matching cannot pair events across different recordings.
        times.append(d["times"] + offset)
        scores.append(s)
        truth_all.extend(g + offset for g in d["truth"])
        offset += float(d["times"].max()) + 1000.0 if d["times"].size else 1000.0
    if not times:
        return 0.5
    th, _ = sweep(np.concatenate(times), np.concatenate(scores), truth_all)
    return th


def evaluate(data, model_name, clips=None, train_on=None):
    """Outer leave-one-clip-out. Returns a result per held-out clip."""
    clips = clips or list(data)
    out = []
    for test in clips:
        pool = [c for c in clips if c != test]
        train_clips = train_on(pool) if train_on else pool
        if not train_clips:
            continue
        d = data[test]
        if model_name == "rule":
            s = rule_scores(d["X"], tf.FEATURE_NAMES)
            th = 0.0
        else:
            th = pick_threshold(model_name, data, train_clips)
            Xtr, ytr = training_split(data, train_clips)
            s = fit_predict(models()[model_name], Xtr, ytr, d["X"])
        r = score_proposals(d["times"], s, th, d["truth"])
        out.append({"clip": test, "n_train": len(train_clips),
                    "threshold": float(th), "touches": len(d["truth"]),
                    **{k: r[k] for k in
                       ("tp", "fp", "fn", "precision", "recall", "f1",
                        "corrections")}})
    return out


def pooled(rows):
    """Micro-averaged figures: sum the counts, then compute the rates.

    Not a mean of per-clip F1. Clip 3 has fourteen touches and clip 1 has three,
    and averaging their F1 scores would weight those equally, which would let a
    good result on three touches offset a bad one on fourteen.
    """
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r_,
            "f1": (2 * p * r_ / (p + r_) if p + r_ else 0.0),
            "corrections": fp + fn}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learned touch proposer vs the hand rule")
    ap.add_argument("--results", default="results_current")
    ap.add_argument("--base", default=".")
    ap.add_argument("--curve", action="store_true", help="learning curve")
    ap.add_argument("--ablation", action="store_true", help="feature ablation")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    data = tf.build_all(args.results, args.base)
    if len(data) < 2:
        print("need at least two labelled clips to hold one out")
        return 1

    report = {}
    print(f"Leave one clip out over {len(data)} clips, "
          f"{sum(len(d['truth']) for d in data.values())} labelled touches")
    print("=" * 74)
    print(f"{'model':<10}{'clip':<16}{'n_tr':>5}{'tp':>4}{'fp':>4}{'fn':>4}"
          f"{'prec':>7}{'rec':>7}{'F1':>7}")
    for name in ("rule", "logistic", "boosted"):
        rows = evaluate(data, name)
        report[name] = {"folds": rows, "pooled": pooled(rows)}
        for r in rows:
            print(f"{name:<10}{r['clip']:<16}{r['n_train']:>5}{r['tp']:>4}"
                  f"{r['fp']:>4}{r['fn']:>4}{r['precision']:>7.2f}"
                  f"{r['recall']:>7.2f}{r['f1']:>7.2f}")
        p = report[name]["pooled"]
        print(f"{name:<10}{'POOLED':<16}{'':>5}{p['tp']:>4}{p['fp']:>4}"
              f"{p['fn']:>4}{p['precision']:>7.2f}{p['recall']:>7.2f}"
              f"{p['f1']:>7.2f}")
        print("-" * 74)

    if args.curve:
        # WHY THE CURVE STARTS AT TWO CLIPS. Choosing the operating point needs
        # an inner leave-one-clip-out inside the training set, so one training
        # clip has no inner fold. An earlier version silently fell back to a
        # fixed 0.5 there, which made the one-clip point the only one whose
        # threshold was untuned, and it duly scored HIGHER than two clips. That
        # was the protocol changing, not the model learning less from more data.
        sizes = [k for k in (2, 3) if k <= len(data) - 1]
        print("\nLearning curve: F1 against number of training clips")
        print(f"{'model':<10}" + "".join(f"{k} clips".rjust(9) for k in sizes)
              + f"{'spread':>9}")
        for name in ("logistic", "boosted"):
            cells, spreads = [], []
            for k in sizes:
                f1s = []
                for train_clips in itertools.combinations(sorted(data), k + 1):
                    # every (k+1)-subset: one clip held out, k used for training
                    for test in train_clips:
                        rest = [c for c in train_clips if c != test]
                        rows = evaluate(data, name, clips=[test],
                                        train_on=lambda pool, r=rest: r)
                        if rows:
                            f1s.append(rows[0]["f1"])
                cells.append(float(np.mean(f1s)))
                spreads.append(float(np.std(f1s)))
            report.setdefault("curve", {})[name] = {
                "sizes": sizes, "mean_f1": cells, "sd_f1": spreads}
            print(f"{name:<10}" + "".join(f"{c:>9.2f}" for c in cells)
                  + f"{max(spreads):>9.2f}")
        print("\nRead the SLOPE against the SPREAD. With four clips the curve has")
        print("two points and the fold-to-fold standard deviation is of the same")
        print("order as any difference between them, so this cannot currently")
        print("distinguish a data-starved model from a flat one. That is a")
        print("statement about the evidence base, not about the model.")

    if args.ablation:
        print("\nFeature ablation: F1 with each group removed")
        base = pooled(evaluate(data, "boosted"))["f1"]
        print(f"{'removed':<14}{'F1':>7}{'delta':>8}")
        print(f"{'nothing':<14}{base:>7.2f}{'':>8}")
        abl = {}
        full = list(tf.FEATURE_NAMES)
        for group, cols in FEATURE_GROUPS.items():
            keep = [i for i, n in enumerate(full) if n not in cols]
            sub = {c: {**d, "X": d["X"][:, keep]} for c, d in data.items()}
            f1 = pooled(evaluate(sub, "boosted"))["f1"]
            abl[group] = f1
            print(f"{group:<14}{f1:>7.2f}{f1 - base:>+8.2f}")
        report["ablation"] = {"base": base, **abl}

    if args.json:
        json.dump(report, open(args.json, "w"), indent=2, default=float)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
