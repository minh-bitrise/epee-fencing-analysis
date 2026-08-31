"""
Epee Fencing Bout Analysis - LLM Tactical Summary (basic implementation)
========================================================================
Third pre-trained model in the pipeline: a large language model that turns
the per-frame metrics CSV produced by run_detection.py into a short,
human-readable tactical summary - the same idea as the AI insights in
platforms like Garmin Connect or Strava.

NOTE: this is deliberately a BASIC first implementation. The prototype
only produces distance and push/pull data, so the summaries are thin.
The pipeline (stats -> prompt -> API -> cached markdown) stays the same
as richer data arrives later (confirmed touches, in-play-only metrics,
tempo, cross-bout profiles) - only the payload grows. See TODO.md B7.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 generate_summary.py --csv results/fencing_clip_distance.csv
    python3 generate_summary.py --csv results/fencing_clip_distance.csv --force

Outputs (next to the CSV):
    <base>_summary.md         the generated tactical summary
    <base>_summary.meta.json  cache record so unchanged data is not re-billed
"""

import argparse
import csv
import hashlib
import json
import os
import statistics

# The LLM to use. Claude Opus 4.8 is the current recommended default.
DEFAULT_MODEL = "claude-opus-4-8"

# Tactical distance bands in metres, front foot to front foot - these mirror
# the values in run_detection.py, where the derivation from weapon geometry
# is documented. Briefly: reach from the front foot is about 1.2 m standing
# and about 2.3 m through a lunge, so a lunge-scored touch lands at roughly
# 2.0 to 2.6 m of front-foot separation.
DIST_CLOSE_M         = 1.5   # infighting; a touch lands without a lunge
DIST_LUNGE_M         = 2.6   # lunge distance; a touch can land with a lunge
DIST_ADVANCE_LUNGE_M = 3.5   # needs a step plus a lunge to reach


# --- data loading and statistics ---------------------------------------

def load_rows(csv_path):
    """Read the per-frame metrics CSV into a list of dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# Play resets to the guard lines after every touch, so over a whole recording
# each fencer should finish within roughly a metre of where they started. That
# physical constraint is a correctness check on net displacement needing no
# ground truth, and this is the magnitude above which it is clearly violated on
# a 14 m piste.
IMPLAUSIBLE_NET_M = 5.0

# Below this magnitude a net displacement says only "finished where they
# started". Fencer 2 on clip 3 measures +0.43 m from raw positions and -0.94 m
# from smoothed endpoints: both readings agree on the substance and disagree on
# the sign, which is what makes the sign meaningless at this scale.
NET_SIGN_FLOOR_M = 1.0


def movement_basis(rows):
    """
    Say where the movement figures were derived from, so the payload records it.

    The route matters. The raw position columns are the measurement; the
    cumulative advance and retreat columns are written after the noise floor, the
    movement cap and the banking buffer have been applied, so anything derived
    from them inherits all three. On clip 3 the two routes disagreed by 3.5 m on
    net displacement, which is why the position columns were added.
    """
    from_positions = bool(rows) and "f1_pos_m" in rows[0]
    return {
        "source": ("raw per-frame positions" if from_positions
                   else "differenced cumulative totals (older CSV, no position "
                        "columns; inherits the noise floor, movement cap and "
                        "banking buffer)"),
        "closing_share_is_window_dependent": True,
        "closing_share_note": (
            "Measured on unsmoothed positions. A 1-to-121 frame median smoothing "
            "sweep moved this figure from 52.4 to 60.9 per cent on one clip, so "
            "read it as approximate to a few points rather than exact."
        ),
    }


def count_side_swaps(rows):
    """
    How many times the two tracked slots exchange sides of the piste.

    A DIRECT test of whether slot identity held, and the one this project was
    missing. Two fencers do not cross on a piste: one stays on the referee's left
    for the whole bout and the other on the right. So a sign change in
    `f1_pos_m - f2_pos_m` is not a fencing event, it is the tracker exchanging
    which fencer each slot is following.

    Measured over the evaluation set, this separates the clips cleanly: clips 1,
    2 and 3 record ZERO swaps and hold one fencer on the left in 100 per cent of
    frames, while clip 4 records 14 swaps and holds f1 on the left in only 26.9
    per cent. That is the explanation for clip 4's otherwise inexplicable +7.08 m
    of net displacement, which the report had recorded as a cause not yet
    identified: a first-to-last displacement for a slot that changed fencer
    fourteen times measures the swaps rather than the fencer.

    Returns (swaps, fraction_of_frames_with_f1_on_the_left, min_separation_m).
    """
    pairs = [(float(r["f1_pos_m"]), float(r["f2_pos_m"])) for r in rows
             if r.get("f1_pos_m") and r.get("f2_pos_m")]
    if len(pairs) < 2:
        return 0, None, None
    diffs = [a - b for a, b in pairs]
    swaps = sum(1 for x, y in zip(diffs, diffs[1:])
                if (x > 0) != (y > 0))
    left_share = sum(1 for d in diffs if d < 0) / len(diffs)
    return swaps, left_share, min(abs(d) for d in diffs)


def movement_quality_warnings(stats, rows=None):
    """
    Warnings about the whole-recording movement figures, or an empty list.

    The check applies to net displacement over the whole recording and to nothing
    else. That figure is a genuine first-to-last displacement and so is subject to
    the reset-to-guard-lines constraint. The in-play figure is not a displacement
    and legitimately exceeds it, so applying the same test there would flag
    correct data as broken.
    """
    out = []

    # Identity first, because it explains the magnitude check below rather than
    # merely accompanying it, and because it fires on evidence rather than on a
    # threshold: any swap at all means a slot changed fencer.
    if rows:
        swaps, left_share, min_sep = count_side_swaps(rows)
        if swaps:
            out.append(
                f"the two tracked slots exchanged sides of the piste {swaps} "
                f"time(s), and slot 1 held the left-hand fencer in only "
                f"{100 * left_share:.0f} per cent of frames. Fencers do not "
                f"cross on a piste, so this is the tracker changing which fencer "
                f"each slot follows, not a fencing event. PER-SLOT FIGURES ARE "
                f"NOT ATTRIBUTABLE TO A PARTICULAR FENCER on this recording: "
                f"net_displacement_m in particular is a first-to-last difference "
                f"and measures the swaps. The two came within "
                f"{min_sep:.2f} m in the measured position, which is where the "
                f"matcher cannot tell them apart.")

    worst = max(abs(stats["fencer_1"]["net_displacement_m"]),
                abs(stats["fencer_2"]["net_displacement_m"]))
    if worst > IMPLAUSIBLE_NET_M:
        out.append(
            f"net_displacement_m is implausible on this recording (largest "
            f"magnitude {worst:.1f} m). Play resets to the guard lines after "
            f"every touch, so each fencer should finish within about a metre of "
            f"where they started. Do not interpret net_displacement_m as "
            f"aggression or territorial gain on this bout. Camera panning was "
            f"tested and ruled out, because panning moves the two fencers' net "
            f"figures in opposite directions. closing_share_pct counts "
            f"directions rather than magnitudes, so it is unaffected; use it "
            f"instead.")
    return out


def compute_stats(rows):
    """
    Aggregate the per-frame rows into the bout-level statistics that the
    LLM will interpret. Returns a plain dict (easy to test and to dump
    into the prompt as JSON).
    """
    if not rows:
        raise ValueError("CSV contains no rows")

    distances = [float(r["distance_raw_m"]) for r in rows if r["distance_raw_m"] != ""]
    if not distances:
        raise ValueError("CSV contains no distance samples")

    total_frames = len(rows)
    duration_s = float(rows[-1]["time_s"])

    # how much of the bout the tracker actually covered
    coverage_pct = 100.0 * len(distances) / total_frames
    pose_frames = sum(1 for r in rows if r["method"] == "pose")
    pose_pct = 100.0 * pose_frames / len(distances)

    # share of time spent in each tactical distance band
    close = sum(1 for d in distances if d <= DIST_CLOSE_M)
    lunge = sum(1 for d in distances if DIST_CLOSE_M < d <= DIST_LUNGE_M)
    adv   = sum(1 for d in distances if DIST_LUNGE_M < d <= DIST_ADVANCE_LUNGE_M)
    out   = len(distances) - close - lunge - adv

    # Movement. The headline figures are net displacement and closing share, not
    # the cumulative push and pull totals.
    #
    # Cumulative path length is not a measurement. Re-measuring clip 3's position
    # series under median smoothing windows from 1 to 121 frames moved the total
    # from 161 m to 33 m with no asymptote, while net displacement stayed at
    # exactly +3.43 m. Path length sums the magnitude of every frame's change, so
    # measurement noise adds to it and never cancels; net is a difference between
    # two positions, so noise cancels. A quantity that changes fivefold with an
    # arbitrary smoothing parameter measures the filter, not the fencer.
    #
    # The totals are NOT passed to the model at all. They were briefly included
    # under names ending "_indicative", and the model handled them correctly,
    # writing "the cumulative push and pull totals are indicative only and are not
    # quoted as distance or aggression". They are dropped anyway, for a reason that
    # is about the prompt rather than the model: a figure that cannot support any
    # claim has no business in the payload, and keeping it meant spending three
    # sentences of prompt talking the model out of a number worth nothing. B1g
    # measured the error at 24 m on a 14 m piste, so this is not an approximation
    # that might be useful with caveats. The totals stay in the CSV, which is the
    # evidence artefact and where the before/after comparison lives.
    from in_play import closing_share, net_forward_movement
    import numpy as np

    all_frames = np.ones(total_frames, dtype=bool)
    fencers = {}
    for key in ("f1", "f2"):
        cs = closing_share(rows, key, all_frames)
        fencers[key] = {
            # Summed over EVERY frame the signed movement telescopes to the
            # first-to-last position difference, so this is a true displacement.
            # The scoped figure in add_in_play_scope is not, and is named
            # differently for that reason.
            "net_displacement_m": round(net_forward_movement(rows, key, all_frames), 2),
            "closing_share_pct": round(100.0 * cs, 1) if cs is not None else None,
        }

    stats = {
        "duration_s": round(duration_s, 1),
        "frames_total": total_frames,
        "coverage_pct": round(coverage_pct, 1),
        "pose_method_pct": round(pose_pct, 1),
        "distance_m": {
            "mean": round(statistics.mean(distances), 2),
            "min": round(min(distances), 2),
            "max": round(max(distances), 2),
            "std": round(statistics.pstdev(distances), 2),
        },
        "time_in_zone_pct": {
            "close_infighting_under_1.5m": round(100.0 * close / len(distances), 1),
            "lunge_distance_1.5_to_2.6m": round(100.0 * lunge / len(distances), 1),
            "advance_lunge_2.6_to_3.5m": round(100.0 * adv / len(distances), 1),
            "out_of_distance_over_3.5m": round(100.0 * out / len(distances), 1),
        },
        "movement_basis": movement_basis(rows),
        "fencer_1": fencers["f1"],
        "fencer_2": fencers["f2"],
    }
    warnings = movement_quality_warnings(stats, rows)
    if warnings:
        stats["data_quality_warnings"] = warnings
    return stats


def add_in_play_scope(stats, rows, touch_path):
    """
    Enrich the payload with touch events and in-play-only metrics.

    Without this the figures span the whole recording, including the walk back to
    the guard line after every touch, which on clip 3 is 43 per cent of the
    frames. Adding both the scoped and unscoped figures lets the model say which
    is which rather than having to hedge every number.

    Note that scoping does not simply shrink the numbers. It raises the net
    movement figure on clip 3, from +3.4 m to +7.6 m, because the resets are
    where ground gained during a phrase is given back. See per_fencer below for
    why the scoped figure is therefore named as movement and not displacement.
    """
    from in_play import load_touch_times, scope_metrics, touch_provenance

    touches = load_touch_times(touch_path)
    if not touches:
        return stats

    scoped = scope_metrics(rows, touches)
    ip = scoped["in_play"]
    stats["touches"] = {
        "count": scoped["touches"],
        "times_s": [round(x, 1) for x in sorted(touches)],
        "provenance": touch_provenance(touch_path),
        "reset_excluded_s": scoped["reset_s"],
    }

    def per_fencer(prefix):
        """
        Movement figures for one fencer, in-play only.

        The scoped figure is deliberately NOT called a displacement, and the
        distinction is not pedantry. Summed over every frame the signed movement
        telescopes to the first-to-last position difference, so it is a true
        displacement. Summed over a masked subset it does not, because the
        excluded resets break the series and the sum jumps across them.

        On clip 3 the in-play figure (+7.6 m) is larger than the whole-recording
        one (+3.4 m), and correctly so: fencers gain ground during a phrase and
        give it back walking to the guard line, so removing the resets removes
        the giving-back. Presenting the scoped number as displacement would
        suggest a fencer finished 7.6 m up a 14 m piste when they finished
        roughly where they started.
        """
        cs = ip[f"{prefix}_closing_share"]
        return {
            "net_forward_movement_m": ip[f"{prefix}_net_m"],
            "closing_share_pct": round(100.0 * cs, 1) if cs is not None else None,
        }

    stats["in_play_only"] = {
        "share_of_recording_pct": round(100.0 * scoped["in_play_fraction"], 1),
        "mean_distance_m": ip["mean_distance_m"],
        "fencer_1": per_fencer("f1"),
        "fencer_2": per_fencer("f2"),
    }

    # Tempo. This is the group of metrics Chapter 1 promises and the pipeline
    # could not produce before touch detection existed, which is why earlier
    # summaries were accurate but thin. The per-exchange list is left out of the
    # payload deliberately: the model needs the distribution to reason about
    # tempo, not fourteen individual intervals it would be tempted to narrate.
    from tempo import compute_tempo, load_touches_with_scorer
    t_touches = load_touches_with_scorer(touch_path)
    tempo = compute_tempo(t_touches, stats["duration_s"])
    tempo.pop("exchanges", None)
    stats["tempo"] = tempo

    # The reset-to-guard-lines plausibility check is NOT repeated here. It belongs
    # to the whole-recording net displacement, which compute_stats already tests,
    # because only that figure is an endpoint measurement. Applying it to the
    # in-play figure would flag correct data: the scoped sum is expected to be
    # larger, for the reason given in per_fencer above.
    return stats


# --- prompt -------------------------------------------------------------

_PROMPT_HEAD = (
    "You are a fencing coach's assistant. You are given automatically "
    "extracted metrics from a single epee bout video. Write a tactical "
    "summary for the fencers and their coach.\n"
    "\n"
    "Important honesty constraints - respect them strictly:\n"
    "- The data comes from a computer-vision prototype. Distances are "
    "estimates (normalised via fencer height), not precise measurements.\n"
)

# Used when no touch data was supplied.
_PROMPT_NO_TOUCHES = (
    "- The metrics are measured over the WHOLE video, including breaks "
    "between touches and walk-backs, because the system does not yet know "
    "when touches happen. Do not treat totals as in-play-only values.\n"
    "- There is no touch/score data yet. Never invent touches, scores, "
    "actions, or events that are not in the data.\n"
)

# Used when touch events are available. The constraints have to change with
# the payload: telling the model there is no touch data while supplying it
# would be contradictory, and would waste the one signal that lets it
# separate active fencing from resets.
_PROMPT_WITH_TOUCHES = (
    "- Touch timestamps ARE available and are listed in the payload. You may "
    "refer to how many touches occurred and to their timing and spacing.\n"
    "- Two sets of movement figures are given. Those under 'in_play_only' "
    "exclude the reset after each touch and are the ones to reason from. The "
    "top-level figures span the whole recording, including walk-backs, and "
    "should only be mentioned if the difference between the two is itself "
    "interesting.\n"
    "- The in-play figure is called 'net_forward_movement_m' and is NOT a "
    "displacement. Excluding the resets breaks the position series, so it can "
    "legitimately be larger than the whole-recording 'net_displacement_m': "
    "fencers gain ground during a phrase and give it back walking to the guard "
    "line. Do not say a fencer finished that far up the piste.\n"
    "- The touch list says WHEN touches happened, not who scored them. Never "
    "attribute a touch to a fencer, state a score, or name a winner. Never "
    "describe the action that produced a touch, since that is not in the "
    "data.\n"
    "- A 'tempo' block is present when touch data is. Use it: touch rate, time "
    "between touches, exchange duration and whether the bout quickened or slowed "
    "are the tactically richest figures available, and they are what distinguish "
    "one bout from another most clearly. If 'scorer_counts' is null, the touch "
    "source did not record who scored; do not infer it, and do not comment on "
    "streaks or momentum in that case.\n"
    "- Check the touch list's 'provenance' field. 'human_confirmed' means a "
    "person labelled these touches and the count is reliable. "
    "'automatic_detector' means they were proposed by the system and the "
    "count is approximate, which you must say. Do not guess which it is.\n"
    "- If a 'data_quality_warnings' field is present, treat every warning in "
    "it as binding and do not use the metrics it names as evidence for any "
    "tactical claim. Mention the limitation in the caveats section.\n"
)

# How to read the movement figures. This is in the prompt rather than left to
# the model because the payload alone cannot convey it: the numbers look
# equally authoritative whatever their provenance, and an earlier version of
# this pipeline showed the model interpreting a mis-specified input entirely
# faithfully. Constraining the reading is the only place the caveat can live.
_PROMPT_MOVEMENT = (
    "\n"
    "Reading the movement figures - follow this exactly:\n"
    f"- 'net_displacement_m' is where a fencer finished relative to where they "
    f"started, positive meaning toward the opponent. It is the reliable "
    f"movement measurement. Under about {NET_SIGN_FLOOR_M:.0f} m it means the "
    f"fencer finished roughly where they began and the SIGN CARRIES NO "
    f"MEANING, so do not describe such a value as gaining or losing ground.\n"
    "- 'closing_share_pct' is the proportion of moving frames spent reducing "
    "the distance to the opponent. This is the figure to use for who pressed "
    "forward more, because it counts the direction of each movement rather "
    "than its size. Near 50 per cent means the fencers traded ground evenly. "
    "It is accurate to a few percentage points only, so treat a small "
    "difference between the two fencers as noise rather than a tendency.\n"
    "- There is deliberately NO figure for total distance covered, or for total "
    "ground advanced versus retreated. That quantity is not measurable from this "
    "data: summing the size of every frame's movement accumulates tracking noise "
    "that never cancels, and on one clip the identical footage gave totals "
    "anywhere between 33 m and 161 m depending only on an arbitrary smoothing "
    "setting. Do not estimate it, do not derive it from the figures that are "
    "present, and do not describe a fencer as having covered any distance.\n"
)

_PROMPT_TAIL = (
    "- If the data is too thin to support a claim, say so rather than "
    "speculating.\n"
    "\n"
    "Reading the distance bands: distances are front foot to front foot, and "
    "the bands follow the standard fencing taxonomy. 'Lunge distance' is the "
    "band in which a touch can actually be scored with a lunge, so it is the "
    "tactically live range, not a safe one. 'Close/infighting' is nearer than "
    "that. 'Advance-lunge' requires a step before a lunge will reach, and "
    "'out of distance' is beyond that. Do not describe lunge distance as "
    "cautious or long range.\n"
    "\n"
    "Formatting constraint: write with short hyphens only. Do not use em "
    "dashes or en dashes anywhere in the output. This matters because the "
    "summary is pasted into a report whose house style forbids them."
)


def build_system_prompt(has_touches=False):
    """
    Assemble the system prompt to match the payload actually supplied.

    The constraints are not static: whether touch data exists changes what the
    model may say and which figures it should reason from. Keeping one fixed
    prompt would either forbid using data that is present or permit claims the
    data cannot support.
    """
    middle = _PROMPT_WITH_TOUCHES if has_touches else _PROMPT_NO_TOUCHES
    return _PROMPT_HEAD + middle + _PROMPT_TAIL + _PROMPT_MOVEMENT


# Retained for callers and tests that want the no-touch prompt by name.
SYSTEM_PROMPT = build_system_prompt(has_touches=False)

USER_TEMPLATE = (
    "Here are the extracted metrics for one epee bout (JSON):\n"
    "\n"
    "{stats_json}\n"
    "\n"
    "Write the summary in exactly this structure, in markdown:\n"
    "\n"
    "## Bout summary\n"
    "One short paragraph describing the overall character of the bout "
    "(distance behaviour, activity levels, who pressed and who yielded).\n"
    "\n"
    "## Observed tendencies\n"
    "3-5 bullet points, each tying a concrete number from the data to a "
    "tactical observation.\n"
    "\n"
    "## Suggestions to explore\n"
    "2-3 bullet points with things the fencers or coach could look at in "
    "the video, phrased as suggestions, not conclusions.\n"
    "\n"
    "## Data caveats\n"
    "One short paragraph reminding the reader of the measurement "
    "limitations described in your instructions."
)


def build_prompt(stats):
    """Fill the user prompt template with the stats payload."""
    return USER_TEMPLATE.format(stats_json=json.dumps(stats, indent=2))


def cache_key(model, system_prompt, user_prompt):
    """Hash everything that influences the output, so we can skip
    re-generating (and re-billing) when nothing changed."""
    blob = "\x00".join([model, system_prompt, user_prompt])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- LLM call -----------------------------------------------------------

def call_llm(model, system_prompt, user_prompt):
    """
    Send the prompt to Claude and return the summary text.
    Imported lazily so the rest of the module (stats, prompt, cache)
    works and is testable without the anthropic package or an API key.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# --- main ---------------------------------------------------------------

def generate(csv_path, model=DEFAULT_MODEL, force=False, touches=None):
    base = os.path.splitext(csv_path)[0]
    out_md = f"{base}_summary.md"
    out_meta = f"{base}_summary.meta.json"

    rows = load_rows(csv_path)
    stats = compute_stats(rows)
    if touches:
        stats = add_in_play_scope(stats, rows, touches)
    has_touches = "touches" in stats
    system_prompt = build_system_prompt(has_touches=has_touches)

    user_prompt = build_prompt(stats)
    key = cache_key(model, system_prompt, user_prompt)

    # cache: skip the API call when data + model + prompts are unchanged
    if not force and os.path.exists(out_md) and os.path.exists(out_meta):
        with open(out_meta) as f:
            meta = json.load(f)
        if meta.get("cache_key") == key:
            print(f"Summary is up to date (cache hit) -> {out_md}")
            return out_md

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Get a key from "
            "https://platform.claude.com/ and run:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )

    scope = "in-play scoped" if has_touches else "whole recording"
    print(f"Generating summary with {model} ({scope})...")
    summary = call_llm(model, system_prompt, user_prompt)

    with open(out_md, "w") as f:
        f.write(summary.rstrip() + "\n")
    with open(out_meta, "w") as f:
        json.dump({"cache_key": key, "model": model, "stats": stats}, f, indent=2)

    print(f"  Summary saved -> {out_md}")
    return out_md


def main():
    parser = argparse.ArgumentParser(description="Generate an LLM tactical summary from a metrics CSV")
    parser.add_argument("--csv", required=True, help="Path to a *_distance.csv produced by run_detection.py")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model ID (default {DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the cached summary is current")
    parser.add_argument("--touches", default=None,
                        help="Touch times CSV (ground truth or detector output). Adds touch "
                             "events and in-play-only metrics to the payload, and switches the "
                             "prompt constraints accordingly.")
    args = parser.parse_args()
    generate(args.csv, model=args.model, force=args.force, touches=args.touches)


if __name__ == "__main__":
    main()
