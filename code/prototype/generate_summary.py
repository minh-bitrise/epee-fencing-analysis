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

    # cumulative push/pull totals are running sums, so the last row holds them
    last = rows[-1]
    fencers = {}
    for key in ("f1", "f2"):
        push = float(last[f"{key}_advance_m"])
        pull = float(last[f"{key}_retreat_m"])
        total = push + pull
        fencers[key] = {
            "push_m": round(push, 2),
            "pull_m": round(pull, 2),
            "net_forward_m": round(push - pull, 2),
            "push_share_pct": round(100.0 * push / total, 1) if total > 0 else None,
        }

    return {
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
        "fencer_1": fencers["f1"],
        "fencer_2": fencers["f2"],
    }


def add_in_play_scope(stats, rows, touch_path):
    """
    Enrich the payload with touch events and in-play-only metrics.

    Without this the totals span the whole recording, including the walk back
    to the guard line after every touch, which on clip 3 is 43 per cent of the
    frames and inflates movement totals by a third. Adding both the scoped and
    unscoped figures lets the model say which is which rather than having to
    hedge every number.
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

    def per_fencer(push, pull):
        """
        Derived movement figures for one fencer, in-play only.

        net_forward_m is included but flagged, because it is the metric most
        corrupted by camera panning: over a whole bout a fencer returns to
        roughly where they started, so a large net value indicates measurement
        error rather than tactics. Supplying it unflagged previously led the
        model to report an impossible +21.88 m as decisive aggression.
        """
        total = push + pull
        return {
            "push_m": round(push, 2),
            "pull_m": round(pull, 2),
            "push_share_pct": round(100.0 * push / total, 1) if total > 0 else None,
            "net_forward_m": round(push - pull, 2),
        }

    f1 = per_fencer(ip["f1_push_m"], ip["f1_pull_m"])
    f2 = per_fencer(ip["f2_push_m"], ip["f2_pull_m"])
    stats["in_play_only"] = {
        "share_of_recording_pct": round(100.0 * scoped["in_play_fraction"], 1),
        "mean_distance_m": ip["mean_distance_m"],
        "fencer_1": f1,
        "fencer_2": f2,
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

    # A piste is 14 m long and fencers reset between touches, so net forward
    # displacement across a bout should be small. Anything large is the
    # panning artefact, and the reader must be told rather than left to
    # interpret it as behaviour.
    worst_net = max(abs(f1["net_forward_m"]), abs(f2["net_forward_m"]))
    if worst_net > 5.0:
        stats["data_quality_warnings"] = [
            f"net_forward_m is unreliable on this recording (largest magnitude "
            f"{worst_net:.1f} m). Fencers reset between touches, so net "
            f"displacement over a bout should be near zero; a large value "
            f"indicates uncorrected camera motion inflating the movement "
            f"totals. Do not interpret net_forward_m as aggression or "
            f"territorial gain. push_share_pct is affected by the same cause "
            f"and should be treated as indicative only."
        ]
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
    "top-level totals span the whole recording, including walk-backs, and "
    "should only be mentioned if the difference between the two is itself "
    "interesting.\n"
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
    return _PROMPT_HEAD + middle + _PROMPT_TAIL


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
