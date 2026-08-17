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


# --- prompt -------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a fencing coach's assistant. You are given automatically "
    "extracted metrics from a single epee bout video. Write a tactical "
    "summary for the fencers and their coach.\n"
    "\n"
    "Important honesty constraints - respect them strictly:\n"
    "- The data comes from a computer-vision prototype. Distances are "
    "estimates (normalised via fencer height), not precise measurements.\n"
    "- The metrics are measured over the WHOLE video, including breaks "
    "between touches and walk-backs, because the system does not yet know "
    "when touches happen. Do not treat totals as in-play-only values.\n"
    "- There is no touch/score data yet. Never invent touches, scores, "
    "actions, or events that are not in the data.\n"
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

def generate(csv_path, model=DEFAULT_MODEL, force=False):
    base = os.path.splitext(csv_path)[0]
    out_md = f"{base}_summary.md"
    out_meta = f"{base}_summary.meta.json"

    stats = compute_stats(load_rows(csv_path))
    user_prompt = build_prompt(stats)
    key = cache_key(model, SYSTEM_PROMPT, user_prompt)

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

    print(f"Generating summary with {model}...")
    summary = call_llm(model, SYSTEM_PROMPT, user_prompt)

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
    args = parser.parse_args()
    generate(args.csv, model=args.model, force=args.force)


if __name__ == "__main__":
    main()
