"""
Measure a piste polygon from a video instead of drawing one by hand.

Competition footage contains referees, spectators and fencers on the adjacent
strip, and the detector knows what a person is, not what a fencer is. Every
evaluation clip has a hand-authored polygon; an uploaded video has none, and
running clip 2 without one dropped coverage from 98.0 to 80.1 per cent.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from ultralytics import YOLO

# Imported rather than redeclared. These constants define what counts as a person
# detection, and a copy that drifted from the pipeline's would mean the polygon
# was measured against a different population than the one it later filters.
from run_detection import (
    CONF_THRESH, IOU_THRESH, MAX_FENCERS, MODEL_NAME, box_height_pixels,
)

# Sampling. Every tenth frame is what the hand measurements used, and it is
# enough: a three minute clip at 25 fps gives around 450 samples, which is a
# large sample of a distribution whose percentiles are all this needs.
SAMPLE_EVERY = 10
MAX_SAMPLE_FRAMES = 6000

# A detection this short is background. Guards against a distant spectator being
# treated as a fencer in a frame where the real fencers are briefly missed.
MIN_HEIGHT_PX = 40

# Margin applied outside the measured fencer band, as a fraction of the median
# fencer height. Deliberately small, because the percentiles below have already
# excluded the outliers and a margin on top of a percentile double-counts. The
# hand-authored clip-2 file used 22 px against a 252 px fencer, which is this.
MARGIN_FRACTION = 0.1

# Percentiles of the measured feet-y band that become the polygon edges. Not the
# extremes: a single frame where a bystander was measured instead of a fencer
# would otherwise stretch the band to include them, which is the failure the
# filter exists to prevent.
LOW_PCT = 1.0
HIGH_PCT = 99.0

# Two boxes are candidates to be the fencer pair only if the shorter is at least
# this fraction of the taller. Rejects pairing a fencer with a distant spectator.
PAIR_HEIGHT_RATIO = 0.5

# Feet-y values further apart than this many median fencer heights are treated as
# belonging to different groups of people.
CLUSTER_GAP_HEIGHTS = 0.5

# How many extra people have to be present before a piste filter is worth
# applying at all. Clip 3, the club training clip that is the project's target
# use case, contains nobody but the two fencers, needs no polygon, and would only
# be put at risk by one.
BYSTANDER_FRAME_FRACTION = 0.10


def _sample_detections(video_path, sample_every=SAMPLE_EVERY,
                       max_frames=MAX_SAMPLE_FRAMES, progress_cb=None):
    """
    Run person detection over sampled frames and return per-frame box lists.

    Detection only, no tracking: identity across frames is irrelevant here and
    `model.track` carries state that would make the sampling stride meaningless.
    """
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame = []
    idx = 0
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                    classes=[0], verbose=False)
            boxes = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = [b for b in results[0].boxes.xyxy.cpu().numpy()
                         if box_height_pixels(b) >= MIN_HEIGHT_PX]
            per_frame.append(boxes)
            if progress_cb is not None:
                progress_cb(idx, min(total, max_frames) if total else max_frames)
        idx += 1
    cap.release()
    return per_frame, idx


def select_fencer_pair(boxes):
    """Pick the two boxes in one frame most likely to be the fencers.

    NOT the two tallest, which is what `calibrate_fixed_scale` uses and what
    the first version of this module used.
    """
    if len(boxes) < 2:
        return None
    ordered = sorted(boxes, key=box_height_pixels, reverse=True)
    tallest = box_height_pixels(ordered[0])
    candidates = [b for b in ordered
                  if box_height_pixels(b) >= PAIR_HEIGHT_RATIO * tallest]
    if len(candidates) < 2:
        return None
    best, best_gap = None, float("inf")
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            gap = abs(float(candidates[i][3]) - float(candidates[j][3]))
            if gap < best_gap:
                best_gap, best = gap, (candidates[i], candidates[j])
    return best


def cluster_1d(values, gap):
    """Split a sorted set of values wherever consecutive ones differ by more than
    `gap`, and return the groups largest first.

    A percentile taken across a bimodal distribution describes neither mode.
    """
    if len(values) == 0:
        return []
    ordered = sorted(float(v) for v in values)
    groups, current = [], [ordered[0]]
    for v in ordered[1:]:
        if v - current[-1] > gap:
            groups.append(current)
            current = [v]
        else:
            current.append(v)
    groups.append(current)
    return sorted(groups, key=len, reverse=True)


def derive_piste(video_path, sample_every=SAMPLE_EVERY, progress_cb=None):
    """Measure a piste polygon for a video.

    Returns a dict describing the result rather than a bare polygon, because
    the caller has to be able to show the user what the measurement was.
    """
    per_frame, frames_read = _sample_detections(
        video_path, sample_every=sample_every, progress_cb=progress_cb)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    sampled = len(per_frame)
    if sampled == 0:
        return {"needed": False, "polygon": None, "confident": False,
                "reason": "no frames could be read from the video",
                "frame_size": [width, height], "measurements": {}}

    feet_y, heights = [], []
    extra_people = 0
    for boxes in per_frame:
        if not boxes:
            continue
        if len(boxes) > MAX_FENCERS:
            extra_people += 1
        pair = select_fencer_pair(boxes)
        if pair is None:
            continue
        for b in pair:
            feet_y.append(float(b[3]))
            heights.append(box_height_pixels(b))

    frames_with_people = sum(1 for b in per_frame if b)
    measurements = {
        "frames_sampled": sampled,
        "frames_read": frames_read,
        "frames_with_people": frames_with_people,
        "frames_with_extra_people": extra_people,
        "extra_people_fraction": round(extra_people / sampled, 3),
        "fencer_samples": len(feet_y),
    }

    if len(feet_y) < 20:
        return {"needed": False, "polygon": None, "confident": False,
                "reason": (f"only {len(feet_y)} usable fencer detections in "
                           f"{sampled} sampled frames, too few to measure a band"),
                "frame_size": [width, height], "measurements": measurements}

    feet = np.array(feet_y)
    median_height = float(np.median(heights))

    # Separate the groups before describing any of them, then keep the largest.
    groups = cluster_1d(feet, CLUSTER_GAP_HEIGHTS * median_height)
    main = np.array(groups[0])
    others = [g for g in groups[1:]]

    low = float(np.percentile(main, LOW_PCT))
    high = float(np.percentile(main, HIGH_PCT))
    margin = MARGIN_FRACTION * median_height

    measurements.update({
        "fencer_feet_y_min": round(float(main.min()), 1),
        "fencer_feet_y_max": round(float(main.max()), 1),
        f"fencer_feet_y_p{LOW_PCT:g}": round(low, 1),
        f"fencer_feet_y_p{HIGH_PCT:g}": round(high, 1),
        "median_fencer_height_px": round(median_height, 1),
        "margin_px": round(margin, 1),
        "groups_found": len(groups),
        "main_group_share": round(len(main) / len(feet), 3),
        "other_groups": [
            {"feet_y": [round(min(g), 1), round(max(g), 1)], "samples": len(g)}
            for g in others],
    })

    # No bystanders means no filter. Applying one anyway can only cost coverage:
    # every frame where a real fencer's feet fall outside a band measured from
    # other frames is a frame thrown away for nothing.
    if extra_people / sampled < BYSTANDER_FRAME_FRACTION:
        return {
            "needed": False, "polygon": None, "confident": True,
            "reason": (f"only {extra_people} of {sampled} sampled frames contain "
                       f"anyone besides the two fencers, so a piste filter would "
                       f"reject nothing and can only cost coverage"),
            "frame_size": [width, height], "measurements": measurements,
        }

    top = max(0.0, low - margin)
    bottom = min(float(height), high + margin)

    # Never let an edge cross into a neighbouring group of people. The margin is
    # there to avoid clipping a real fencer, and where another group sits close
    # by, half the distance to it is as far as the margin can go without
    # readmitting exactly what the filter exists to exclude.
    for g in others:
        g_low, g_high = min(g), max(g)
        if g_high < main.min():
            top = max(top, (g_high + main.min()) / 2.0)
        elif g_low > main.max():
            bottom = min(bottom, (main.max() + g_low) / 2.0)

    # How many people the band will still let through. This is the check that
    # matters, and coverage does not provide it.
    inside_extra = 0
    for boxes in per_frame:
        if not boxes:
            continue
        pair = select_fencer_pair(boxes)
        chosen = {id(b) for b in (pair or ())}
        for b in boxes:
            if id(b) not in chosen and top <= float(b[3]) <= bottom:
                inside_extra += 1
    per_frame_extra = inside_extra / sampled if sampled else 0.0
    measurements["unselected_people_inside_band"] = inside_extra
    measurements["unselected_per_sampled_frame"] = round(per_frame_extra, 3)

    # Two separate ways the result can be weak, reported separately because they
    # call for different responses.
    CROWDED_PER_FRAME = 0.5
    covers_everything = (bottom - top) >= 0.9 * height
    still_crowded = per_frame_extra >= CROWDED_PER_FRAME
    confident = not (covers_everything or still_crowded)

    reason = (f"{extra_people} of {sampled} sampled frames contain people besides "
              f"the two fencers; fencer feet measured in y "
              f"[{main.min():.0f}, {main.max():.0f}]")
    if covers_everything:
        reason += (f". The measured band covers {(bottom - top) / height:.0%} of "
                   f"the frame height, so it will reject little. Check whether "
                   f"the two people it found really are the fencers.")
    if still_crowded:
        reason += (f". About {per_frame_extra:.1f} other people per frame still "
                   f"fall inside this band, so it will not exclude them. This "
                   f"happens when the strip recedes from the camera and other "
                   f"people stand at the same apparent depth as its far end; "
                   f"raising the top edge excludes them at the cost of the far "
                   f"end of the strip.")

    return {
        "needed": True,
        "polygon": [[0, round(top, 1)], [width, round(top, 1)],
                    [width, round(bottom, 1)], [0, round(bottom, 1)]],
        "confident": confident,
        "reason": reason,
        "frame_size": [width, height],
        "measurements": measurements,
    }


def write_piste_file(result, path, video_path=""):
    """Write a derived polygon in the same schema the hand-authored files use.

    The measurements are written into the file's comment.
    """
    if not result.get("polygon"):
        raise ValueError("no polygon to write")
    m = result["measurements"]
    comment = (
        f"Piste polygon DERIVED BY MEASUREMENT from {os.path.basename(video_path)} "
        f"by derive_piste.py, not placed by eye. "
        f"Frame {result['frame_size'][0]}x{result['frame_size'][1]}. "
        f"Sampled {m.get('frames_sampled')} frames; "
        f"{m.get('frames_with_extra_people')} of them contain people besides the "
        f"two fencers. Fencer feet y measured in "
        f"[{m.get('fencer_feet_y_min')}, {m.get('fencer_feet_y_max')}], "
        f"p{LOW_PCT:g}={m.get(f'fencer_feet_y_p{LOW_PCT:g}')}, "
        f"p{HIGH_PCT:g}={m.get(f'fencer_feet_y_p{HIGH_PCT:g}')}. "
        f"Edges are those percentiles plus a margin of "
        f"{m.get('margin_px')} px, which is {MARGIN_FRACTION:g} of the median "
        f"fencer height of {m.get('median_fencer_height_px')} px."
    )
    with open(path, "w") as f:
        json.dump({"_comment": comment, "polygon": result["polygon"]}, f, indent=2)
    return path


def main():
    p = argparse.ArgumentParser(
        description="Measure a piste polygon from a video")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default=None,
                   help="Write the polygon here. Without it, the measurement is "
                        "printed and nothing is written.")
    p.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    p.add_argument("--json", action="store_true",
                   help="Print the full result as JSON, for a caller to parse.")
    p.add_argument("--output-json", default=None,
                   help="Write the full result to this path. Preferred over "
                        "--json by the web layer, which merges this process's "
                        "output streams into a log and so cannot also read a "
                        "structured result from them.")
    p.add_argument("--progress", action="store_true",
                   help="Print machine-readable 'PROGRESS done total' lines for "
                        "a supervising process.")
    args = p.parse_args()

    def report(done, total):
        print(f"PROGRESS {done} {total}", flush=True)

    result = derive_piste(args.video, sample_every=args.sample_every,
                          progress_cb=report if args.progress else None)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Piste needed:  {result['needed']}")
        print(f"Confident:     {result['confident']}")
        print(f"Reason:        {result['reason']}")
        for k, v in result["measurements"].items():
            print(f"  {k}: {v}")
        if result["polygon"]:
            print(f"Polygon:       {result['polygon']}")

    if args.output and result["polygon"]:
        write_piste_file(result, args.output, args.video)
        print(f"Written -> {args.output}")
    elif args.output:
        print("Nothing written: no polygon was derived.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
