"""
Epee Fencing Bout Analysis - Piste Region Derivation
====================================================
Measure a piste polygon from a video instead of drawing one by hand.

WHY THIS EXISTS. Competition footage contains people who are not the two
fencers: a referee, spectators, and fencers on the adjacent strip. The detector
knows what a person is and not what a fencer is, so without a spatial filter it
will happily lock a tracker slot onto the referee. `PisteRegion` in
run_detection.py is that filter, and it is loaded from a small JSON polygon file
authored once per clip. Every clip in the evaluation has one. A video uploaded
through the web interface has nothing, and the gap is not cosmetic: rebuilding
the reference results without the configs dropped clip 2 from 98.0 per cent
coverage to 80.1.

WHY IT MEASURES RATHER THAN ASKS. The obvious answer is to let the user draw the
strip over the first frame. That is exactly the approach that already failed on
this project. The first clip-2 polygon was placed by eye with its top edge at
y=230, which let the adjacent piste in and *raised* the count of physically
implausible distance samples from 53 to 252, while the headline coverage number
went up. The polygons that work were arrived at differently: sample every tenth
frame, take the two tallest person detections in each, look at where their feet
actually fall, and put the boundary just outside that band with a margin. Clip
2's top edge of 380 comes from a measured 1st-percentile feet-y of 402; clip 4's
260 comes from a measured cluster boundary. This module automates that
procedure, so the interface asks the user to confirm a measurement rather than to
produce a guess.

WHAT IT CANNOT DO. It separates people by where they stand, so it cannot reject a
referee standing on the strip, and it assumes the two tallest detections are
usually the fencers. That assumption is the same one `calibrate_fixed_scale`
already makes and it holds because the fencers are nearest the camera, but it
degrades when a bystander is nearer still. The derived polygon is therefore
reported with the measurements behind it and the caller is expected to show them,
not to treat the output as authoritative.

Run standalone with:
    python3 derive_piste.py --video fencing_clip2.mp4 --output piste_derived.json
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
# belonging to different groups of people. Measured: on the broadcast clip the
# spectator band ends at y=236 and the fencer band starts at y=400, a gap of 164
# px against a 252 px fencer, while the widest gap inside the fencer band is
# under 50. Any threshold between those two separates them, and half a fencer
# sits in the middle of that range.
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
    """
    Pick the two boxes in one frame most likely to be the fencers.

    NOT the two tallest, which is what `calibrate_fixed_scale` uses and what the
    first version of this module used. Measurement shows why: on the broadcast
    clip the two tallest boxes have a median vertical separation of 296 px,
    because the referee stands nearest the camera and is therefore the tallest
    box in the frame. Taking him as a fencer put the measured band's 99th
    percentile at y=718 and produced a polygon covering the whole frame, which
    filters nothing. The pipeline's own documented failure mode is a bystander
    being accepted into a tracked slot, so a piste filter derived by admitting
    the referee would be worse than useless.

    Two fencers stand on one strip, so they are at almost the same depth and
    their feet sit at almost the same height in the image. Choosing the closest
    pair by feet-y drops that median separation from 296 px to 7. The height
    ratio guard stops the rule pairing two distant spectators, who are close
    together in feet-y but much shorter than anyone on the strip.

    Returns (box_a, box_b) or None when the frame has fewer than two candidates.
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
    """
    Split a sorted set of values wherever consecutive ones differ by more than
    `gap`, and return the groups largest first.

    A percentile taken across a bimodal distribution describes neither mode. On
    the broadcast clip roughly a tenth of frames contribute a spectator pair at
    y around 220 and the rest contribute fencers at 400 to 520, so the 1st
    percentile of everything lands at 214, in the spectators, and a band built
    from it admits them. Separating the groups first and describing only the
    largest is what the hand-authored polygons did by reading a histogram.
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
    """
    Measure a piste polygon for a video.

    Returns a dict describing the result rather than a bare polygon, because the
    caller has to be able to show the user what the measurement was. The
    `needed` flag distinguishes "no polygon, because the footage has no
    bystanders to reject" from "no polygon, because measurement failed", and
    those two call for opposite responses from the interface.
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

    # How many people the band will still let through.
    #
    # This is the check that matters, and coverage does not provide it. Measured
    # on clip 1, a derived band and the hand-authored one give almost the same
    # tracking coverage, 92.5 against 92.9 per cent, while physically impossible
    # distance readings go from 1 to 37 and the largest measured separation goes
    # from 6.44 m to 9.63 m. The cause is that the officials sit behind the far
    # end of the strip, so their feet land at y 430-460, inside a fencer band
    # that runs from 421 to 695 because the piste recedes from the camera. No
    # horizontal band can separate them: the strip itself spans that depth. The
    # hand-authored polygon resolved it by starting at y=470 and giving up the
    # far end of the strip, which is a trade requiring the knowledge that the
    # officials are there at all.
    #
    # So rather than guess, this counts the detections that fall inside the band
    # and were NOT chosen as part of a fencer pair, and hands the number to the
    # interface. A band the fencers share with nobody reports close to zero; one
    # that still contains bystanders says so, and the user can drag the edge.
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
    # call for different responses. A band covering the whole frame filters
    # nothing. A band that still contains other people filters the wrong thing.
    #
    # The 0.5 threshold rests on three clips and should be treated as
    # provisional. Measured, unselected people per sampled frame come out at 0.33
    # on clip 2, where the derived band matches the hand-authored one exactly,
    # and at 0.59 and 1.88 on clips 4 and 1, where the derived band is measurably
    # worse. It separates the cases available, and three points is not many. The
    # measurement itself is reported on every result regardless, and it is the
    # substance; this flag only decides whether to add a sentence explaining it.
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
    """
    Write a derived polygon in the same schema the hand-authored files use.

    The measurements are written into the file's comment. That is where the
    hand-authored files record their reasoning, and a derived file that did not
    would be indistinguishable from one placed by eye, which is the exact
    distinction this project has already paid to learn.
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
