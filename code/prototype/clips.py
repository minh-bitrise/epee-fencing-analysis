"""
The evaluation set: which clips exist, where their labels are, and which of them
are the same recording.

WHY THIS IS A MODULE AND NOT A DICT IN EACH SCRIPT. The mapping from clip to
ground-truth file was written out twice, in `touch_features.py` and in
`evaluate_pose.py`, and adding footage meant editing both. On a project whose
recurring defect is a fact stated in one place and contradicted in another, two
copies of the evaluation set is a defect waiting to be written.

WHY GROUPS EXIST, AND WHY THEY ARE NOT OPTIONAL. `fencing_clip7a` and
`fencing_clip7b` are two windows of ONE video: same camera, same venue, same
lighting, same two fencers. Held out separately in a leave-one-out protocol, a
model trains on one and is tested on the other, which is close to testing it on
its own training data and would inflate every figure quietly and plausibly.

The unit that generalisation is claimed over is the RECORDING, so the recording
is what must be held out. `GROUP` says which recording each clip came from, and
the cross-validation folds over groups rather than over files. A clip with no
entry is its own group, which is the right default: footage from separate
sources is independent unless something says otherwise.
"""

# clip basename -> its hand-labelled touches, relative to this directory
CLIPS = {
    "fencing_clip":   "ground_truth/fencing_clip1_touches.csv",
    "fencing_clip2":  "ground_truth/fencing_clip2_touches.csv",
    "fencing_clip3":  "ground_truth/fencing_clip3_touches.csv",
    "fencing_clip4":  "ground_truth/fencing_clip4_touches.csv",
    # Second batch, 17 Sep 2026. Labels do not exist yet; every consumer skips a
    # clip whose ground-truth file is absent, so these are harmless until they do.
    "fencing_clip5":  "ground_truth/fencing_clip5_touches.csv",
    "fencing_clip6":  "ground_truth/fencing_clip6_touches.csv",
    "fencing_clip7a": "ground_truth/fencing_clip7a_touches.csv",
    "fencing_clip7b": "ground_truth/fencing_clip7b_touches.csv",
    "fencing_clip8":  "ground_truth/fencing_clip8_touches.csv",
}

# clip basename -> the recording it came from. Absent means "its own recording".
GROUP = {
    "fencing_clip7a": "warofroses",
    "fencing_clip7b": "warofroses",
}


def group_of(clip):
    """The independent recording a clip belongs to."""
    return GROUP.get(clip, clip)


def groups(clips):
    """The distinct recordings covered by `clips`, in first-seen order."""
    out = []
    for c in clips:
        g = group_of(c)
        if g not in out:
            out.append(g)
    return out


def folds(clips):
    """Leave-one-RECORDING-out splits as (test_clips, train_clips) pairs.

    Yields one split per recording, not one per file, so two windows of the same
    video never land on opposite sides of a split.
    """
    for g in groups(clips):
        test = [c for c in clips if group_of(c) == g]
        train = [c for c in clips if group_of(c) != g]
        if test and train:
            yield test, train
