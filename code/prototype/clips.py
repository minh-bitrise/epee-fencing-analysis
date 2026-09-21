"""
The evaluation set: which clips exist, where their labels are, and which of them
are the same recording.

One definition, because the mapping was written out twice and adding footage
meant editing both. Groups are not optional: 7a and 7b are two windows of ONE
video, so holding them out separately would test a model on its own training
data and inflate every figure quietly.
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
