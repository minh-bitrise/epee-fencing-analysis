"""Tests for the evaluation set and its grouping.

The grouping is the whole point of this module: get it wrong and a model is
tested on the recording it was trained on, which produces a better number and no
error message.
"""

import clips


def test_an_ungrouped_clip_is_its_own_recording():
    assert clips.group_of("fencing_clip3") == "fencing_clip3"


def test_two_windows_of_one_video_share_a_group():
    assert clips.group_of("fencing_clip7a") == clips.group_of("fencing_clip7b")


def test_groups_counts_recordings_not_files():
    names = ["fencing_clip3", "fencing_clip7a", "fencing_clip7b"]
    assert len(clips.groups(names)) == 2


def test_folds_never_split_one_recording_across_the_divide():
    """The test this module exists for. Holding out 7a while training on 7b
    means training and testing on the same camera, venue and pair of fencers."""
    names = ["fencing_clip", "fencing_clip3", "fencing_clip7a", "fencing_clip7b"]
    for test, train in clips.folds(names):
        assert not (set(test) & set(train))
        overlap = {clips.group_of(c) for c in test} & {clips.group_of(c) for c in train}
        assert not overlap, f"recording on both sides: {overlap}"


def test_a_grouped_pair_is_held_out_together():
    names = ["fencing_clip3", "fencing_clip7a", "fencing_clip7b"]
    held = [sorted(t) for t, _ in clips.folds(names)]
    assert ["fencing_clip7a", "fencing_clip7b"] in held


def test_folds_skips_a_split_with_nothing_to_train_on():
    assert list(clips.folds(["fencing_clip7a", "fencing_clip7b"])) == []


def test_every_clip_has_a_labels_path():
    for name in clips.CLIPS:
        assert clips.CLIPS[name].startswith("ground_truth/")


def test_every_grouped_clip_is_a_known_clip():
    """A typo in GROUP would silently group nothing."""
    for name in clips.GROUP:
        assert name in clips.CLIPS
