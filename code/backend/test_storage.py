"""
Tests for disk accounting and cleanup.

The whole risk in this module is deleting something that cannot be regenerated,
so most of these assert what is left ALONE rather than what is removed. The
annotations in particular are the user's own labelling: 36 hand-marked lunges on
one clip and 27 hand-labelled touches across four, none of which any amount of
reprocessing would bring back.

Run with:
    cd code/backend && python3 -m pytest test_storage.py -v
"""

import os
import time

import pytest

from storage import clean, find_orphan_transcodes, stale_jobs, storage_report


@pytest.fixture
def layout(tmp_path):
    """A results directory, a cache beside it, and things that must survive."""
    results = tmp_path / "results_current"
    results.mkdir()
    (results / "bout_annotated.mp4").write_bytes(b"annotated video" * 100)
    (results / "bout_distance.csv").write_text("frame,time_s\n0,0.0\n")

    cache = tmp_path / "web_video"
    cache.mkdir()
    live = cache / "results_current__bout_annotated.h264.mp4"
    live.write_bytes(b"transcoded" * 100)
    # make the cache newer than its source, which is the healthy case
    os.utime(live, (time.time() + 10, time.time() + 10))

    orphan = cache / "results_gone__missing_annotated.h264.mp4"
    orphan.write_bytes(b"orphan" * 100)

    return {"root": tmp_path, "results": results, "cache": cache,
            "live": live, "orphan": orphan}


class TestFindOrphans:
    def test_finds_a_transcode_whose_source_is_gone(self, layout):
        found = find_orphan_transcodes(str(layout["cache"]),
                                       [str(layout["results"])])
        assert str(layout["orphan"]) in found

    def test_leaves_a_current_transcode_alone(self, layout):
        found = find_orphan_transcodes(str(layout["cache"]),
                                       [str(layout["results"])])
        assert str(layout["live"]) not in found

    def test_finds_a_transcode_older_than_its_source(self, layout):
        """
        A stale entry will be re-encoded on next view anyway, so it is occupying
        disk to no end. Regenerating the reference results left 64 MB of exactly
        this behind.
        """
        os.utime(layout["live"], (1, 1))       # cache far older than source
        found = find_orphan_transcodes(str(layout["cache"]),
                                       [str(layout["results"])])
        assert str(layout["live"]) in found

    def test_ignores_files_it_did_not_write(self, layout):
        # A name without the `dir__stem` encoding did not come from the
        # transcoder, and guessing at it is how a cleanup deletes something else.
        stranger = layout["cache"] / "somebody-elses-file.h264.mp4"
        stranger.write_bytes(b"x")
        found = find_orphan_transcodes(str(layout["cache"]),
                                       [str(layout["results"])])
        assert str(stranger) not in found

    def test_a_missing_cache_directory_is_not_an_error(self, tmp_path):
        assert find_orphan_transcodes(str(tmp_path / "nope"), []) == []


class TestClean:
    def test_removes_orphans_and_keeps_the_live_cache_by_default(self, layout):
        result = clean(str(layout["cache"]), [str(layout["results"])])
        assert result["removed"] == 1
        assert not layout["orphan"].exists()
        assert layout["live"].exists(), "a serviceable transcode was deleted"

    def test_clears_everything_only_when_asked(self, layout):
        result = clean(str(layout["cache"]), [str(layout["results"])],
                       orphans_only=False)
        assert result["removed"] == 2
        assert not layout["live"].exists()

    def test_never_touches_the_source_video_or_the_results(self, layout):
        """
        The transcodes are derived and the annotated videos are what they are
        derived FROM. Deleting a source would turn a few seconds of re-encoding
        into a full reprocessing run.
        """
        clean(str(layout["cache"]), [str(layout["results"])], orphans_only=False)
        assert (layout["results"] / "bout_annotated.mp4").exists()
        assert (layout["results"] / "bout_distance.csv").exists()

    def test_reports_what_it_freed(self, layout):
        # A silent success gives the user no way to tell "nothing needed doing"
        # from "the button is broken".
        result = clean(str(layout["cache"]), [str(layout["results"])])
        assert result["freed_mb"] >= 0
        assert result["files"], "removed files should be named"


class TestStorageReport:
    def test_separates_reclaimable_from_evidence(self, layout, tmp_path):
        report = storage_report(
            str(layout["cache"]), [str(layout["results"])],
            str(tmp_path / "uploads"), str(tmp_path / "jobs"),
            str(tmp_path / "results_uploads"))
        cache = next(c for c in report["categories"]
                     if c["name"] == "browser video cache")
        assert cache["reclaimable"] is True
        # Everything else must be marked as not reclaimable: uploads are the
        # user's only copy of their footage, and results are what the report
        # quotes.
        assert all(not c["reclaimable"] for c in report["categories"]
                   if c["name"] != "browser video cache")

    def test_counts_the_reclaimable_files(self, layout, tmp_path):
        report = storage_report(
            str(layout["cache"]), [str(layout["results"])],
            str(tmp_path / "u"), str(tmp_path / "j"), str(tmp_path / "r"))
        assert report["reclaimable_files"] == 1
        assert report["whole_cache_mb"] >= report["reclaimable_mb"]


class TestStaleJobs:
    def test_reports_old_finished_jobs_without_removing_them(self, tmp_path):
        """
        Reported and never removed. A job record is the only account of what was
        run and with which settings, it costs a few hundred bytes, and this
        project's evaluation depends on being able to say how a result was
        produced.
        """
        from jobs import JobStore
        store = JobStore(str(tmp_path / "jobs"))
        old = store.create(filename="old.mp4")
        store.update(old["job_id"], state="done",
                     finished_at=time.time() - 60 * 86400)
        recent = store.create(filename="new.mp4")
        store.update(recent["job_id"], state="done", finished_at=time.time())

        found = stale_jobs(store, older_than_days=30)
        assert [j["job_id"] for j in found] == [old["job_id"]]
        assert store.load(old["job_id"]) is not None, "it was removed, not reported"

    def test_a_job_that_never_finished_is_not_stale(self, tmp_path):
        from jobs import JobStore
        store = JobStore(str(tmp_path / "jobs"))
        store.create(filename="running.mp4", state="running")
        assert stale_jobs(store, older_than_days=0) == []
