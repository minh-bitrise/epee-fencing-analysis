"""
Report where the disk went, and reclaim the part of it that is derived.

Every bout ever viewed leaves a transcoded copy of its annotated video behind,
because OpenCV writes MPEG-4 Part 2 and browsers refuse to decode it. On the
development machine that cache reached 313 MB with no way to reclaim it short of
a terminal. Only derived files are removable here; source uploads and annotations
are not.
"""

import os
import time


def _dir_size(path):
    """Total bytes under a directory, missing directories counting as zero."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # A file removed between listing and sizing is not an error; it
                # is one fewer byte in use, which is what the caller asked about.
                continue
    return total


def _mb(n):
    return round(n / (1024 * 1024), 1)


def find_orphan_transcodes(web_video_dir, results_dirs):
    """Cached transcodes that can never be served, or will be discarded anyway.

    Two kinds, both pure waste. An ORPHAN has lost the annotated video it was
    made from, so it can never be served and can never be regenerated.
    """
    orphans = []
    if not os.path.isdir(web_video_dir):
        return orphans
    by_name = {os.path.basename(d.rstrip("/")): d for d in results_dirs}
    for name in sorted(os.listdir(web_video_dir)):
        if not name.endswith(".h264.mp4"):
            continue
        stem = name[: -len(".h264.mp4")]
        if "__" not in stem:
            # Not a name this module wrote. Left alone rather than guessed at.
            continue
        parent, base = stem.split("__", 1)
        source_dir = by_name.get(parent)
        path = os.path.join(web_video_dir, name)
        if source_dir is None:
            orphans.append(path)
            continue
        source = os.path.join(source_dir, f"{base}.mp4")
        if not os.path.exists(source):
            orphans.append(path)
            continue
        # STALE counts too. A cache older than the video it was made from is
        # already going to be re-encoded on next view, because the serving code
        # compares modification times, so it is occupying disk to no end. This
        # is not hypothetical: regenerating the reference results left four
        # stale transcodes of about 60 MB each behind.
        try:
            if os.path.getmtime(path) < os.path.getmtime(source):
                orphans.append(path)
        except OSError:
            continue
    return orphans


def storage_report(web_video_dir, results_dirs, upload_root, job_root,
                   upload_results_root):
    """
    Where the disk went, and how much of it is derived rather than evidence.

    Reported per category rather than as one number, because the answer a user
    needs is not "how much" but "what can go": the cache is regenerable, the
    uploads are their footage, and the results are what the report cites.
    """
    cache = _dir_size(web_video_dir)
    orphans = find_orphan_transcodes(web_video_dir, results_dirs)
    orphan_bytes = sum(os.path.getsize(p) for p in orphans
                       if os.path.exists(p))
    return {
        "categories": [
            {"name": "browser video cache", "mb": _mb(cache),
             "reclaimable": True,
             "note": "regenerated on next view, a few seconds per bout"},
            {"name": "uploaded source video", "mb": _mb(upload_root and
                                                        _dir_size(upload_root)),
             "reclaimable": False,
             "note": "the only copy of what the user provided"},
            {"name": "processing output for uploads", "mb":
                _mb(_dir_size(upload_results_root)),
             "reclaimable": False,
             "note": "delete the bout to remove it"},
            {"name": "job records and logs", "mb": _mb(_dir_size(job_root)),
             "reclaimable": False, "note": "small, and the record of what ran"},
        ],
        "reclaimable_files": len(orphans),
        "reclaimable_mb": _mb(orphan_bytes),
        "whole_cache_mb": _mb(cache),
    }


def clean(web_video_dir, results_dirs, orphans_only=True):
    """Remove cached transcodes. Orphans by default, the whole cache on request.

    Defaulting to orphans is the meaningful default rather than the timid one:
    an orphan costs nothing to remove because it can never be served, whereas
    clearing the live cache costs a re-encode the next time somebody opens each
    bout.
    """
    if orphans_only:
        targets = find_orphan_transcodes(web_video_dir, results_dirs)
    elif os.path.isdir(web_video_dir):
        targets = [os.path.join(web_video_dir, n)
                   for n in os.listdir(web_video_dir)
                   if n.endswith(".h264.mp4")]
    else:
        targets = []

    removed, freed = [], 0
    for path in targets:
        try:
            freed += os.path.getsize(path)
            os.remove(path)
            removed.append(os.path.basename(path))
        except OSError:
            continue
    return {"removed": len(removed), "freed_mb": _mb(freed),
            "files": removed[:20], "orphans_only": orphans_only}


def stale_jobs(job_store, older_than_days=30):
    """Finished job records older than a cutoff, as candidates for removal.

    Reported and never removed automatically. A job record is the only account
    of what was run and with which settings, it is a few hundred bytes, and
    this project's evaluation depends on being able to say how a result was
    produced.
    """
    cutoff = time.time() - older_than_days * 86400
    out = []
    for job in job_store.list():
        finished = job.get("finished_at")
        if finished and finished < cutoff:
            out.append({"job_id": job["job_id"],
                        "filename": job.get("filename"),
                        "state": job["state"],
                        "age_days": round((time.time() - finished) / 86400)})
    return out
