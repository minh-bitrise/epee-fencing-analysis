"""
Check the report's mechanically verifiable claims against the code and results.

WHY THIS EXISTS. The report is written as development proceeds, so its numbers
are snapshots of a system that kept moving. Three of them had already gone stale
without anyone noticing: the test count drifted three times, Appendix B
described a CSV schema the pipeline no longer produced, and Appendix A listed
the modules as they were several weeks earlier. None of those are hard to spot
once you look; the problem is that nothing made anyone look.

WHAT IT CAN AND CANNOT CHECK. Only claims with a single authoritative source in
the repository: test counts against a live run, constants against the module
that defines them, the CSV schema against the writer, figure references against
the filesystem, figure numbering against document order, and the model results
against the JSON the evaluation wrote. It cannot check a claim about what the
evidence MEANS, which is most of the report and all of the interesting part.
A clean run says the report is not stale; it says nothing about whether it is
right.

Run before submitting, and after any change to the pipeline:
    python3 final/check_claims.py
    python3 final/check_claims.py --skip-tests    (fast, no suites run)
"""

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = ["final/submission_report_full.md"]


def read(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


class Checks:
    def __init__(self):
        self.failures, self.passes, self.skipped = [], 0, 0

    def ok(self, label):
        self.passes += 1
        print(f"  ok    {label}")

    def fail(self, label, detail):
        self.failures.append((label, detail))
        print(f"  FAIL  {label}: {detail}")

    def skip(self, label, why):
        self.skipped += 1
        print(f"  skip  {label} ({why})")

    def equal(self, label, got, want):
        if got == want:
            self.ok(f"{label} = {want}")
        else:
            self.fail(label, f"report says {want}, actual is {got}")


def count_tests(cmd, cwd, pattern):
    try:
        out = subprocess.run(cmd, cwd=os.path.join(ROOT, cwd), shell=True,
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(pattern, out.stdout + out.stderr)
    return int(m.group(1)) if m else None


def check_test_counts(c, report):
    m = re.search(r"supported by ([\d,]+) automated tests: ([\d,]+) over the pipeline, "
                  r"([\d,]+) over the application\s+layer and ([\d,]+) over the interface",
                  report)
    if not m:
        c.fail("test counts", "the sentence stating them was not found; has it been reworded?")
        return {}
    total, pipe, back, front = (int(x.replace(",", "")) for x in m.groups())
    if pipe + back + front != total:
        c.fail("test counts add up", f"{pipe}+{back}+{front} = {pipe+back+front}, not {total}")
    else:
        c.ok(f"test counts add up to {total}")

    actual = {
        "pipeline": (count_tests("python3 -m pytest -q", "code/prototype",
                                 r"(\d+) passed"), pipe),
        "backend": (count_tests("python3 -m pytest -q", "code/backend",
                                r"(\d+) passed"), back),
        "frontend": (count_tests("npx vitest run", "code/frontend",
                                 r"Tests\s+(\d+) passed"), front),
    }
    for name, (got, want) in actual.items():
        if got is None:
            c.skip(f"{name} test count", "suite did not report a count")
        else:
            c.equal(f"{name} tests", got, want)
    return {k: v[0] for k, v in actual.items()}


def check_readme_counts(c, counted):
    """The README quotes its own test counts, and they drift too.

    It had 320, 163 and 41 against actuals of 426, 185 and 126, which is the
    first file anyone opens. `counted` is what the live run found, passed in so
    the suites are not run twice.
    """
    readme = read("README.md")
    pairs = [("pipeline", r"pytest -q\s+# pipeline: (\d+)"),
             ("backend", r'not slow"\s+# API: (\d+)'),
             ("frontend", r"npm test\s+# interface: (\d+)")]
    for name, pattern in pairs:
        m = re.search(pattern, readme)
        if not m:
            c.skip(f"README {name} count", "not found; has the block been reworded?")
            continue
        got = counted.get(name)
        if got is None:
            c.skip(f"README {name} count", "no live count to compare against")
        elif name == "backend":
            # The README quotes the fast run, which deselects the slow tests.
            if abs(got - int(m.group(1))) > 3:
                c.fail(f"README {name} count",
                       f"README says {m.group(1)}, full run is {got}; the gap should be "
                       f"only the slow tests")
            else:
                c.ok(f"README {name} count {m.group(1)}, full run {got}")
        else:
            c.equal(f"README {name} count", got, int(m.group(1)))


def check_evaluation_set(c):
    """Every ground-truth file must be reachable, and every group real.

    The failure this prevents: labelling a new clip, dropping the CSV in, and
    having every evaluation silently ignore it because nothing maps the clip
    name to that file. Nothing errors in that case. The numbers just quietly
    stay as they were.
    """
    proto = os.path.join(ROOT, "code/prototype")
    sys.path.insert(0, proto)
    try:
        import clips as clipset
    except Exception as e:                                  # noqa: BLE001
        c.skip("evaluation set", f"clips.py did not import: {e}")
        return

    on_disk = {f for f in os.listdir(os.path.join(proto, "ground_truth"))
               if f.endswith("_touches.csv")}
    mapped = {os.path.basename(v) for v in clipset.CLIPS.values()}
    orphans = sorted(on_disk - mapped)
    if orphans:
        c.fail("evaluation set",
               f"labelled but unreachable, nothing in clips.py maps to them: {orphans}")
    else:
        c.ok(f"{len(on_disk)} label files, all reachable from clips.py")

    unknown = sorted(set(clipset.GROUP) - set(clipset.CLIPS))
    if unknown:
        c.fail("evaluation set", f"GROUP names unknown clips: {unknown}")
    else:
        labelled = [k for k, v in clipset.CLIPS.items()
                    if os.path.exists(os.path.join(proto, v))]
        c.ok(f"{len(labelled)} clips labelled, "
             f"{len(clipset.groups(labelled))} independent recordings")


def check_readme_commands(c):
    """Every script the README tells the reader to run must exist."""
    readme = read("README.md")
    missing = [n for n in sorted(set(re.findall(r"python3 (\w+)\.py", readme)))
               if not os.path.exists(os.path.join(ROOT, "code/prototype", f"{n}.py"))]
    if missing:
        c.fail("README commands", f"documented but absent: {missing}")
    else:
        c.ok("every script the README names exists")


def check_constants(c, report):
    """Numbers the report quotes that a module defines."""
    sys.path.insert(0, os.path.join(ROOT, "code", "prototype"))
    try:
        import detect_touches as dt
    except Exception as e:                                  # noqa: BLE001
        c.skip("detector constants", f"import failed: {e}")
        return
    pairs = [
        ("prominence", dt.MIN_PROMINENCE_M,
         r"distance minimum has (?:at least )?([\d.]+) m of\s+prominence"),
        ("separation", dt.MIN_SEPARATION_M,
         r"fencers then separate by (?:at least )?([\d.]+) m"),
    ]
    for label, value, pattern in pairs:
        m = re.search(pattern, report)
        if not m:
            c.skip(f"{label} constant", "not quoted in this document")
            continue
        c.equal(f"{label} constant", value, float(m.group(1)))


def check_csv_schema(c, report):
    """Appendix B against the columns run_detection actually writes."""
    src = read("code/prototype/run_detection.py")
    m = re.search(r"csv\.DictWriter\(f, fieldnames=\[(.*?)\]\)", src, re.S)
    if not m:
        c.skip("CSV schema", "could not locate the writer's fieldnames")
        return
    actual = set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    m2 = re.search(r"The per-frame CSV has columns (.*?)\.\n", report, re.S)
    if not m2:
        c.fail("CSV schema", "Appendix B's column list was not found")
        return
    documented = set(re.findall(r"`([a-z0-9_]+)`", m2.group(1)))
    documented -= {"pose", "bbox"}          # the `method` values, not columns
    missing = actual - documented
    extra = documented - actual
    if missing or extra:
        c.fail("CSV schema", f"undocumented: {sorted(missing) or 'none'}; "
                             f"documented but not written: {sorted(extra) or 'none'}")
    else:
        c.ok(f"CSV schema, {len(actual)} columns")


def check_figures(c, report, path):
    refs = re.findall(r"\]\((figures/[\w.]+)\)", report)
    for r in sorted(set(refs)):
        if not os.path.exists(os.path.join(ROOT, "final", r)):
            c.fail("figure file", f"{r} is referenced by {path} and does not exist")
    if refs:
        c.ok(f"{len(set(refs))} figure files exist")

    # Numbering must follow document order, or a cross-reference in the prose
    # points at the wrong picture and nothing warns.
    nums = re.findall(r"!\[\*\*Figure (\d+)\.(\d+):", report)
    by_chapter = {}
    for ch, n in nums:
        by_chapter.setdefault(ch, []).append(int(n))
    for ch, seq in by_chapter.items():
        if seq != sorted(seq) or seq != list(range(1, len(seq) + 1)):
            c.fail("figure numbering", f"chapter {ch} figures appear as {seq}")
        else:
            c.ok(f"chapter {ch} figures numbered {seq}")

    # Every "Figure N.M" mentioned in prose must exist as a caption.
    captions = {f"{a}.{b}" for a, b in nums}
    for a, b in re.findall(r"Figure (\d+)\.(\d+)(?!:)", report):
        if f"{a}.{b}" not in captions:
            c.fail("figure reference", f"prose cites Figure {a}.{b}, which has no caption")


def check_model_results(c, report):
    """Chapter 5's model comparison against the JSON the evaluation wrote."""
    p = os.path.join(ROOT, "code/prototype/results_current/touch_model_eval.json")
    if not os.path.exists(p):
        c.skip("model results", "touch_model_eval.json not present")
        return
    with open(p) as f:
        d = json.load(f)
    # Matches the figures wherever the sentence around them is reworded. The
    # first version keyed on the exact phrase "The rule wins", which stopped
    # matching the moment the rule stopped clearly winning, and the check then
    # SKIPPED rather than failing: it reported nothing wrong about a passage
    # whose numbers were entirely stale.
    m = re.search(r"F1 ([\d.]+) against ([\d.]+) for gradient boosted\s+"
                  r"trees and ([\d.]+) for logistic", report)
    if not m:
        c.skip("model results", "the sentence stating them was not found")
        return
    want = [float(x) for x in m.groups()]
    got = [round(d[k]["pooled"]["f1"], 2) for k in ("rule", "boosted", "logistic")]
    c.equal("model F1 (rule, boosted, logistic)", got, want)

    sep = round(d["ablation"]["separation"] - d["ablation"]["base"], 2)
    # Case-insensitive because the two documents word this sentence differently,
    # and an EXPLICIT skip because the first version used a bare `if` and
    # silently checked nothing on one of them. A check that quietly does nothing
    # is worse than no check: it reports a pass it never performed.
    m2 = re.search(r"[Ss]eparation now costs ([\d.]+) when\s+removed", report)
    if m2:
        c.equal("separation ablation", abs(sep), float(m2.group(1)))
    else:
        c.skip("separation ablation", "the sentence stating it was not found")


# submission_report_full.md is the WORKING document and is deliberately over the
# 9,500 total, because that cap is not established (TODO C-CAP). Failing on it
# every run would train the reader to ignore this tool, which is the only way a
# checker like this actually goes wrong.
OVER_CAP_BY_DESIGN = {"final/submission_report_full.md"}

# There is no longer a derived snapshot: the real limit turned out to be 10,500,
# not the 9,500 that was assumed, so the cut-down version was deleted rather than
# maintained. These constants stay because the staleness check below is cheap and
# the situation could recur.
WORKING = "final/submission_report_full.md"
DERIVED = "final/submission_report.md"


def last_commit(path):
    """Unix time of the last commit touching `path`, or None."""
    out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", path],
                         cwd=ROOT, capture_output=True, text=True)
    t = out.stdout.strip()
    return int(t) if t.isdigit() else None


def check_derived_is_current(c):
    """Warn when the derived snapshot is behind the working document."""
    if not all(os.path.exists(os.path.join(ROOT, p)) for p in (WORKING, DERIVED)):
        c.skip("derived snapshot", "only one report document is present")
        return
    w, d = last_commit(WORKING), last_commit(DERIVED)
    if w is None or d is None:
        c.skip("derived snapshot", "no commit history for one of the documents")
        return
    if w > d:
        days = (w - d) / 86400.0
        c.fail("derived snapshot",
               f"{DERIVED} was last committed {days:.1f} days before {WORKING}. It is a "
               f"snapshot, not a second copy: re-derive it before submitting anything from "
               f"it, or delete it if the word cap turns out not to exist (TODO C-CAP)")
    else:
        c.ok("derived snapshot is level with the working document")


def check_word_count(c, path):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "final/wordcount.py"),
                          os.path.join(ROOT, path)],
                         capture_output=True, text=True)
    m = re.search(r"TOTAL\s+(\d+)\s+(\d+)", out.stdout)
    if not m:
        c.skip("word count", "wordcount.py produced no total")
        return
    total, cap = int(m.group(1)), int(m.group(2))
    if path in OVER_CAP_BY_DESIGN:
        c.ok(f"word count {total}, over {cap} by design (see TODO C-CAP)")
    elif total > cap:
        c.fail("word count", f"{total} against a cap of {cap}. NOTE: the cap itself is "
                             f"unverified, see TODO C-CAP")
    else:
        c.ok(f"word count {total} of {cap}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the three suites")
    args = ap.parse_args(argv)

    c = Checks()
    for path in REPORTS:
        if not os.path.exists(os.path.join(ROOT, path)):
            continue
        print(f"\n{path}")
        report = read(path)
        check_figures(c, report, path)
        check_constants(c, report)
        check_csv_schema(c, report)
        check_model_results(c, report)
        check_word_count(c, path)
    print("\nREADME")
    check_readme_commands(c)
    print("\ndocuments")
    check_derived_is_current(c)
    print("\nevaluation set")
    check_evaluation_set(c)

    if args.skip_tests:
        print("\ntest counts: skipped")
    else:
        print("\ntest counts (running the suites, this takes a minute)")
        # Counted once: both documents state the same numbers, and checking the
        # first document that states them is enough.
        counted = {}
        for path in REPORTS:
            report = read(path)
            if "automated tests" in report:
                counted = check_test_counts(c, report)
                break
        check_readme_counts(c, counted or {})

    print(f"\n{c.passes} passed, {len(c.failures)} failed, {c.skipped} skipped")
    for label, detail in c.failures:
        print(f"  {label}: {detail}")
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
