"""
Count the words the submission is judged on, per chapter.

WHY THIS EXISTS RATHER THAN `wc -w`. The brief excludes diagrams, figures,
tables, references and the title page from the count. A raw word count over the
markdown therefore overstates the total by everything in the evidence tables,
which on this report is substantial. It also understates the risk of the
opposite mistake: quietly assuming an exclusion the brief does not grant.

So both numbers are reported. `prose` is what the caps are checked against;
`raw` is what a reader counting naively would get, and the gap between them is
the amount of the report that rests on the exclusion being real.
"""
import re
import sys

CAPS = {
    "1. Introduction": 1000,
    "2. Literature Review": 2500,
    "3. Design": 2000,
    "4. Implementation": 2500,
    "5. Evaluation": 2500,
    "6. Conclusion": 1000,
}
# Verified 18 Sep 2026 against the module's final report instructions, the
# coursework page itself. The per-chapter limits sum to 11,500; the total is the
# strict one and the spread between chapters is deliberately flexible.
TOTAL_CAP = 10500

# Not counted: everything after this heading, plus the two below it.
UNCOUNTED = ("References", "Appendices")


# The brief requires each chapter title to carry its own word count, as in
# "1. Introduction (783/1000 words)". That makes the heading a number that goes
# stale on every edit, in the most visible place in the document, so it is
# written by --stamp rather than by hand and checked by check_claims.py.
STAMP = re.compile(r"\s*\(\s*[\d,]+\s*/\s*[\d,]+\s*words\s*\)\s*$", re.I)


def bare(title):
    """A chapter title without any stamped word count."""
    return STAMP.sub("", title).strip()


def split_chapters(text):
    out, current, buf = {}, None, []
    for line in text.split("\n"):
        m = re.match(r"^## (.+?)\s*$", line)
        if m:
            if current:
                out[current] = "\n".join(buf)
            current, buf = bare(m.group(1)), []
            continue
        if current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf)
    return out


def counts(body):
    """Raw words, and the words the caps are checked against.

    CAPTIONS RUN OVER SEVERAL LINES, which an earlier version of this got wrong.
    It excluded any line STARTING with "![", so the first line of each caption
    was excluded and its continuation lines were counted as prose. Every figure
    in the report therefore charged about thirty words of its own caption
    against the chapter that held it. A caption block is excluded from the line
    that opens it through the line that closes the markdown image.
    """
    raw, prose, in_code, in_caption = 0, 0, False, False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        n = len(stripped.split())
        raw += n
        was_caption = in_caption
        if not in_caption and stripped.startswith("!["):
            in_caption = True
            was_caption = True
        # A blank line also closes it. Without that guard a caption whose image
        # link was malformed would silently exclude the rest of the chapter,
        # which is the failure mode that matters: it under-counts, so nothing
        # would look wrong.
        if in_caption and (not stripped or ("](" in stripped and ")" in stripped)):
            in_caption = False
        # Excluded from the counted total: tables, code, and figure captions.
        if in_code or was_caption or stripped.startswith("|"):
            continue
        prose += n
    return raw, prose


def stamp(path):
    """Write each chapter's word count into its own heading, as the brief asks.

    Rewrites rather than appends, so running it twice does not produce
    "(985/1000 words) (985/1000 words)". Only chapters with a cap are stamped:
    the appendices and references have no limit to state.
    """
    text = open(path).read()
    chapters = split_chapters(text)
    out = []
    for line in text.split("\n"):
        m = re.match(r"^## (.+?)\s*$", line)
        if m:
            name = bare(m.group(1))
            if name in CAPS and chapters.get(name) is not None:
                prose = counts(chapters[name])[1]
                line = f"## {name} ({prose}/{CAPS[name]} words)"
        out.append(line)
    open(path, "w").write("\n".join(out))
    print(f"stamped word counts into the chapter headings of {path}")
    return 0


def main(path):
    chapters = split_chapters(open(path).read())
    total_prose = total_raw = 0
    print(f"{'chapter':<26}{'prose':>8}{'cap':>8}{'over':>8}{'raw':>8}")
    for name, cap in CAPS.items():
        body = chapters.get(name)
        if body is None:
            print(f"{name:<26}{'MISSING':>8}")
            continue
        raw, prose = counts(body)
        total_prose += prose
        total_raw += raw
        over = prose - cap
        print(f"{name:<26}{prose:>8}{cap:>8}{over:>+8}{raw:>8}")
    print(f"{'':<26}{'-' * 32}")
    print(f"{'TOTAL':<26}{total_prose:>8}{TOTAL_CAP:>8}"
          f"{total_prose - TOTAL_CAP:>+8}{total_raw:>8}")
    uncounted = sum(counts(chapters.get(k, ""))[1] for k in UNCOUNTED)
    print(f"{'(appendices, uncounted)':<26}{uncounted:>8}")
    return 0 if total_prose <= TOTAL_CAP else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--stamp"]
    target = args[0] if args else "final/final_report.md"
    sys.exit(stamp(target) if "--stamp" in sys.argv else main(target))
