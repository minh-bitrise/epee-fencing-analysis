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
    "4. Implementation": 2000,
    "5. Evaluation": 2500,
    "6. Conclusion": 1000,
}
TOTAL_CAP = 9500

# Not counted: everything after this heading, plus the two below it.
UNCOUNTED = ("References", "Appendices")


def split_chapters(text):
    out, current, buf = {}, None, []
    for line in text.split("\n"):
        m = re.match(r"^## (.+?)\s*$", line)
        if m:
            if current:
                out[current] = "\n".join(buf)
            current, buf = m.group(1), []
            continue
        if current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf)
    return out


def counts(body):
    raw, prose, in_code = 0, 0, False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        n = len(stripped.split())
        raw += n
        # Excluded from the counted total: tables, code, and figure captions.
        if in_code or stripped.startswith("|") or stripped.startswith("!["):
            continue
        prose += n
    return raw, prose


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
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "final/final_report.md"))
