"""
Check that every number in a generated summary came from the payload it was given.

The language model is the one model here not scored against ground truth, and its
documented failure is fabrication: fluent prose carrying figures never in the
input. This checks GROUNDING, not truth. A summary whose every figure appears in
the payload can still mislead by selecting favourable numbers or asserting a
cause the data does not support.
"""
import argparse
import json
import os
import re

# Tolerance follows the precision the summary itself uses rather than being a
# fixed constant. A figure written "45.2" claims one decimal, so it matches a
# payload value within 0.05; one written "52" claims none, so it matches within
# 0.5. Choosing a single tolerance instead would either reject correct rounding
# or accept genuinely different numbers.
def tolerance_for(text):
    if "." in text:
        return 0.5 * 10 ** -len(text.split(".")[1])
    return 0.5


WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}

# Ordinals, fractions and vague quantifiers are not claims about the data.
# "the first half", "a third of the time", "both fencers": none of these assert
# a measured value, and counting them as unsupported would drown the signal.
IGNORE_WORDS = {"first", "second", "third", "half", "both", "one", "two"}


def key_numbers(payload):
    """Numbers appearing in payload KEY NAMES rather than values.

    The zone breakdown is keyed `lunge_distance_1.5_to_2.6m`, so the band
    boundaries a summary quotes are supplied by the payload but are not values
    in it.
    """
    out = set()
    if isinstance(payload, dict):
        for k, v in payload.items():
            for m in re.finditer(r"\d+(?:\.\d+)?", str(k)):
                out.add(float(m.group(0)))
            out |= key_numbers(v)
    elif isinstance(payload, list):
        for v in payload:
            out |= key_numbers(v)
    return out


def flatten(payload, prefix=""):
    """Every numeric value in the payload, keyed by its path."""
    out = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        out[prefix] = float(payload)
    return out


def extract_numbers(markdown):
    """
    Numeric claims in the summary, as (text, value) pairs.

    Written-out numbers are included because a model writes "fourteen touches"
    as readily as "14", and excluding them would let the most quotable claims
    through unchecked.
    """
    found = []
    for m in re.finditer(r"\d+(?:\.\d+)?", markdown):
        found.append((m.group(0), float(m.group(0))))
    for word, value in WORD_NUMBERS.items():
        if word in IGNORE_WORDS:
            continue
        for _ in re.finditer(rf"\b{word}\b", markdown, re.I):
            found.append((word, float(value)))
    return found


def derivable(value, values, tol, pairs=True):
    """Whether a simple relation between payload values produces this figure.

    The first version searched differences and sums over every pair, which
    seemed a conservative way to avoid false alarms.
    """
    vals = list(values)
    for a in vals:
        if abs(abs(a) - value) <= tol or abs(a * 100 - value) <= tol:
            return True
    if not pairs:
        return False
    for a in vals:
        for b in vals:
            if abs(abs(a - b) - value) <= tol or abs(a + b - value) <= tol:
                return True
    return False


def audit(markdown, payload, pairs=False):
    values = flatten(payload)
    pool = set(values.values()) | key_numbers(payload)
    grounded, derived, unsupported = [], [], []

    for text, value in extract_numbers(markdown):
        tol = tolerance_for(text)
        if any(abs(v - value) <= tol for v in pool):
            grounded.append((text, value))
        elif derivable(value, pool, tol, pairs=pairs):
            derived.append((text, value))
        else:
            unsupported.append((text, value))

    total = len(grounded) + len(derived) + len(unsupported)
    return {
        "figures": total,
        "grounded": grounded,
        "derived": derived,
        "unsupported": unsupported,
        "grounded_pct": round(100.0 * len(grounded) / total, 1) if total else None,
        "supported_pct": (round(100.0 * (len(grounded) + len(derived)) / total, 1)
                          if total else None),
        "payload_values": len(values),
    }


def sensitivity(payload, trials=200, seed=7, pairs=False):
    """What fraction of FABRICATED figures this audit would catch, by magnitude.

    A null result is only readable alongside what the method could have
    detected, and the same applies to a clean audit.
    """
    import random
    rng = random.Random(seed)
    values = flatten(payload)
    pool = set(values.values()) | key_numbers(payload)

    bands = {
        "small integers (0-20)": lambda: float(rng.randint(0, 20)),
        "larger integers (21-500)": lambda: float(rng.randint(21, 500)),
        "one decimal": lambda: round(rng.uniform(0, 100), 1),
        "two decimals": lambda: round(rng.uniform(0, 10), 2),
    }
    out = {}
    for name, draw in bands.items():
        caught = 0
        for _ in range(trials):
            v = draw()
            text = repr(v) if v % 1 else str(int(v))
            tol = tolerance_for(text)
            grounded = any(abs(p - v) <= tol for p in pool)
            if not grounded and not derivable(v, pool, tol, pairs=pairs):
                caught += 1
        out[name] = round(100.0 * caught / trials, 1)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Audit a generated summary against the payload it was given")
    p.add_argument("--summary", required=True, help="the .md the model produced")
    p.add_argument("--meta", default=None,
                   help="the .meta.json holding the payload; defaults to the "
                        "summary path with .meta.json substituted")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--pairs", action="store_true",
                   help="also treat a difference or sum of any two payload "
                        "values as a derivation. Measured to destroy "
                        "sensitivity; off by default.")
    p.add_argument("--sensitivity", action="store_true",
                   help="report what fraction of fabricated figures would be "
                        "caught, by magnitude")
    args = p.parse_args()

    meta_path = args.meta or args.summary.replace(".md", ".meta.json")
    payload = json.load(open(meta_path))["stats"]
    result = audit(open(args.summary).read(), payload, pairs=args.pairs)

    name = os.path.basename(args.summary)
    print(f"{name}: {result['figures']} figures against "
          f"{result['payload_values']} payload values")
    print(f"  grounded    {len(result['grounded']):>3}  "
          f"({result['grounded_pct']}%)")
    print(f"  derived     {len(result['derived']):>3}")
    print(f"  UNSUPPORTED {len(result['unsupported']):>3}")
    if result["unsupported"]:
        print("      " + ", ".join(t for t, _ in result["unsupported"]))
    if args.verbose:
        print("  derived:", ", ".join(t for t, _ in result["derived"]))
    if args.sensitivity:
        print("\n  detection rate against fabricated figures:")
        for band, pct in sensitivity(payload, pairs=args.pairs).items():
            print(f"      {band:<26} {pct:>5}%")
    return 1 if result["unsupported"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
