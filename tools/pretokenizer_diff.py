#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["regex"]
# ///
"""Differential test: toby's pre-tokenizer vs. the actual GPT-2 split pattern.

The C++ suite can only assert *properties* (losslessness, in-bounds views) --
checking exact chunk boundaries needs an oracle, and the oracle is a regex with
\\p{L} classes and lookahead that std::regex cannot express. Hence Python: the
`regex` module supports the pattern verbatim, so the reference here is the real
thing rather than a reimplementation that could be wrong in the same way.

Usage (uv fetches `regex` itself; no venv to manage):

    ./tools/pretokenizer_diff.py build/clang-asan/bin/pretokenize_dump
    ./tools/pretokenizer_diff.py <dump-binary> --cases 50000 --max-len 120 --seed 7

Prefer an ASan+UBSan build of the dump binary: then a divergence and a memory
error get caught in the same run.

Exit status is 0 only if every case matches, so this drops into CI as-is.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

import regex

# The GPT-2 pre-tokenizer pattern, verbatim. Order matters: the alternation is
# tried left to right, so the contraction cases win over the punctuation run, and
# `\s+(?!\S)` (whitespace not followed by non-whitespace) wins over bare `\s+`.
GPT2_PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

# ASCII only: toby's pretokenize throws on bytes >= 0x80 for now. Widen this the
# moment that changes -- non-ASCII is where the next divergences will be.
ALPHABET = "aBz19 \t\n\r\v\f'\"!.,-_#*()[]{}<>/@$%^&+=~;:?"

# Inputs worth checking every run regardless of the seed: each one is a rule
# boundary, and several are cases that regressed at some point.
SEEDS = [
    "",
    "hello",
    " hello world",
    "don't stop",
    "abc123def",
    "a\tb",
    "a\t\tb",
    "a\t\t\tb",
    "a \nb",
    "a   b",
    "a   ",
    "  a",
    "hi!'",
    "!'!",
    "''t",
    "!'s",
    "'tis",
    "don'T",
]


def generate(count: int, max_len: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    cases = list(SEEDS)
    cases += [
        "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(0, max_len)))
        for _ in range(count)
    ]
    return cases


def run_dump(binary: str, cases: list[str]) -> list[str]:
    payload = "\n".join(case.encode().hex() for case in cases)
    proc = subprocess.run(
        [binary], input=payload, capture_output=True, text=True, timeout=600, check=False
    )
    if proc.returncode != 0:
        # A nonzero exit here is usually a sanitizer report, which is the whole
        # reason to point this at an ASan build. Surface it verbatim.
        print(f"{binary} exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr, end="")
        sys.exit(1)

    lines = proc.stdout.splitlines()
    if len(lines) != len(cases):
        print(
            f"expected {len(cases)} output lines, got {len(lines)} "
            "(did the binary hang or die mid-stream?)",
            file=sys.stderr,
        )
        sys.exit(1)
    return lines


def parse(line: str) -> list[str] | None:
    """Decode one output line into chunks, or None if the call threw."""
    if line == "THREW":
        return None
    if not line:
        return []
    return [bytes.fromhex(h).decode("utf-8", "surrogateescape") for h in line.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", help="path to the pretokenize_dump executable")
    parser.add_argument("--cases", type=int, default=20000, help="random cases (default 20000)")
    parser.add_argument("--max-len", type=int, default=80, help="max input length (default 80)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--show", type=int, default=10, help="mismatches to print (default 10)")
    args = parser.parse_args()

    cases = generate(args.cases, args.max_len, args.seed)
    lines = run_dump(args.binary, cases)

    mismatches = 0
    lossy = 0

    for case, line in zip(cases, lines, strict=True):
        want = GPT2_PATTERN.findall(case)
        got = parse(line)

        if got is None:
            mismatches += 1
            if mismatches <= args.show:
                print(f"THREW  {case!r}\n  want {want}")
            continue

        # Report a lost byte separately from a misplaced boundary: they have
        # different causes, and losslessness is the stronger invariant.
        if "".join(got) != case:
            lossy += 1
            mismatches += 1
            if mismatches <= args.show:
                print(f"LOSSY  {case!r}\n  got  {got}")
            continue

        if got != want:
            mismatches += 1
            if mismatches <= args.show:
                print(f"DIFF   {case!r}\n  got  {got}\n  want {want}")

    total = len(cases)
    if mismatches:
        hidden = mismatches - min(mismatches, args.show)
        extra = f" ({hidden} more not shown)" if hidden else ""
        print(f"\nFAIL: {mismatches}/{total} mismatched, {lossy} lossy{extra}")
        print(f"reproduce with --seed {args.seed} --cases {args.cases} --max-len {args.max_len}")
        return 1

    print(f"OK: {total} cases match the GPT-2 pattern (seed {args.seed}, max-len {args.max_len})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
