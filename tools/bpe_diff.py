#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken"]
# ///
"""Differential test: toby's full encode() vs. tiktoken, at the token-id level.

The sibling script pretokenizer_diff.py checks *where the chunk boundaries land*.
This one checks the thing that actually matters: the ids you hand the model. They
are separate tests on purpose -- a pre-tokenizer bug and a merge-ordering bug
produce the same symptom (wrong ids) and you want to know which one you have.

tiktoken is the reference rather than a hand-rolled BPE loop, for the same reason
the regex module is the reference over there: an oracle you wrote yourself can be
wrong in exactly the same way as the code under test.

    ./tools/bpe_diff.py build/clang-asan/bin/bpe_dump
    ./tools/bpe_diff.py <dump-binary> --cases 50000 --encoding gpt2 --seed 7

Exit status is 0 only if every case matches, so this drops into CI as-is.

-----------------------------------------------------------------------------
THE C++ SIDE IS YOURS TO WRITE (it needs an encoder, which does not exist yet).

Build `tools/bpe_dump.cpp` the same shape as the existing `pretokenize_dump.cpp`
-- copy its hex/IO scaffolding, swap what it calls. The wire protocol this script
expects:

  * reads one hex-encoded input per line on stdin;
  * writes one line per input: the token ids in order, decimal, comma-separated;
  * writes an empty line for an input that encodes to no tokens;
  * writes the single token THREW if the call threw.

Encode with specials DISABLED (the `encode_ordinary` path): this harness feeds
random bytes, and some of them will spell `<|endoftext|>` sooner or later. Text
input must never be able to produce a special id -- that is the whole invariant,
so the test should be exercising the path that enforces it.

Once bpe_dump exists, wire it into CTest next to pretokenizer_diff_gpt2 in
tests/CMakeLists.txt -- the add_test() block there is copy-and-edit.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

import tiktoken

# ASCII only, matching the pre-tokenizer's current limits. Widen this the moment
# non-ASCII lands -- see --unicode below, which is already wired for that day.
ASCII_ALPHABET = "aBz19 \t\n\r\v\f'\"!.,-_#*()[]{}<>/@$%^&+=~;:?"

# Characters chosen to hit the cases that break byte-level BPE specifically:
# multi-byte codepoints, a combining mark, a non-Latin script, an astral-plane
# emoji (4 UTF-8 bytes), and NBSP -- which looks like a space and is not one.
UNICODE_EXTRA = "éüñ\u0301αβшΩ日本語🙂🇺🇸\u00a0\u200b\u2014"

# Inputs worth checking every run regardless of the seed. Each is a rule boundary
# or a known-nasty case rather than a random string.
SEEDS = [
    "",
    "hello",
    " hello world",
    "don't stop",
    "abc123def",
    "1234567",
    "a   b",
    "  a",
    "a \nb",
    # Indentation: where GPT-2 and cl100k diverge most for code.
    "def f():\n    return 1",
    "\t\tif x:\n\t\t\tpass",
    # Special-token text. MUST come back as ordinary tokens, never as the real id.
    "<|endoftext|>",
    "hi <|endoftext|> bye",
    "<|im_start|>system",
    # Long digit runs, where the {1,3} vs greedy difference shows up.
    "3.14159265358979",
    "2024-01-31T12:00:00Z",
]


def generate(count: int, max_len: int, seed: int, unicode: bool) -> list[str]:
    rnd = random.Random(seed)
    alphabet = ASCII_ALPHABET + (UNICODE_EXTRA if unicode else "")
    cases = list(SEEDS)
    cases += [
        "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, max_len)))
        for _ in range(count)
    ]
    return cases


def run_dump(binary: str, cases: list[str]) -> list[str]:
    payload = "\n".join(case.encode().hex() for case in cases)
    proc = subprocess.run(
        [binary], input=payload, capture_output=True, text=True, timeout=900, check=False
    )
    if proc.returncode != 0:
        # Usually a sanitizer report, which is the reason to point this at an
        # ASan build. Surface it verbatim.
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


def parse(line: str) -> list[int] | None:
    """Decode one output line into ids, or None if the call threw."""
    if line == "THREW":
        return None
    if not line:
        return []
    try:
        return [int(part) for part in line.split(",")]
    except ValueError:
        print(f"unparseable output line: {line!r}", file=sys.stderr)
        sys.exit(1)


def show_mismatch(case: str, got: list[int], want: list[int], enc: tiktoken.Encoding) -> None:
    """Print the first differing position, decoded, not just two id lists.

    A bare pair of 200-element lists tells you nothing. The index of the first
    divergence plus the token text on each side usually names the bug outright.
    """
    print(f"DIFF   {case!r}")
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            print(f"  first differs at {i}: got {g} ({enc.decode([g])!r}) "
                  f"want {w} ({enc.decode([w])!r})")
            break
    else:
        shorter, longer = ("got", "want") if len(got) < len(want) else ("want", "got")
        print(f"  {shorter} is a prefix of {longer} ({len(got)} vs {len(want)} tokens)")
    print(f"  got  {got}")
    print(f"  want {want}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", help="path to the bpe_dump executable")
    parser.add_argument("--encoding", default="gpt2", help="tiktoken encoding (default gpt2)")
    parser.add_argument("--cases", type=int, default=20000, help="random cases (default 20000)")
    parser.add_argument("--max-len", type=int, default=80, help="max input length (default 80)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--show", type=int, default=10, help="mismatches to print (default 10)")
    parser.add_argument("--unicode", action="store_true",
                        help="include non-ASCII in the random alphabet")
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    cases = generate(args.cases, args.max_len, args.seed, args.unicode)
    lines = run_dump(args.binary, cases)

    mismatches = 0

    for case, line in zip(cases, lines, strict=True):
        # encode_ordinary, not encode: specials must be unreachable from text,
        # so the reference has to treat "<|endoftext|>" as ordinary characters.
        want = enc.encode_ordinary(case)
        got = parse(line)

        if got is None:
            mismatches += 1
            if mismatches <= args.show:
                print(f"THREW  {case!r}\n  want {want}")
            continue

        if got != want:
            mismatches += 1
            if mismatches <= args.show:
                show_mismatch(case, got, want, enc)

    total = len(cases)
    if mismatches:
        hidden = mismatches - min(mismatches, args.show)
        extra = f" ({hidden} more not shown)" if hidden else ""
        print(f"\nFAIL: {mismatches}/{total} mismatched{extra}")
        print(f"reproduce with --seed {args.seed} --cases {args.cases} --max-len {args.max_len}")
        return 1

    print(f"OK: {total} cases match {args.encoding} (seed {args.seed}, max-len {args.max_len})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
