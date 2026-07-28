#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Download real tokenizer vocabularies to test the loader against.

The synthetic fixtures in tests/fixtures/ cover every parser branch, but they are
ten entries each. Some things only show up at full scale -- a 50k-entry vocab, a
merges.txt with tokens that look like comments, a 9MB tokenizer.json -- so it is
worth pointing the loader at the real thing.

    ./tools/fetch_vocab.py gpt2              # vocab.json + merges.txt
    ./tools/fetch_vocab.py llama3 qwen3      # tokenizer.json each
    ./tools/fetch_vocab.py --list

Files land in vocab/<name>/ (gitignored -- these are hundreds of MB in total and
reproducible from this script).

Note these are downloads from HuggingFace over the network, so this is a manual
developer tool, not part of the test suite: CTest must stay hermetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

BASE = "https://huggingface.co/{repo}/resolve/main/{name}"

# Deliberately a mix: the GPT-2 pair to exercise load_gpt2_vocab, byte-level BPE
# tokenizer.json files to exercise load_tokenizer_json, and one SentencePiece
# model that the loader is supposed to REJECT.
SOURCES: dict[str, tuple[str, list[str]]] = {
    "gpt2": ("openai-community/gpt2", ["vocab.json", "merges.txt"]),
    "llama3": ("meta-llama/Meta-Llama-3-8B-Instruct", ["tokenizer.json"]),
    "qwen3": ("Qwen/Qwen3-8B", ["tokenizer.json"]),
    # Unigram/SentencePiece: load_tokenizer_json must throw on this one.
    "gemma2": ("google/gemma-2-2b-it", ["tokenizer.json"]),
}

# Some repos are gated: HuggingFace answers 401/403 unless the caller is logged
# in and has accepted the license. Llama 3 is the usual offender. Say so plainly
# rather than writing an HTML error page to disk and letting the loader choke.
GATED_HINT = (
    "this repo is gated -- accept its license on huggingface.co, then either\n"
    "  export HF_TOKEN=hf_...   or   run `huggingface-cli login`"
)


def fetch(repo: str, name: str, dest: Path, token: str | None) -> None:
    url = BASE.format(repo=repo, name=name)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise SystemExit(f"{url}\n  {r.status_code}: {GATED_HINT}")
        r.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Stream to a .part file and rename on success: a half-downloaded
        # tokenizer.json that looks complete is a genuinely annoying bug to chase.
        part = dest.with_suffix(dest.suffix + ".part")
        written = 0
        with part.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                written += len(chunk)
        part.rename(dest)

    print(f"  {name}: {written / 1024:.0f} KiB -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=["gpt2"], help="which vocabularies to fetch")
    parser.add_argument("--list", action="store_true", help="list known names and exit")
    parser.add_argument("--out", type=Path, default=Path("vocab"), help="output dir (default vocab/)")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    if args.list:
        for name, (repo, files) in SOURCES.items():
            print(f"{name:10s} {repo}  ({', '.join(files)})")
        return 0

    import os

    token = args.token or os.environ.get("HF_TOKEN")

    for name in args.names:
        if name not in SOURCES:
            print(f"unknown vocabulary {name!r}; try --list", file=sys.stderr)
            return 2

        repo, files = SOURCES[name]
        print(f"{name} ({repo}):")
        for filename in files:
            fetch(repo, filename, args.out / name / filename, token)

    return 0


if __name__ == "__main__":
    sys.exit(main())
