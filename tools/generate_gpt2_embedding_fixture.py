#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "safetensors==0.6.2",
#   "torch==2.8.0",
#   "transformers==4.56.0",
# ]
# ///
"""Generate deterministic GPT-2 embedding reference outputs.

The fixture records 33 token ids, 33 independently sampled position ids, and
the result of Hugging Face GPT-2's embedding stage in evaluation mode:

    transformer.wte(input_ids) + transformer.wpe(position_ids)

Dropout is disabled by ``eval()``, so it does not alter the sum. The model
revision is pinned below; changing it intentionally changes the oracle.

This is a manual networked developer tool. The generated fixture itself is
checked in, so tests that consume it remain hermetic.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import GPT2Model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "embedding" / "gpt2.safetensors"

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEED = 0x70B7
CASE_COUNT = 33


def write_safetensors(tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """Write this fixture canonically so regeneration has a stable checksum."""
    header: dict[str, object] = {"__metadata__": metadata}
    payload = bytearray()

    dtype_names = {torch.float32: "F32", torch.uint16: "U16"}
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        start = len(payload)
        payload.extend(tensor.view(torch.uint8).numpy().tobytes())
        header[name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }

    encoded_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    encoded_header += b" " * (-len(encoded_header) % 8)
    OUTPUT.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)

    with safe_open(OUTPUT, framework="pt", device="cpu") as fixture:
        assert fixture.metadata() == metadata
        assert fixture.keys() == sorted(tensors)
        for name, expected in tensors.items():
            assert torch.equal(fixture.get_tensor(name), expected.cpu())


def main() -> None:
    model = GPT2Model.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()

    rng = random.Random(SEED)
    hf_input_ids = torch.tensor(
        [rng.randrange(model.config.vocab_size) for _ in range(CASE_COUNT)],
        dtype=torch.int64,
    )
    hf_position_ids = torch.tensor(
        [rng.randrange(model.config.n_positions) for _ in range(CASE_COUNT)],
        dtype=torch.int64,
    )

    with torch.inference_mode():
        embeddings = model.wte(hf_input_ids) + model.wpe(hf_position_ids)

    tensors = {
        "embeddings": embeddings.contiguous(),
        "input_ids": hf_input_ids.to(torch.uint16),
        "position_ids": hf_position_ids.to(torch.uint16),
    }
    metadata = {
        "case_count": str(CASE_COUNT),
        "description": "GPT-2 token plus position embedding reference outputs",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": str(SEED),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(tensors, metadata)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
