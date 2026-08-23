#!/usr/bin/env python3
"""Generate the small SafeTensors fixtures used by the test suite.

This intentionally uses only the Python standard library.  Keeping the writer
here makes the binary fixture reproducible without making safetensors a project
dependency.
"""

import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tests" / "fixtures" / "safetensors"


def write_fixture(name: str, header: object, data: bytes = b"") -> None:
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * (-len(encoded_header) % 8)

    output = OUTPUT_DIR / name
    output.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + data)
    print(f"wrote {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


def main() -> None:
    scalar = struct.pack("<f", 1.5)
    # 50256 is GPT-2's <|endoftext|> id: it overflows a signed I16 (max 32767)
    # but fits U16, which is the whole point of picking U16 for token ids.
    token_ids = struct.pack("<4H", 50256, 20, 30, 40)
    projection = struct.pack("<6f", -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    data = scalar + token_ids + projection

    header = {
        "__metadata__": {"description": "Small deterministic toby test fixture"},
        "scalar": {
            "dtype": "F32",
            "shape": [],
            "data_offsets": [0, len(scalar)],
        },
        "token_ids": {
            "dtype": "U16",
            "shape": [4],
            "data_offsets": [len(scalar), len(scalar) + len(token_ids)],
        },
        "projection": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [len(scalar) + len(token_ids), len(data)],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_fixture("basic.safetensors", header, data)

    write_fixture(
        "scalar.safetensors",
        {
            "scalar": {
                "dtype": "F32",
                "shape": [],
                "data_offsets": [0, len(scalar)],
            }
        },
        scalar,
    )

    write_fixture(
        "empty_tensor.safetensors",
        {
            "empty": {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [0, 0],
            }
        },
    )

    # The header promises two F32 values (8 bytes), but the payload contains one.
    write_fixture(
        "truncated_data.safetensors",
        {
            "truncated": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 2 * len(scalar)],
            }
        },
        scalar,
    )

    write_fixture(
        "missing_shape.safetensors",
        {"missing_shape": {"dtype": "F32", "data_offsets": [0, len(scalar)]}},
        scalar,
    )

    write_fixture(
        "invalid_dtype_type.safetensors",
        {
            "invalid_dtype": {
                "dtype": 32,
                "shape": [],
                "data_offsets": [0, len(scalar)],
            }
        },
        scalar,
    )

    write_fixture(
        "invalid_shape_type.safetensors",
        {
            "invalid_shape": {
                "dtype": "F32",
                "shape": "scalar",
                "data_offsets": [0, len(scalar)],
            }
        },
        scalar,
    )

    write_fixture(
        "invalid_offsets_type.safetensors",
        {
            "invalid_offsets": {
                "dtype": "F32",
                "shape": [],
                "data_offsets": [0, "4"],
            }
        },
        scalar,
    )


if __name__ == "__main__":
    main()
