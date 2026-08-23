# SafeTensors fixtures

The valid fixtures are:

- `basic.safetensors`, containing:
  - `scalar`: an `F32` scalar;
  - `token_ids`: a `U16` vector with shape `[4]`;
  - `projection`: an `F32` matrix with shape `[2, 3]`;
  - a `__metadata__` object.
- `scalar.safetensors`, containing only one `F32` scalar.
- `empty_tensor.safetensors`, containing an `F32` tensor with shape `[0]`.

The intentionally malformed fixtures each isolate one validation error:

- `truncated_data.safetensors`: the payload is shorter than its declared offsets;
- `missing_shape.safetensors`: the tensor has no `shape` field;
- `invalid_dtype_type.safetensors`: `dtype` is a number instead of a string;
- `invalid_shape_type.safetensors`: `shape` is a string instead of an array;
- `invalid_offsets_type.safetensors`: an offset is a string instead of an integer.

Regenerate all fixtures from the repository root with:

```sh
uv run tools/generate_safetensors_fixture.py
```

The generator uses only Python's standard library.
