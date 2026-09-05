# Embedding fixtures

`gpt2.safetensors` contains 33 deterministic reference cases for GPT-2's
embedding stage. Its tensors are:

- `input_ids`: 33 `uint16` random token ids, sampled from `[0, 50257)`;
- `position_ids`: 33 independently random `uint16` positions, sampled from
  `[0, 1024)`;
- `embeddings`: the `float32` result of
  `transformer.wte(input_ids) + transformer.wpe(position_ids)`, with shape
  `[33, 768]`.

The SafeTensors metadata records the model id, exact Hugging Face revision,
random seed, and case count. The output is computed with the model in evaluation
mode, so GPT-2's embedding dropout is disabled.

Regenerate the fixture from the repository root with:

```sh
uv run --script tools/generate_gpt2_embedding_fixture.py
```

The first run downloads the pinned GPT-2 model. Generation is deliberately not
wired into CTest: the checked-in fixture keeps future tests hermetic and avoids
a runtime Python or network dependency.
