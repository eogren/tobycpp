# `src/core/` — protected learning zone

This directory and `include/toby/core/` hold the **inference engine you are
writing yourself to learn**. Per [`CLAUDE.md`](../../CLAUDE.md), AI assistants
must not implement or edit anything here — they may only read it and discuss it
Socratically.

Right now it contains a single placeholder (`version.cpp` +
`include/toby/core/version.hpp`) whose only job is to prove the build/test
pipeline works end to end. Replace and grow it as you build the real thing —
tensors, memory/layout, ops, the inference loop, and so on.

Everything outside this zone (build system, server plumbing, tests, tooling) is
fair game for AI help.
