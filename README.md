# toby

A C++ inference server, built **from scratch for learning**. The engine lives
in `toby::core` and is written by hand on purpose; the surrounding
infrastructure (build, tooling, server plumbing, tests) is conventional and
AI-assisted. See [`CLAUDE.md`](CLAUDE.md) for how AI assistants are expected to
help here, and [`docs/learning-mode.md`](docs/learning-mode.md) for the
Socratic prompt skeleton.

## Requirements

- CMake ≥ 3.25, Ninja
- Clang 20 + libc++ (`clang-20`, `libc++-20-dev`)
- C++23 (uses `std::print`)

On Ubuntu 24.04 (Noble):

```bash
sudo apt install clang-20 clang-tidy-20 clang-format-20 \
                 libc++-20-dev libc++abi-20-dev lld-20 \
                 cmake ninja-build
```

## Build & test

```bash
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug

# Run the server skeleton
./build/clang-debug/bin/toby_server
```

Other presets: `clang-release`, `clang-asan` (Address+UB), `clang-tsan`
(Thread), `clang-tidy` (static analysis).
Run `cmake --list-presets` to see them all.

## Layout

```
include/toby/core/   Public engine headers   [PROTECTED — yours]
src/core/            Engine implementation    [PROTECTED — yours]
src/main.cpp         Server entry point       (plumbing)
tests/               Catch2 unit tests
cmake/               Warnings / sanitizers / static-analysis modules
CMakePresets.json    Toolchain + build presets
```

## Quality gates

- **Warnings:** a strict curated set (`cmake/CompilerWarnings.cmake`), treated
  as errors on the Clang preset.
- **Sanitizers:** ASan+UBSan and TSan presets, wired from the start.
- **Static analysis:** clang-tidy via the `clang-tidy` preset.
- **Formatting:** `.clang-format`, enforced by a pre-commit hook (the hook
  self-provisions its clang-format binary — no system install needed).

Enable the git hooks once:

```bash
uv tool install pre-commit   # if not already installed
pre-commit install
```
