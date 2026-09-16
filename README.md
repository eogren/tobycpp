# toby

A C++ inference server built **from scratch for learning**. The server currently
exposes `GET /healthz` on `127.0.0.1:8080`; inference is under development.
See [`AGENTS.md`](AGENTS.md) for the learning boundaries and
[`docs/learning-mode.md`](docs/learning-mode.md) for the Socratic workflow.

## Requirements

- CMake ≥ 3.25, Ninja
- Linux: Clang 20 + libc++ (`clang-20`, `libc++-20-dev`), ICU, and OpenSSL 3 headers
- macOS: a recent Xcode command-line toolchain, ICU, and OpenSSL 3
  (`brew install icu4c openssl@3`)
- C++23 (uses `std::print`)

On Ubuntu 24.04 (Noble), with the LLVM 20 package repository configured
(see the setup in [CI](.github/workflows/ci.yml)):

```bash
sudo apt install clang-20 clangd-20 clang-tidy-20 clang-format-20 \
                 libc++-20-dev libc++abi-20-dev lld-20 \
                 libicu-dev libssl-dev cmake ninja-build
```

## Build & test

On Linux:

```bash
cmake --preset clang-debug
cmake --build --preset clang-debug
ctest --preset clang-debug

# Run the server skeleton
./build/clang-debug/bin/toby_server
```

On macOS:

```bash
cmake --preset macos-debug
cmake --build --preset macos-debug
ctest --preset macos-debug

# Run the server skeleton
./build/macos-debug/bin/toby_server
```

Other Linux presets: `clang-release`, `clang-asan` (Address+UB), `clang-tsan`
(Thread), `clang-tidy`, and the CUDA presets below. macOS has equivalents for
debug, release, sanitizers, and analysis; see [macOS clang-tidy setup](docs/tooling.md#clang-tidy-on-macos).
Run `cmake --list-presets` to list available configure presets.

Differential tests require `uv`. The BPE differential test also requires
`./tools/fetch_vocab.py gpt2`; reconfigure afterward to register it with CTest.
These tests are omitted when their prerequisites are missing.

### CUDA

GPU support is off by default. The `clang-cuda` and `clang-cuda-release`
presets enable it and require `nvcc`, the CUDA toolkit, and an NVIDIA driver.
They use Clang 20 with libstdc++ (including its development headers) to match
NVCC's host standard library, and disable libc++-specific hardening.
Use `-DCUDAToolkit_ROOT=...` to specify a toolkit location.

```bash
cmake --preset clang-cuda
cmake --build --preset clang-cuda

# Compiler/driver/toolkit sanity check
./build/clang-cuda/bin/cuda_probe
```

The architecture defaults to `native`, requiring a visible GPU during
configuration. For a headless build or a different target GPU, specify a
compute capability supported by your toolkit, for example:

```bash
cmake --preset clang-cuda -DCMAKE_CUDA_ARCHITECTURES=120
```

See [CUDA development](docs/cuda.md) for `clang-cuda-tidy`, multi-architecture
builds, and PTX/SASS inspection.

## VS Code

Open the repository root, install the recommended extensions, and ensure
`clangd` is on `PATH`. Run **CMake: Select Configure Preset** (`clang-debug` on
Linux or `macos-debug` on macOS), then **CMake: Configure**.

CMake Tools copies the active preset's compilation database to the repository
root for clangd, as configured in [`.vscode/settings.json`](.vscode/settings.json).
See [development tooling](docs/tooling.md) for troubleshooting and CUDA editor setup.

## Layout

```
include/toby/core/          Public engine headers  [PROTECTED — yours]
include/toby/safetensors/   Tensor and safetensors headers
include/toby/tokenize/      Tokenizer headers
src/core/                  Engine implementation [PROTECTED — yours]
src/tokenize/              Tokenizer implementation
src/main.cpp               Server entry point (plumbing)
tests/                     Catch2 tests and fixtures
tools/                     Developer tools and differential tests
cmake/                     Warnings / sanitizers / static-analysis modules
CMakePresets.json           Toolchain + build presets
```

## Quality gates

The Clang and macOS presets treat warnings as errors; sanitizer and clang-tidy
presets provide runtime checks and static analysis. Pre-commit hooks enforce
`.clang-format` (C/C++) and `.gersemirc` (CMake), downloading their own formatters.

Enable the git hooks once:

```bash
uv tool install pre-commit   # if not already installed
pre-commit install
```
