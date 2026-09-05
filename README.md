# toby

A C++ inference server, built **from scratch for learning**. The engine lives
in `toby::core` and is written by hand on purpose; the surrounding
infrastructure (build, tooling, server plumbing, tests) is conventional and
AI-assisted. See [`CLAUDE.md`](CLAUDE.md) for how AI assistants are expected to
help here, and [`docs/learning-mode.md`](docs/learning-mode.md) for the
Socratic prompt skeleton.

## Requirements

- CMake ≥ 3.25, Ninja
- Linux: Clang 20 + libc++ (`clang-20`, `libc++-20-dev`) and OpenSSL 3 headers
- macOS: a recent Xcode command-line toolchain, ICU, and OpenSSL 3
  (`brew install icu4c openssl@3`)
- C++23 (uses `std::print`)

On Ubuntu 24.04 (Noble):

```bash
sudo apt install clang-20 clangd-20 clang-tidy-20 clang-format-20 \
                 libc++-20-dev libc++abi-20-dev lld-20 \
                 libssl-dev cmake ninja-build
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

Other presets: `clang-release`, `clang-asan` (Address+UB), `clang-tsan`
(Thread), `clang-cuda` (see below), plus corresponding `macos-*` presets. Linux
static analysis uses `clang-tidy`; macOS uses `macos-tidy` as described below.
Run `cmake --list-presets` to see them all.

### CUDA

GPU support is off by default; `-DTOBY_ENABLE_CUDA=ON` (or the `clang-cuda` /
`clang-cuda-release` presets) enables `.cu` compilation and links first-party
targets against the CUDA runtime. A missing nvcc or toolkit is a configure
error rather than a silent CPU-only build. CPU and macOS configurations never
probe for CUDA.

Requires an NVIDIA driver and the CUDA toolkit (headers + `libcudart`).
CMake finds it via `nvcc` on `PATH`, then `CUDAToolkit_ROOT`/`CUDA_PATH`, then
`/usr/local/cuda`; point it elsewhere with `-DCUDAToolkit_ROOT=...`.

```bash
cmake --preset clang-cuda
cmake --build --preset clang-cuda

# Compiler/driver/toolkit sanity check: properties + a tiny kernel round trip
./build/clang-cuda/bin/cuda_probe
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which is the best choice for a
local development build that will run on the same GPU. It produces a smaller,
faster build but requires a visible GPU while configuring. For a headless build
or a known deployment GPU, pass its compute capability explicitly:

```bash
cmake --preset clang-cuda -DCMAKE_CUDA_ARCHITECTURES=120
```

For a binary shipped to several GPU generations, use real machine code for
each supported architecture and optionally PTX for forward compatibility on
the newest one, for example
`-DCMAKE_CUDA_ARCHITECTURES=90-real\;120-real\;120-virtual`. Only list
architectures supported by the installed toolkit; every extra entry increases
compile time and binary size. Avoid `all`/`all-major` for routine development.

#### Inspecting generated GPU code

Normal CUDA builds embed device code in object files and executables rather
than leaving standalone `.ptx` files. To inspect your custom kernels, run the
toolkit's `cuobjdump` on the executable, library, or object file that contains
them. PTX is present when `CMAKE_CUDA_ARCHITECTURES` includes virtual code; an
unsuffixed architecture such as `120` requests both real and virtual code.

```bash
cuobjdump --dump-ptx path/to/your/kernel_target
```

PTX is NVIDIA's virtual instruction set, not the final instructions executed by
an SM. To inspect the final SASS machine code instead, use
`cuobjdump --dump-sass path/to/your/kernel_target`.

### clang-tidy on macOS

Xcode provides AppleClang but not a matching analyzer. Install Homebrew LLVM
and put its compiler and tools first on `PATH` so analysis uses a matched pair:

```bash
brew install llvm
PATH="$(brew --prefix llvm)/bin:$PATH" cmake --preset macos-tidy
cmake --build --preset macos-tidy
```

The `macos-tidy` preset deliberately rejects AppleClang rather than silently
combining different compiler and analyzer versions. The Linux `clang-tidy`
preset remains pinned to the LLVM 20 compiler and analyzer used in CI.

## VS Code

Open the repository root in VS Code and install its recommended extensions:

- **clangd** for completion, diagnostics, navigation, and clang-tidy feedback;
- **CMake Tools** for configuring and building the CMake presets;
- **CodeLLDB** for debugging Clang-built binaries.

The checked-in [`.vscode/settings.json`](.vscode/settings.json) finds `clangd`
on `PATH`. CMake Tools copies the active preset's compilation database to the
repository root, where clangd discovers it without a platform-specific path.
Configure the appropriate preset at least once before expecting accurate editor
diagnostics:

```bash
cmake --preset macos-debug  # macOS
# cmake --preset clang-debug  # Linux
```

After installing `clangd`, run **Developer: Reload Window** from the VS Code
command palette. On macOS, add `$(brew --prefix llvm)/bin` to the environment
used to launch VS Code if you want Homebrew's clangd. To inspect the exact
command clangd uses for a file, open **View: Output** and select **clangd**.

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
- **Static analysis:** clang-tidy via `clang-tidy` with LLVM 20 on Linux or
  `macos-tidy` with a matched Homebrew LLVM toolchain on macOS.
- **Formatting:** `.clang-format` for C/C++ and `.gersemirc` for CMake, enforced
  by pre-commit hooks that self-provision their formatter binaries.

Enable the git hooks once:

```bash
uv tool install pre-commit   # if not already installed
pre-commit install
```
