# Development tooling

## clang-tidy on macOS

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

## VS Code troubleshooting

See the [README](../README.md#vs-code) for initial setup.

After installing `clangd`, run **Developer: Reload Window** from the VS Code
command palette. On macOS, add `$(brew --prefix llvm)/bin` to the environment
used to launch VS Code if you want Homebrew's clangd. To inspect the exact
command clangd uses for a file, open **View: Output** and select **clangd**.

For Linux CUDA development, select `clang-cuda-tidy` with **CMake: Select
Configure Preset**, then run **CMake: Configure** to refresh the compilation
database used by clangd. This adds build-time analysis; the workspace already
enables clangd's editor-time clang-tidy checks. Use `clang-cuda` if you prefer
faster builds while retaining those editor checks.
