# CUDA development

See the [README](../README.md#cuda) for prerequisites and a basic build.

## C++ standard library

The Linux CUDA presets use Clang 20 with libstdc++ for host C++ and CUDA
sources. NVIDIA supports libstdc++ as NVCC's Linux host standard library;
using it throughout the build keeps shared C++ types compatible. Other Clang
presets continue to use libc++.

These presets disable `TOBY_HARDENING`, which configures libc++-specific
checks. C++ sources still use C++23; availability of library features depends
on the installed libstdc++ version.

The current code needs libstdc++ 14 or newer for `std::ranges::to` and `<print>`.
See the
[libstdc++ feature status](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html).
Install the matching development headers and runtime; Clang remains the
compiler. Both Clang and NVCC's host compilation must select the same library.

On Ubuntu, install the GCC 14 toolchain (including its libstdc++ headers):

```bash
sudo apt-get install g++-14
```

After switching an existing CUDA build from libc++, reconfigure and rebuild
all targets, including dependencies:

```bash
cmake --preset clang-cuda --fresh
cmake --build --preset clang-cuda
ctest --preset clang-cuda
```

## Build-time analysis

To also run clang-tidy on first-party C++ sources during compilation, use:

```bash
cmake --preset clang-cuda-tidy
cmake --build --preset clang-cuda-tidy
ctest --preset clang-cuda-tidy
```

This preset inherits `clang-cuda` and enables the matched LLVM 20 analyzer.
The build's clang-tidy integration checks C++ sources, not `.cu` kernels.

## GPU architectures

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

## Inspecting generated GPU code

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
