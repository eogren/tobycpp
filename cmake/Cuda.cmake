# Defines an INTERFACE target `toby_cuda` carrying what first-party code needs to
# call the CUDA *runtime* API -- cudaMalloc/cudaFree/cudaMemcpy, streams, events,
# and device queries. Link it PRIVATE alongside toby::warnings and friends.
#
# The target always exists. When CUDA is off (or unavailable, as on macOS) it is
# an empty stub that only defines TOBY_HAVE_CUDA=0, so a target's link list does
# not have to be written conditionally and host code can branch on the macro:
#
#     #if TOBY_HAVE_CUDA
#     #  include <cuda_runtime.h>
#     #endif
#
# NO `enable_language(CUDA)` HERE -- ON PURPOSE.
#
# Enabling the CUDA language makes CMake probe for and validate nvcc at configure
# time, which we do not need to *call* the runtime: cuda_runtime.h is ordinary
# C++, and libcudart is an ordinary shared library. Any host compiler links it.
# Adding a .cu file later is a contained change; see the recipe at the bottom.
#
# Discovery is explicit rather than opportunistic (same reasoning as the
# zlib/brotli switches in Dependencies.cmake): with TOBY_ENABLE_CUDA=ON a missing
# toolkit is a hard configure error, so a machine never silently produces a
# CPU-only build that was supposed to have GPU support.

add_library(toby_cuda INTERFACE)
add_library(toby::cuda ALIAS toby_cuda)

if(NOT TOBY_ENABLE_CUDA)
  target_compile_definitions(toby_cuda INTERFACE TOBY_HAVE_CUDA=0)
  message(STATUS "toby: CUDA disabled (-DTOBY_ENABLE_CUDA=ON to enable)")
  return()
endif()

# FindCUDAToolkit locates the toolkit from nvcc on PATH, then from CUDAToolkit_ROOT
# / the CUDA_PATH environment variable. A default Linux install puts nvcc in
# /usr/local/cuda/bin, which is not on PATH unless the user added it -- so fall
# back to the canonical symlink before giving up. Anything unusual (a side-by-side
# 12.x for a driver that cannot run 13.x, a container mount) is handled by passing
# -DCUDAToolkit_ROOT=/path/to/cuda, which takes precedence over this.
if(NOT DEFINED CUDAToolkit_ROOT AND NOT DEFINED ENV{CUDA_PATH} AND EXISTS /usr/local/cuda)
  set(CUDAToolkit_ROOT /usr/local/cuda)
endif()

find_package(CUDAToolkit REQUIRED)

# CUDA::cudart is the shared libcudart. CUDA::cudart_static is the alternative and
# is what nvcc links by default -- it drops the runtime .so from the deployment
# set at the cost of ~1MB per binary and a link against rt/pthread/dl. Shared is
# the better default here because every binary in this tree is built and run from
# the same machine, and because ldconfig already resolves libcudart.so.13 from the
# toolkit's own conf file on a normal install.
#
# Neither library is the driver. libcuda.so ships with the *driver*, not the
# toolkit, and libcudart dlopens it at process start -- which is why this works
# under WSL2, where the driver lives in /usr/lib/wsl/lib and there is no
# /dev/nvidia* device node in the usual place.
target_link_libraries(toby_cuda INTERFACE CUDA::cudart)
target_compile_definitions(toby_cuda INTERFACE TOBY_HAVE_CUDA=1)

# The toolkit headers arrive through CUDA::cudart's INTERFACE_INCLUDE_DIRECTORIES.
# CMake passes an IMPORTED target's interface includes as -isystem, so they are
# already exempt from our warnings-as-errors -- no SYSTEM keyword needed the way
# the FetchContent dependencies need one.

message(STATUS "toby: CUDA runtime ${CUDAToolkit_VERSION} from ${CUDAToolkit_LIBRARY_ROOT}")

# ---------------------------------------------------------------------------
# WHEN NVCC IS ACTUALLY NEEDED (writing .cu kernels), the delta is:
#
#   1. project(... LANGUAGES CXX CUDA) -- or `enable_language(CUDA)` guarded by
#      TOBY_ENABLE_CUDA, which keeps CPU-only configures free of the nvcc probe.
#   2. set(CMAKE_CUDA_STANDARD 20) + CMAKE_CUDA_STANDARD_REQUIRED ON. nvcc 13
#      does not accept -std=c++23 for device code, so a .cu file cannot be C++23
#      even though the rest of the tree is. Keep kernels behind a plain C++ header
#      so this ceiling stays inside the .cu files.
#   3. set(CMAKE_CUDA_ARCHITECTURES 120) for this box -- an RTX PRO 500 Blackwell
#      is compute capability 12.0. `native` asks nvcc to detect the local GPU;
#      a list like "90;120" or "120-real;120-virtual" builds a fat binary that
#      also runs elsewhere.
#   4. set(CMAKE_CUDA_HOST_COMPILER ...) -- nvcc drives a *host* compiler for the
#      non-device half of a .cu file, and it defaults to g++/libstdc++, which
#      would mix standard libraries with our libc++ objects. Either point it at
#      clang++-20 (nvcc 13 supports clang as host) or, cleaner, keep .cu files
#      free of our C++23/libc++ types so the two halves only meet at a plain-C
#      launcher signature.
#   5. Sanitizers: ASan/TSan flags do not survive being handed to nvcc. Add
#      $<$<COMPILE_LANGUAGE:CXX>:...> genex guards in Sanitizers.cmake, or use
#      the toolkit's own compute-sanitizer for device code.
# ---------------------------------------------------------------------------
