# Defines an INTERFACE target `toby_cuda` carrying what first-party code needs to
# compile CUDA kernels and call the runtime API. Link it PRIVATE alongside
# toby::warnings and friends.
#
# The target always exists. When CUDA is off (or unavailable, as on macOS) it is
# an empty stub that only defines TOBY_HAVE_CUDA=0, so a target's link list does
# not have to be written conditionally and host code can branch on the macro:
#
#     #if TOBY_HAVE_CUDA
#     #  include <cuda_runtime.h>
#     #endif
#
# Discovery is explicit rather than opportunistic (same reasoning as the
# zlib/brotli switches in Dependencies.cmake): with TOBY_ENABLE_CUDA=ON a missing
# nvcc/toolkit is a hard configure error, so a machine never silently produces
# a CPU-only build that was supposed to have GPU support.

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

# Keep CUDA's language ceiling local to .cu files. Host-only sources remain
# C++23. CUDA extensions are unnecessary for ordinary kernel launch syntax.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(
    FATAL_ERROR
    "toby: CMAKE_CUDA_ARCHITECTURES must be non-empty. Use 'native' for a "
    "local GPU build or an explicit capability such as 120 for a headless build."
  )
endif()

# This is deliberately guarded by TOBY_ENABLE_CUDA. CPU-only configurations --
# including every macOS preset -- never probe for nvcc.
enable_language(CUDA)
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

message(
  STATUS
  "toby: CUDA ${CUDAToolkit_VERSION}; compiler=${CMAKE_CUDA_COMPILER}; "
  "architectures=${CMAKE_CUDA_ARCHITECTURES}"
)

# Keep the boundary between nvcc-compiled code and the libc++ C++23 host code
# narrow and ABI-neutral (plain-C launcher functions and CUDA runtime types).
# That lets nvcc use its supported default host compiler without leaking one
# standard library's objects or exceptions into the other side.
