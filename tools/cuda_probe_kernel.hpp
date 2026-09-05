#pragma once

#include <cuda_runtime_api.h>

// Keep the nvcc/libstdc++ <-> Clang/libc++ boundary free of C++ library types,
// ownership, and exceptions. Kernel launchers in the engine should follow the
// same pattern.
extern "C" cudaError_t cuda_probe_launch_increment(float* device_values);
