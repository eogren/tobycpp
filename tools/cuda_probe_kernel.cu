#include "cuda_probe_kernel.hpp"

namespace {

__global__ void increment_first(float* const values) {
    values[0] += 1.0F;
}

} // namespace

extern "C" cudaError_t cuda_probe_launch_increment(float* const device_values) {
    increment_first<<<1, 1>>>(device_values);
    return cudaGetLastError();
}
