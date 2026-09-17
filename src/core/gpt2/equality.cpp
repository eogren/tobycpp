#include "equality_gpu.h"
#include "toby/gpt2/kernels.h"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>

using toby::tensors::DataType;
using toby::tensors::DeviceType;
using toby::tensors::Tensor;

namespace {
template <typename T> bool tensors_equal_cpu_compare(const T* t1, const T* t2, std::size_t size) {
    // TOOD: could try to make this simd, but compiler probably does it
    for (std::size_t i = 0; i < size; i++) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if (t1[i] != t2[i]) {
            return false;
        }
    }

    return true;
}

/**
 * On CPU kernel for equality. Assumes the checks about size, shape, dtype, etc are a
 * already done.
 */
bool tensors_equal_cpu(const Tensor& t1, const Tensor& t2) {
    switch (t1.dtype()) {
    case DataType::U16: {
        const auto* p1 = static_cast<const std::uint16_t*>(t1.data());
        const auto* p2 = static_cast<const std::uint16_t*>(t2.data());

        return tensors_equal_cpu_compare(p1, p2, t1.shape().numel());
    }
    case DataType::F32: {
        const auto* p1 = static_cast<const float*>(t1.data());
        const auto* p2 = static_cast<const float*>(t2.data());

        return tensors_equal_cpu_compare(p1, p2, t1.shape().numel());
    }
    default:
        throw std::invalid_argument{"unsupported type"};
    }
}
} // namespace

namespace toby::gpt2 {
bool tensors_equal(const Tensor& t1, const Tensor& t2) {
    if (t1.device() != t2.device()) {
        throw std::invalid_argument{
            std::format("Can't compare t1 on {} and t2 on {}", t1.device(), t2.device())};
    }

    if (t1.dtype() != t2.dtype()) {
        return false;
    }

    if (t1.shape() != t2.shape()) {
        return false;
    }

    if (t1.device() == DeviceType::CPU) {
        return tensors_equal_cpu(t1, t2);
    }
    if (t1.device() == DeviceType::GPU) {
        return detail::tensors_equal_gpu(t1, t2);
    }

    throw std::invalid_argument{"Unknown device type"};
}
} // namespace toby::gpt2
