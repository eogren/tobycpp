#include "toby/safetensors/arena.hpp"

#include "cpu_arena.hpp"
#include "toby/safetensors/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>

#if !TOBY_HAVE_CUDA
#include <stdexcept>
#endif

#if TOBY_HAVE_CUDA
#include "gpu_arena.hpp"
#endif

using toby::tensors::DeviceType;

namespace toby::tensors {
Arena::~Arena() = default;

std::unique_ptr<Arena> Arena::alloc_anonymous(std::size_t size, DeviceType device_type) {
    switch (device_type) {
    case DeviceType::CPU:
        return std::make_unique<detail::CpuArena>(size);
    case DeviceType::GPU:
#if TOBY_HAVE_CUDA
        return std::make_unique<detail::GpuArena>(size);
#else
        throw std::invalid_argument{"no gpus in this build"};
#endif
    }
}

std::span<const std::byte> Arena::byte_span() const {
    return std::span{static_cast<std::byte*>(base_), size_};
}
} // namespace toby::tensors
