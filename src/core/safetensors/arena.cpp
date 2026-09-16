#include "toby/safetensors/arena.hpp"

#include "cpu_arena.hpp"
#include "toby/safetensors/align.hpp"
#include "toby/safetensors/memory_transfer.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#if !TOBY_HAVE_CUDA
#include <stdexcept>
#endif

#if TOBY_HAVE_CUDA
#include "gpu_arena.hpp"
#endif

using toby::tensors::DeviceType;

namespace toby::tensors {
Arena::~Arena() = default;

std::unique_ptr<Arena> Arena::alloc_anonymous(std::size_t size, DeviceType device_type,
                                              std::optional<std::string_view> name) {
    switch (device_type) {
    case DeviceType::CPU:
        return std::make_unique<detail::CpuArena>(name, size);
    case DeviceType::GPU:
#if TOBY_HAVE_CUDA
        return std::make_unique<detail::GpuArena>(name, size);
#else
        throw std::invalid_argument{"no gpus in this build"};
#endif
    }
}

std::span<std::byte> Arena::alloc(std::size_t len) {
    auto start_offset = align_up(used_, 64);
    std::size_t next_used{};
    if (__builtin_add_overflow(start_offset, len, &next_used)) {
        throw std::invalid_argument{"Overflow when trying to calculate new arena size"};
    }

    if (next_used > size_) {
        throw std::invalid_argument{std::format("Arena too full to allocate {} more bytes", len)};
    }

    used_ = next_used;
    // NOLINTNEXTLINE
    return std::span{static_cast<std::byte*>(base_) + start_offset, len};
}

// TODO(you): review and remove this marker
void Arena::memcpy(std::span<const std::byte> src, DeviceType device, std::size_t offset) {
    size_t end{};
    if (__builtin_add_overflow(src.size(), offset, &end)) {
        throw std::invalid_argument{"overflow calculating end of memcpy"};
    }

    if (end > used()) {
        throw std::invalid_argument{"not enough room left in arena"};
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    auto* dest = static_cast<std::byte*>(base()) + offset;

    copy_bytes(std::span{dest, src.size()}, device_, src, device);
}

std::span<std::byte> Arena::alloc_from_cpu_ptr(std::span<const std::byte> in) {
    auto ret = alloc(in.size_bytes());
    copy_bytes(ret, device_, in, DeviceType::CPU);
    return ret;
}

void Arena::bulk_memcpy(const std::vector<MemcpyInfo>& copies) {
    for (const auto& copy : copies) {
        // Raw source pointers retain their caller-validated range contract.
        const auto* src = static_cast<const std::byte*>(copy.src);
        if (copy.src_offset != 0) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            src += copy.src_offset;
        }
        memcpy(std::span{src, copy.size}, copy.src_device, copy.new_offset);
    }
}

std::span<const std::byte> Arena::byte_span() const {
    return std::span{static_cast<std::byte*>(base_), used_};
}
} // namespace toby::tensors
