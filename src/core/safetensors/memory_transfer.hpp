#ifndef SAFETENSORS_MEMORY_TRANSFER_HPP
#define SAFETENSORS_MEMORY_TRANSFER_HPP

#include "toby/safetensors/tensor_types.hpp"

#include <cstddef>
#include <span>

namespace toby::tensors::detail {
/**
 * Copy src.size_bytes() bytes; dst may be larger. Returns only after completion.
 * Throws std::invalid_argument for an undersized destination, invalid device,
 * or GPU use in a CPU-only build (including empty GPU transfers).
 *
 * GPU storage belongs to the current CUDA device; callers must finish any
 * producers on other streams before calling. Uses the default CUDA stream.
 */
void copy_bytes(std::span<std::byte> dst, DeviceType dst_device, std::span<const std::byte> src,
                DeviceType src_device);
} // namespace toby::tensors::detail

#endif
