#ifndef SAFETENSORS_ALIGN_HPP
#define SAFETENSORS_ALIGN_HPP

#include <cstddef>

namespace toby::tensors {

/// Rounds `size` up to the nearest multiple of `alignment`.
///
/// `alignment` need not be a power of two. Returns `size` unchanged when it is
/// already a multiple.
[[nodiscard]] constexpr std::size_t align_up(std::size_t size, std::size_t alignment) noexcept {
    return (size + alignment - 1) / alignment * alignment;
}

} // namespace toby::tensors

#endif
