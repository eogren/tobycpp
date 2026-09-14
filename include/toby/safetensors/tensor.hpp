#ifndef SAFETENSORS_TENSOR_HPP
#define SAFETENSORS_TENSOR_HPP

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace toby::tensors {
constexpr size_t bytes_per_elem(const DataType type) {
    switch (type) {
    case toby::tensors::DataType::F32:
        return 4;
    case toby::tensors::DataType::U16:
        return 2;
    }
}

class TensorShape {
public:
    constexpr TensorShape(std::initializer_list<std::size_t> dims) : TensorShape(std::span{dims}) {}

    constexpr TensorShape(std::span<const size_t> dims) : dims_{-1, -1, -1, -1} {
        if (dims.size() > 4) {
            throw std::invalid_argument{"only up to 4 dims supported"};
        }

        std::size_t working_numel = 1;
        for (auto dim : dims) {
            if (__builtin_mul_overflow(working_numel, dim, &working_numel)) {
                throw std::overflow_error{"tensor too big; numel would overflow"};
            }
        }
        std::ranges::copy(dims, dims_.begin());
    }

    /** Iterate through dimensions of the shape. Must not outlive the shape itself. */
    [[nodiscard]] constexpr auto dimensions() const {
        return std::views::iota(std::size_t{0}, ndim()) |
               std::views::transform([this](std::size_t axis) { return (*this)[axis]; });
    }

    [[nodiscard]] constexpr std::size_t ndim() const {
        const auto* it = std::ranges::find(dims_, -1);
        return static_cast<std::size_t>(it - dims_.begin());
    }

    [[nodiscard]] constexpr std::size_t numel() const {
        auto dims = ndim();
        std::size_t ret = 1;

        for (std::size_t i = 0; i < dims; i++) {
            const auto d = operator[](i);
            if (d != 0 && ret > std::numeric_limits<std::size_t>::max() / d) {
                throw std::overflow_error{"TensorShape::numel overflowed size_t"};
            }
            ret *= d;
        }

        return ret;
    }

    constexpr std::size_t operator[](std::size_t idx) const {
        auto val = dims_.at(idx);
        assert(val >= 0);
        return static_cast<std::size_t>(val);
    }

    constexpr bool operator==(const TensorShape& others) const {
        if (this == &others) {
            return true;
        }

        return dims_ == others.dims_;
    }

private:
    std::array<int32_t, 4> dims_;
};

// Element count times bytes-per-element, checked for size_t overflow. Shared
// by Tensor::size_bytes() and Tensor::from_cpu_ptr() so there's one place that
// knows how to compute a tensor's byte size safely.
inline std::size_t checked_size_bytes(const TensorShape& shape, DataType dtype) {
    const auto count = shape.numel();
    const auto elem_bytes = bytes_per_elem(dtype);
    if (elem_bytes != 0 && count > std::numeric_limits<std::size_t>::max() / elem_bytes) {
        throw std::overflow_error{"tensor size in bytes overflowed size_t"};
    }
    return count * elem_bytes;
}

class Tensor {
public:
    [[nodiscard]] std::string_view name() const { return name_; }

    [[nodiscard]] DeviceType device() const { return device_; }

    [[nodiscard]] DataType dtype() const { return dtype_; }

    [[nodiscard]] const void* data() const { return data_; }

    [[nodiscard]] TensorShape shape() const { return shape_; }

    [[nodiscard]] std::size_t size_bytes() const { return checked_size_bytes(shape_, dtype_); }

    /**
        Retrieve the given value from this tensor as a u16. May involve a blocking memcpy from
        GPU to CPU if this is a GPU tensor.

        If the dtype is not U16 or someting that can upconvert to it (U4, U8, etc), this will throw.
        If the indices are out of bound or wrong shape, std::invalid_argument will be thrown.
    */
    [[nodiscard]] std::uint16_t at_u16(std::initializer_list<std::size_t> indices) const;

    static Tensor from_ptr(std::string_view name, DeviceType device, DataType dtype,
                           std::span<const std::byte> base, TensorShape shape) {
        const auto expected_bytes = checked_size_bytes(shape, dtype);
        if (expected_bytes != base.size_bytes()) {
            throw std::invalid_argument{
                std::format("Expected tensor to be exactly {} bytes, got {}", expected_bytes,
                            base.size_bytes())};
        }
        return Tensor{name, device, dtype, base, shape};
    }

    // Borrows CPU memory; the backing storage must outlive this tensor and its copies.
    static Tensor from_cpu_ptr(std::string_view name, DataType dtype,
                               std::span<const std::byte> base, TensorShape shape) {
        return from_ptr(name, DeviceType::CPU, dtype, base, shape);
    }

private:
    Tensor(std::string_view name, DeviceType device, DataType dtype,
           std::span<const std::byte> base, TensorShape shape)
        : data_(base.data()), device_(device), dtype_(dtype), shape_(shape), name_(name) {}

    const void* data_;
    DeviceType device_;
    DataType dtype_;
    TensorShape shape_;
    std::string name_;
};

/**
 * Take a list of scalars and convert them to a rank-1 vector. They will be stored
 * in the given Arena (which implicitly picks device type as well)
 */
Tensor u16_from_scalars(Arena& arena, std::initializer_list<const std::uint16_t> indices,
                        std::optional<std::string_view> name = {});
} // namespace toby::tensors

template <> struct std::formatter<toby::tensors::TensorShape> : std::formatter<std::string> {
    auto format(const toby::tensors::TensorShape& s, std::format_context& ctx) const {
        std::string buf = "[";
        for (std::size_t i = 0; i < s.ndim(); ++i) {
            if (i != 0U) {
                buf += ", ";
            }
            buf += std::to_string(s[i]);
        }
        buf += "]";
        return std::formatter<std::string>::format(buf, ctx);
    }
};

namespace toby::tensors {
inline std::string to_string(const TensorShape& shape) {
    return std::format("{}", shape);
}
} // namespace toby::tensors
#endif
