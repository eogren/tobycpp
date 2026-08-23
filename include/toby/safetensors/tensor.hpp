#ifndef SAFETENSORS_TENSOR_HPP
#define SAFETENSORS_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace toby::tensors {
enum class DeviceType : std::uint8_t { CPU, GPU };
enum class DataType : std::uint8_t { F32, U16 };

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
    constexpr TensorShape(std::span<uint32_t> dims) : dims_{-1, -1, -1, -1} {
        if (dims.size() > 4) {
            throw std::invalid_argument{"only up to 4 dims supported"};
        }

        std::ranges::transform(dims, dims_.begin(), [](const uint32_t n) {
            if (n > std::numeric_limits<std::int32_t>::max()) {
                throw std::invalid_argument{std::format("dim {} too big", n)};
            }

            return static_cast<std::int32_t>(n);
        });
    }

    [[nodiscard]] constexpr std::size_t ndim() const {
        const auto* it = std::ranges::find(dims_, -1);
        return static_cast<std::size_t>(it - dims_.begin());
    }

    [[nodiscard]] constexpr std::size_t numel() const {
        auto dims = ndim();
        std::size_t ret = 1;

        for (std::size_t i = 0; i < dims; i++) {
            const auto d = static_cast<std::size_t>(operator[](i));
            if (d != 0 && ret > std::numeric_limits<std::size_t>::max() / d) {
                throw std::overflow_error{"TensorShape::numel overflowed size_t"};
            }
            ret *= d;
        }

        return ret;
    }

    constexpr std::uint32_t operator[](std::size_t idx) const {
        auto val = dims_.at(idx);
        assert(val >= 0);
        return static_cast<std::uint32_t>(val);
    }

private:
    std::array<int32_t, 4> dims_;
};

// Element count times bytes-per-element, checked for size_t overflow. Shared
// by Tensor::size_bytes() and Tensor::from_ptr() so there's one place that
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

    [[nodiscard]] DataType dtype() const { return dtype_; }

    [[nodiscard]] const void* data() const { return data_; }

    [[nodiscard]] TensorShape shape() const { return shape_; }

    [[nodiscard]] std::size_t size_bytes() const { return checked_size_bytes(shape_, dtype_); }

    static Tensor from_ptr(std::string_view name, DataType dtype, std::span<const std::byte> base,
                           TensorShape shape) {
        const auto expected_bytes = checked_size_bytes(shape, dtype);
        if (expected_bytes != base.size_bytes()) {
            throw std::invalid_argument{
                std::format("Expected tensor to be exactly {} bytes, got {}", expected_bytes,
                            base.size_bytes())};
        }
        return Tensor{name, dtype, base, shape};
    }

private:
    Tensor(std::string_view name, DataType dtype, std::span<const std::byte> base,
           TensorShape shape)
        : data_(base.data()), dtype_(dtype), shape_(shape), name_(name) {}

    const void* data_;
    DataType dtype_;
    TensorShape shape_;
    std::string name_;
};

template <typename T> class TypedTensor {
private:
    T* data_; // points inside an arena somewhere
    std::array<uint32_t, 4> shape_;
};
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
