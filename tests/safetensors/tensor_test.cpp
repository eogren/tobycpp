#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>

using toby::tensors::DataType;
using toby::tensors::Tensor;
using toby::tensors::TensorShape;

TEST_CASE("TensorShape reports ndim and dims for a partial shape", "[tensor_shape]") {
    std::array<std::size_t, 2> dims{2, 3};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.ndim() == 2);
    CHECK(shape[0] == 2);
    CHECK(shape[1] == 3);
}

TEST_CASE("TensorShape supports the full 4 dims", "[tensor_shape]") {
    std::array<std::size_t, 4> dims{1, 2, 3, 4};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.ndim() == 4);
    CHECK(shape[0] == 1);
    CHECK(shape[1] == 2);
    CHECK(shape[2] == 3);
    CHECK(shape[3] == 4);
}

TEST_CASE("TensorShape supports a scalar (0-dim) shape", "[tensor_shape]") {
    std::array<std::size_t, 0> dims{};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.ndim() == 0);
}

TEST_CASE("TensorShape rejects more than 4 dims", "[tensor_shape]") {
    std::array<std::size_t, 5> dims{1, 2, 3, 4, 5};
    CHECK_THROWS_AS(TensorShape{std::span{dims}}, std::invalid_argument);
}

TEST_CASE("TensorShape numel is 1 for a scalar (0-dim) shape", "[tensor_shape]") {
    std::array<std::size_t, 0> dims{};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.numel() == 1);
}

TEST_CASE("TensorShape numel is the product of all dims", "[tensor_shape]") {
    std::array<std::size_t, 3> dims{2, 3, 4};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.numel() == 24);
}

TEST_CASE("TensorShape numel works for a single dim", "[tensor_shape]") {
    std::array<std::size_t, 1> dims{7};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.numel() == 7);
}

TEST_CASE("TensorShape numel works for the full 4 dims", "[tensor_shape]") {
    std::array<std::size_t, 4> dims{1, 2, 3, 4};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.numel() == 24);
}

TEST_CASE("TensorShape numel is 0 when any dim is 0", "[tensor_shape]") {
    std::array<std::size_t, 3> dims{2, 0, 4};
    const TensorShape shape{std::span{dims}};

    CHECK(shape.numel() == 0);
}

TEST_CASE("TensorShape numel throws on overflow", "[tensor_shape]") {
    constexpr auto max_dim = static_cast<std::size_t>(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> dims{max_dim, max_dim, max_dim};
    CHECK_THROWS_AS(TensorShape{std::span{dims}}, std::overflow_error);
}

TEST_CASE("Tensor size_bytes is numel times bytes-per-element", "[tensor]") {
    std::array<std::size_t, 2> dims{2, 3};
    const TensorShape shape{std::span{dims}};

    std::array<std::byte, 24> buf{};
    const Tensor t = Tensor::from_cpu_ptr("weight", DataType::F32, std::span{buf}, shape);

    CHECK(t.size_bytes() == 24);
}

TEST_CASE("Tensor size_bytes accounts for element width", "[tensor]") {
    std::array<std::size_t, 2> dims{2, 3};
    const TensorShape shape{std::span{dims}};

    std::array<std::byte, 12> buf{};
    const Tensor t = Tensor::from_cpu_ptr("weight", DataType::U16, std::span{buf}, shape);

    CHECK(t.size_bytes() == 12);
}

TEST_CASE("Tensor from_cpu_ptr rejects a buffer of the wrong size", "[tensor]") {
    std::array<std::size_t, 2> dims{2, 3};
    const TensorShape shape{std::span{dims}};

    std::array<std::byte, 23> buf{};
    CHECK_THROWS_AS(Tensor::from_cpu_ptr("weight", DataType::F32, std::span{buf}, shape),
                    std::invalid_argument);
}

TEST_CASE("TensorShape formats as a bracketed, comma-separated list", "[tensor_shape]") {
    std::array<std::size_t, 3> dims{4, 8, 16};
    const TensorShape shape{std::span{dims}};

    CHECK(std::format("{}", shape) == "[4, 8, 16]");
    CHECK(toby::tensors::to_string(shape) == "[4, 8, 16]");
}

TEST_CASE("TensorShape formats a scalar shape as an empty list", "[tensor_shape]") {
    std::array<std::size_t, 0> dims{};
    const TensorShape shape{std::span{dims}};

    CHECK(std::format("{}", shape) == "[]");
}

TEST_CASE("TensorShape basic equality", "[tensor_shape]") {
    CHECK(TensorShape{4} == TensorShape{4});
    CHECK(TensorShape{4} != TensorShape{3});
    CHECK(TensorShape{4, 1} != TensorShape{4});
}

TEST_CASE("Can initialize tensor from scalars", "[tensor]") {
    auto arena = toby::tensors::Arena::alloc_anonymous(4 * sizeof(std::uint16_t),
                                                       toby::tensors::DeviceType::CPU);
    REQUIRE(arena.get() != nullptr);
    auto tensor = toby::tensors::u16_from_scalars(*arena, {1, 2, 3, 4});
    CHECK(tensor.dtype() == DataType::U16);
    CHECK(tensor.device() == toby::tensors::DeviceType::CPU);
    CHECK(tensor.shape() == TensorShape{4});
    CHECK(tensor.at_u16({0}) == 1);
    CHECK(tensor.at_u16({1}) == 2);
    CHECK(tensor.at_u16({2}) == 3);
    CHECK(tensor.at_u16({3}) == 4);
}
