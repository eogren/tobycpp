#include "gpt2/kernels.h"
#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

using toby::tensors::Arena;
using toby::tensors::DeviceType;
using toby::tensors::Tensor;
using toby::tensors::TensorShape;

namespace {
struct ArenaFixture {
    std::unique_ptr<toby::tensors::Arena> cpu =
        toby::tensors::Arena::alloc_anonymous(1024UZ * 1024, toby::tensors::DeviceType::CPU);

#if TOBY_HAVE_CUDA
    std::unique_ptr<toby::tensors::Arena> gpu =
        toby::tensors::Arena::alloc_anonymous(1024UZ * 1024, toby::tensors::DeviceType::GPU);
#endif

    Tensor create_f32_tensor(std::string_view name, DeviceType device, TensorShape shape,
                             std::initializer_list<float> values) {
        return create_f32_tensor(name, device, shape, std::span<const float>{values});
    }

    Tensor create_f32_tensor(std::string_view name, DeviceType device, TensorShape shape,
                             std::span<const float> values) {
        if (values.size() != shape.numel()) {
            throw std::invalid_argument{
                std::format("Tensor of shape {} must be initilaized with {} elements, but got {}",
                            shape, shape.numel(), values.size())};
        }

        if (shape.ndim() != 1) {
            throw std::invalid_argument{"Right now we only support 1D tensor creation"};
        }

        auto& arena = (device == DeviceType::GPU) ? gpu : cpu;

        return toby::tensors::f32_from_scalars(*arena, values, name);
    }

    Tensor create_u16_tensor(std::string_view name, DeviceType device, TensorShape shape,
                             std::initializer_list<const std::uint16_t> values) {
        return create_u16_tensor(name, device, shape, std::span<const std::uint16_t>{values});
    }

    Tensor create_u16_tensor(std::string_view name, DeviceType device, TensorShape shape,
                             std::span<const std::uint16_t> values) {
        if (values.size() != shape.numel()) {
            throw std::invalid_argument{
                std::format("Tensor of shape {} must be initilaized with {} elements, but got {}",
                            shape, shape.numel(), values.size())};
        }

        if (shape.ndim() != 1) {
            throw std::invalid_argument{"Right now we only support 1D tensor creation"};
        }

        auto& arena = (device == DeviceType::GPU) ? gpu : cpu;

        return toby::tensors::u16_from_scalars(*arena, values, name);
    }
};

} // namespace

// GPT-2 kernel test cases go here.
TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different DTypes", "[tensor_equality]") {
#if TOBY_HAVE_CUDA
    const auto device = GENERATE(DeviceType::CPU, DeviceType::GPU);
#else
    const auto device = GENERATE(DeviceType::CPU);
#endif

    auto t1 = create_f32_tensor("f32", device, TensorShape({3}), {1.0, 2.0, 3.0});
    auto t2 = create_u16_tensor("u16", device, TensorShape{3}, {1, 2, 3});

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different Devices", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
#if TOBY_HAVE_CUDA
    const auto device1 = GENERATE(DeviceType::CPU, DeviceType::GPU);
    const auto device2 = GENERATE(DeviceType::CPU, DeviceType::GPU);

    if (device1 != device2) {
        auto t1 = create_f32_tensor("f32", device1, TensorShape({3}), {1.0, 2.0, 3.0});
        auto t2 = create_f32_tensor("f32", device2, TensorShape({3}), {1.0, 2.0, 3.0});

        CHECK_THROWS(toby::gpt2::tensors_equal(t1, t2));
    }
#endif
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different Shapes", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
#if TOBY_HAVE_CUDA
    const auto device = GENERATE(DeviceType::CPU, DeviceType::GPU);
#else
    const auto device = GENERATE(DeviceType::CPU);
#endif

    auto t1 = create_f32_tensor("f32", device, TensorShape({3}), {1.0, 2.0, 3.0});
    auto t2 = create_f32_tensor("f32", device, TensorShape({4}), {1.0, 2.0, 3.0, 4.0});

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: u16", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
#if TOBY_HAVE_CUDA
    const auto device = GENERATE(DeviceType::CPU, DeviceType::GPU);
#else
    const auto device = GENERATE(DeviceType::CPU);
#endif

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<std::uint16_t> values(1028);
    std::iota(values.begin(), values.end(), 1); // NOLINT(modernize-use-ranges)

    auto t1 = create_u16_tensor("u16_1", device, TensorShape({values.size()}), values);
    auto t2 = create_u16_tensor("u16_2", device, TensorShape({values.size()}), values);

    CHECK(toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_inequality: u16", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
#if TOBY_HAVE_CUDA
    const auto device = GENERATE(DeviceType::CPU, DeviceType::GPU);
#else
    const auto device = GENERATE(DeviceType::CPU);
#endif

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<std::uint16_t> values(1028);
    std::iota(values.begin(), values.end(), 1); // NOLINT(modernize-use-ranges)

    auto t1 = create_u16_tensor("u16_1", device, TensorShape({values.size()}), values);
    values[1] = 9999;
    auto t2 = create_u16_tensor("u16_2", device, TensorShape({values.size()}), values);

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}
