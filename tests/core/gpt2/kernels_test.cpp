#include "toby/gpt2/kernels.h"
#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#if TOBY_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
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
std::vector<DeviceType> available_devices() {
    std::vector<DeviceType> devices{DeviceType::CPU};
#if TOBY_HAVE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
        devices.push_back(DeviceType::GPU);
    }
#endif
    return devices;
}

struct ArenaFixture {
    std::unique_ptr<toby::tensors::Arena> cpu =
        toby::tensors::Arena::alloc_anonymous(1024UZ * 1024, toby::tensors::DeviceType::CPU);

#if TOBY_HAVE_CUDA
    std::unique_ptr<toby::tensors::Arena> gpu;
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
#if TOBY_HAVE_CUDA
        if (device == DeviceType::GPU && !arena) {
            arena = toby::tensors::Arena::alloc_anonymous(1024UZ * 1024, DeviceType::GPU);
        }
#endif

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
#if TOBY_HAVE_CUDA
        if (device == DeviceType::GPU && !arena) {
            arena = toby::tensors::Arena::alloc_anonymous(1024UZ * 1024, DeviceType::GPU);
        }
#endif

        return toby::tensors::u16_from_scalars(*arena, values, name);
    }
};

} // namespace

// GPT-2 kernel test cases go here.
TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different DTypes", "[tensor_equality]") {
    const auto device = GENERATE_COPY(from_range(available_devices()));

    auto t1 = create_f32_tensor("f32", device, TensorShape({3}), {1.0, 2.0, 3.0});
    auto t2 = create_u16_tensor("u16", device, TensorShape{3}, {1, 2, 3});

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different Devices", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
#if TOBY_HAVE_CUDA
    const auto device1 = GENERATE_COPY(from_range(available_devices()));
    const auto device2 = GENERATE_COPY(from_range(available_devices()));

    if (device1 != device2) {
        auto t1 = create_f32_tensor("f32", device1, TensorShape({3}), {1.0, 2.0, 3.0});
        auto t2 = create_f32_tensor("f32", device2, TensorShape({3}), {1.0, 2.0, 3.0});

        CHECK_THROWS(toby::gpt2::tensors_equal(t1, t2));
    }
#endif
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: Different Shapes", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
    const auto device = GENERATE_COPY(from_range(available_devices()));

    auto t1 = create_f32_tensor("f32", device, TensorShape({3}), {1.0, 2.0, 3.0});
    auto t2 = create_f32_tensor("f32", device, TensorShape({4}), {1.0, 2.0, 3.0, 4.0});

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: u16", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
    const auto device = GENERATE_COPY(from_range(available_devices()));

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<std::uint16_t> values(1028);
    std::iota(values.begin(), values.end(), 1); // NOLINT(modernize-use-ranges)

    auto t1 = create_u16_tensor("u16_1", device, TensorShape({values.size()}), values);
    auto t2 = create_u16_tensor("u16_2", device, TensorShape({values.size()}), values);

    CHECK(toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_inequality: u16", "[tensor_equality]") {
    // if no cuda, different devices aren't possible so we can just skip it
    const auto device = GENERATE_COPY(from_range(available_devices()));

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<std::uint16_t> values(1028);
    std::iota(values.begin(), values.end(), 1); // NOLINT(modernize-use-ranges)

    auto t1 = create_u16_tensor("u16_1", device, TensorShape({values.size()}), values);
    values[1] = 9999;
    auto t2 = create_u16_tensor("u16_2", device, TensorShape({values.size()}), values);

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: f32", "[tensor_equality]") {
    const auto device = GENERATE_COPY(from_range(available_devices()));

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<float> values(1028);
    std::iota(values.begin(), values.end(), 1.0F); // NOLINT(modernize-use-ranges)

    auto t1 = create_f32_tensor("f32_1", device, TensorShape({values.size()}), values);
    auto t2 = create_f32_tensor("f32_2", device, TensorShape({values.size()}), values);

    CHECK(toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_inequality: f32", "[tensor_equality]") {
    const auto device = GENERATE_COPY(from_range(available_devices()));

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    std::vector<float> values(1028);
    std::iota(values.begin(), values.end(), 1.0F); // NOLINT(modernize-use-ranges)

    auto t1 = create_f32_tensor("f32_1", device, TensorShape({values.size()}), values);
    values.back() = 9999.0F;
    auto t2 = create_f32_tensor("f32_2", device, TensorShape({values.size()}), values);

    CHECK(!toby::gpt2::tensors_equal(t1, t2));
}

TEST_CASE_METHOD(ArenaFixture, "tensor_equality: f32 special values", "[tensor_equality]") {
    const auto device = GENERATE_COPY(from_range(available_devices()));

    INFO("device = " << (device == DeviceType::CPU ? "CPU" : "GPU"));
    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    SECTION("matching infinities and signed zero") {
        auto t1 = create_f32_tensor("f32_1", device, TensorShape{3}, {infinity, -infinity, 0.0F});
        auto t2 = create_f32_tensor("f32_2", device, TensorShape{3}, {infinity, -infinity, -0.0F});
        CHECK(toby::gpt2::tensors_equal(t1, t2));
    }

    SECTION("opposite infinities") {
        auto t1 = create_f32_tensor("f32_1", device, TensorShape{1}, {infinity});
        auto t2 = create_f32_tensor("f32_2", device, TensorShape{1}, {-infinity});
        CHECK(!toby::gpt2::tensors_equal(t1, t2));
    }

    SECTION("NaN compared with itself") {
        auto t1 = create_f32_tensor("f32_1", device, TensorShape{1}, {nan});
        auto t2 = create_f32_tensor("f32_2", device, TensorShape{1}, {nan});
        CHECK(!toby::gpt2::tensors_equal(t1, t2));
    }
}
