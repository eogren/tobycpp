#include "support/temp_file.hpp"
#include "toby/safetensors/safetensors.hpp"
#include "toby/safetensors/tensor.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr std::filesystem::path fixtures() {
    return std::filesystem::path{TOBY_TEST_FIXTURE_DIR};
}

constexpr std::filesystem::path good_safetensor() {
    return fixtures() / "safetensors" / "basic.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path scalar_safetensor() {
    return fixtures() / "safetensors" / "scalar.safetensors";
}

constexpr std::filesystem::path empty_tensor_safetensor() {
    return fixtures() / "safetensors" / "empty_tensor.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path truncated_data_safetensor() {
    return fixtures() / "safetensors" / "truncated_data.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path missing_shape_safetensor() {
    return fixtures() / "safetensors" / "missing_shape.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path invalid_dtype_type_safetensor() {
    return fixtures() / "safetensors" / "invalid_dtype_type.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path invalid_shape_type_safetensor() {
    return fixtures() / "safetensors" / "invalid_shape_type.safetensors";
}

[[maybe_unused]] constexpr std::filesystem::path invalid_offsets_type_safetensor() {
    return fixtures() / "safetensors" / "invalid_offsets_type.safetensors";
}
} // namespace

using toby::tensors::DataType;
using toby::tensors::DeviceType;
using toby::tensors::load_safetensors;
using toby::tensors::Tensor;
using toby::test::TempFile;

namespace {
template <typename T> std::vector<T> values(const Tensor& tensor) {
    std::vector<T> result(tensor.size_bytes() / sizeof(T));
    std::memcpy(result.data(), tensor.data(), tensor.size_bytes());
    return result;
}
} // namespace

TEST_CASE("safetensors loads tensor metadata and payload values", "[safetensors]") {
    auto [arena, tensors] =
        load_safetensors(good_safetensor(), DeviceType::CPU, [](std::string_view) { return true; });

    REQUIRE(arena != nullptr);
    REQUIRE(tensors.size() == 3);

    const auto scalar = std::ranges::find(tensors, "scalar", &Tensor::name);
    REQUIRE(scalar != tensors.end());
    CHECK(scalar->dtype() == DataType::F32);
    CHECK(scalar->shape().ndim() == 0);
    CHECK(values<float>(*scalar) == std::vector{1.5F});

    const auto token_ids = std::ranges::find(tensors, "token_ids", &Tensor::name);
    REQUIRE(token_ids != tensors.end());
    CHECK(token_ids->dtype() == DataType::U16);
    REQUIRE(token_ids->shape().ndim() == 1);
    CHECK(token_ids->shape()[0] == 4);
    CHECK(values<std::uint16_t>(*token_ids) == std::vector<std::uint16_t>{50256, 20, 30, 40});

    const auto projection = std::ranges::find(tensors, "projection", &Tensor::name);
    REQUIRE(projection != tensors.end());
    CHECK(projection->dtype() == DataType::F32);
    REQUIRE(projection->shape().ndim() == 2);
    CHECK(projection->shape()[0] == 2);
    CHECK(projection->shape()[1] == 3);
    CHECK(values<float>(*projection) == std::vector{-1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F});
}

TEST_CASE("safetensors loads selected tensor values", "[safetensors]") {
    auto [arena, tensors] =
        load_safetensors(good_safetensor(), DeviceType::CPU,
                         [](const std::string_view name) { return name == "token_ids"; });

    REQUIRE(arena != nullptr);
    REQUIRE(tensors.size() == 1);
    CHECK(tensors.front().name() == "token_ids");
    CHECK(values<std::uint16_t>(tensors.front()) == std::vector<std::uint16_t>{50256, 20, 30, 40});
}

TEST_CASE("safetensors permits an empty filtered result", "[safetensors]") {
    auto [arena, tensors] = load_safetensors(good_safetensor(), DeviceType::CPU,
                                             [](std::string_view) { return false; });

    REQUIRE(arena != nullptr);
    CHECK(arena->byte_span().empty());
    CHECK(tensors.empty());
}

TEST_CASE("safetensors loads a zero-element tensor", "[safetensors]") {
    auto [arena, tensors] = load_safetensors(empty_tensor_safetensor(), DeviceType::CPU,
                                             [](std::string_view) { return true; });

    REQUIRE(arena != nullptr);
    REQUIRE(tensors.size() == 1);
    CHECK(tensors.front().name() == "empty");
    CHECK(tensors.front().shape().ndim() == 1);
    CHECK(tensors.front().shape()[0] == 0);
    CHECK(tensors.front().size_bytes() == 0);
}

TEST_CASE("safetensors rejects a file shorter than its length field", "[safetensors]") {
    const TempFile short_file{"short"};

    CHECK_THROWS_AS(
        load_safetensors(short_file.path(), DeviceType::CPU, [](std::string_view) { return true; }),
        std::invalid_argument);
}

TEST_CASE("safetensors rejects malformed tensors instead of loading partially", "[safetensors]") {
    const auto load_all = [](const std::filesystem::path& path) {
        return load_safetensors(path, DeviceType::CPU, [](std::string_view) { return true; });
    };

    CHECK_THROWS(load_all(truncated_data_safetensor()));
    CHECK_THROWS(load_all(missing_shape_safetensor()));
    CHECK_THROWS(load_all(invalid_dtype_type_safetensor()));
    CHECK_THROWS(load_all(invalid_shape_type_safetensor()));
    CHECK_THROWS(load_all(invalid_offsets_type_safetensor()));
}
