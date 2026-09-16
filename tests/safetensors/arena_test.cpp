#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/memory_transfer.hpp"
#include "toby/safetensors/tensor_types.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

using toby::tensors::Arena;
using toby::tensors::DeviceType;

namespace {
bool is_aligned(const void* ptr, std::size_t alignment) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    return (iptr & (alignment - 1)) == 0;
}

template <typename T1, typename T2, std::size_t ExtentA, std::size_t ExtentB>
bool is_subspan(std::span<T1, ExtentA> spanA, std::span<T2, ExtentB> spanB) {
    if (spanA.empty()) {
        return true; // An empty span is technically a subset of any span
    }
    if (spanB.empty()) {
        return false;
    }

    const std::byte* start_a = std::as_bytes(spanA).data();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::byte* end_a = spanA.data() + spanA.size();

    const std::byte* start_b = std::as_bytes(spanB).data();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::byte* end_b = spanB.data() + spanB.size();

    // Check if memory addresses of A fall completely inside the boundaries of B
    return (start_a >= start_b) && (end_a <= end_b);
}

std::vector<std::byte> to_cpu(std::span<const std::byte> data, DeviceType data_device) {
    std::vector<std::byte> ret(data.size_bytes());
    toby::tensors::copy_bytes(ret, DeviceType::CPU, data, data_device);
    return ret;
}
} // namespace

// Replace each SKIP with your setup and assertions as you define the contract.
// Run just these cases with: ./build/clang-debug/bin/toby_safetensors_tests "[arena]"

TEST_CASE("Arena CPU construction and metadata", "[arena][cpu]") {
    auto arena = Arena::alloc_anonymous(256, DeviceType::CPU, "arena-test");
    REQUIRE(arena != nullptr);

    CHECK(arena->device() == DeviceType::CPU);
    CHECK(arena->name() == "arena-test");
    CHECK(arena->byte_span().size_bytes() == 0);
}

TEST_CASE("Arena allocation and alignment", "[arena][cpu]") {
    auto arena = Arena::alloc_anonymous(256, DeviceType::CPU, "arena-test");
    REQUIRE(arena != nullptr);

    auto first = arena->alloc(1);
    auto second = arena->alloc(129);

    // should be out of space already!
    CHECK_THROWS(arena->alloc(1));

    CHECK(first.size_bytes() == 1);
    CHECK(is_aligned(first.data(), 64));
    CHECK(is_subspan(first, arena->byte_span()));

    CHECK(second.size_bytes() == 129);
    CHECK(is_aligned(second.data(), 64));
    CHECK(is_subspan(second, arena->byte_span()));
}

TEST_CASE("Arena capacity boundaries", "[arena][cpu]") {
    auto arena = Arena::alloc_anonymous(258, DeviceType::CPU, "arena-test");
    REQUIRE(arena != nullptr);

    auto first = arena->alloc(258);
    CHECK(first.size_bytes() == 258);
}

TEST_CASE("Arena allocation arithmetic limits", "[arena][cpu]") {
    auto arena = Arena::alloc_anonymous(258, DeviceType::CPU, "arena-test");
    REQUIRE(arena != nullptr);
    CHECK_THROWS(arena->alloc(std::numeric_limits<std::size_t>::max()));
}

TEST_CASE("Arena copies from CPU memory", "[arena][cpu]") {
    // TODO(you): Supply a byte buffer to alloc_from_cpu_ptr() and check the
    // returned storage. What should happen when the source changes afterward?
    std::array<std::byte, 4> data{static_cast<std::byte>(1), static_cast<std::byte>(2),
                                  static_cast<std::byte>(3), static_cast<std::byte>(4)};
#if TOBY_HAVE_CUDA
    auto device_type = GENERATE(DeviceType::CPU, DeviceType::GPU);
#else
    auto device_type = GENERATE(DeviceType::CPU);
#endif
    auto arena = Arena::alloc_anonymous(4, device_type);
    auto span = arena->alloc_from_cpu_ptr(data);
    data[0] = static_cast<std::byte>(10);

    CHECK(span.size_bytes() == 4);

    // the span might point into device memory...
    auto check_arena = Arena::alloc_anonymous(4, DeviceType::CPU);
    auto bytes = to_cpu(span, device_type);
    std::vector<std::byte> expected{static_cast<std::byte>(1), static_cast<std::byte>(2),
                                    static_cast<std::byte>(3), static_cast<std::byte>(4)};
    CHECK(bytes == expected);
}
