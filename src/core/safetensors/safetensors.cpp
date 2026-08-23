#include "toby/safetensors/safetensors.hpp"

#include "cpu_arena.hpp"
#include "toby/safetensors/align.hpp"
#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/file.hpp"
#include "toby/safetensors/tensor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <ranges>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using json = nlohmann::json;
using std::to_string;

namespace {
constexpr size_t max_json_len = 1024L * 1024L;

constexpr toby::tensors::DataType to_datatype(std::string_view s) {
    if (s == "U16") {
        return toby::tensors::DataType::U16;
    }

    if (s == "F32") {
        return toby::tensors::DataType::F32;
    }

    throw std::invalid_argument{std::format("Unsupported datatype {}", s)};
}

constexpr std::size_t element_size(toby::tensors::DataType data_type) {
    switch (data_type) {
    case toby::tensors::DataType::F32:
        return 4;
    case toby::tensors::DataType::U16:
        return 2;
    }
}

constexpr toby::tensors::TensorShape parse_shape(const auto& shape) {
    if (!shape.is_array() || shape.size() > 4) {
        throw std::invalid_argument{
            std::format("Expected shape to be an array size 0-4, was {}", to_string(shape))};
    }

    std::array<std::uint32_t, 4> buf{};
    try {
        std::ranges::transform(shape, buf.begin(),
                               [](const json& dim) { return dim.get<std::uint32_t>(); });
    } catch (json::type_error& e) {
        throw std::invalid_argument{std::format("error parsing shapes array: {}", e.what())};
    }

    return toby::tensors::TensorShape{std::span{buf}.first(shape.size())};
}

std::span<const std::byte> get_data_pointer(std::span<const std::byte> base,
                                            toby::tensors::DataType data_type,
                                            const auto& offsets) {
    if (!offsets.is_array() || offsets.size() != 2) {
        throw std::invalid_argument{
            std::format("'data_offsets' must be a 2-element array, got {}", to_string(offsets))};
    }

    std::uint64_t start_offset{};
    std::uint64_t end_offset{};

    offsets[0].get_to(start_offset);
    offsets[1].get_to(end_offset);

    if (end_offset < start_offset) {
        throw std::invalid_argument{
            std::format("end offset {} is before start offset {}", end_offset, start_offset)};
    }

    auto size = end_offset - start_offset;
    if (size % element_size(data_type) != 0) {
        throw std::invalid_argument{
            std::format("expected size {} to be divisible by {}", size, element_size(data_type))};
    }

    if (start_offset > base.size() || size > base.size() - start_offset) {
        throw std::invalid_argument{
            std::format("data offsets [{}, {}] exceed the {}-byte tensor payload", start_offset,
                        end_offset, base.size())};
    }

    return base.subspan(start_offset, size);
}
} // namespace

namespace toby::tensors::detail {
std::tuple<std::unique_ptr<Arena>, std::vector<Tensor>>
parse_safetensors(const std::filesystem::path& in) {
    auto fd = open_file(in);
    auto arena = std::make_unique<CpuArena>(ScopedMapping::from_fd(fd.fd(), FileMode::Read));
    auto data = arena->byte_span();

    if (data.size() < sizeof(std::uint64_t)) {
        throw std::invalid_argument{std::format(
            "safetensors corrupt: file is {} bytes; expected an 8-byte header", data.size())};
    }

    std::uint64_t json_len{};
    std::memcpy(&json_len, data.data(), sizeof(json_len));

    if constexpr (std::endian::native == std::endian::big) {
        json_len = std::byteswap(json_len);
    }

    if (json_len > max_json_len) {
        throw std::invalid_argument{"JSON len parsed from header is {} - too large"};
    }

    if (data.size() < sizeof(json_len) || json_len > data.size() - sizeof(json_len)) {
        throw std::invalid_argument{std::format(
            "safetensors corrupt: header says json for {} bytes but file size is only {}", json_len,
            data.size())};
    }

    auto json_span = data.subspan(sizeof(json_len), json_len);
    auto json_as_string =
        std::string_view{reinterpret_cast<const char*>(json_span.data()), // NOLINT
                         json_span.size()};
    auto parsed = json::parse(json_as_string);
    if (!parsed.is_object()) {
        throw std::invalid_argument{
            std::format("Expected root of json area to be an object. Got {}", to_string(parsed))};
    }
    auto safetensor_data = data.subspan(sizeof(json_len) + json_len);

    auto safetensor_vecs = std::vector<Tensor>{};

    for (const auto& item : parsed.items()) {
        if (item.key() == "__metadata__") {
            continue;
        }

        const auto& tensor_name = item.key();
        const auto& tensor = item.value();

        if (!tensor.is_object()) {
            throw std::invalid_argument{
                std::format("safetensors tensor '{}': expected an object, got {}", tensor_name,
                            tensor.type_name())};
        }

        const auto dtype_it = tensor.find("dtype");
        const auto offsets_it = tensor.find("data_offsets");
        const auto shape_it = tensor.find("shape");
        if (dtype_it == tensor.end() || offsets_it == tensor.end() || shape_it == tensor.end()) {
            throw std::invalid_argument{std::format(
                "safetensors tensor '{}': missing dtype, data_offsets, or shape", tensor_name)};
        }

        try {
            const std::string_view dtype_view = dtype_it->get_ref<const std::string&>();
            auto dtype = to_datatype(dtype_view);
            auto shape = parse_shape(*shape_it);
            auto pointers = get_data_pointer(safetensor_data, dtype, *offsets_it);
            spdlog::debug("safetensors: found tensor '{}' dtype={} shape={} span=[{}, {}]",
                          tensor_name, to_string(*dtype_it), to_string(shape),
                          static_cast<const void*>(pointers.data()), pointers.size());
            safetensor_vecs.push_back(Tensor::from_ptr(tensor_name, dtype, pointers, shape));
        } catch (const std::exception& e) {
            throw std::invalid_argument{
                std::format("safetensors tensor '{}': {}", tensor_name, e.what())};
        }
    }

    return std::make_tuple(std::move(arena), std::move(safetensor_vecs));
}

namespace {
struct TensorMap {
    const void* src;
    std::size_t src_offset;
    std::size_t new_offset;
};
} // namespace

std::tuple<std::unique_ptr<Arena>, std::vector<Tensor>>
safetensors_to_arena(const std::vector<Tensor>& tensors, DeviceType device_type) {
    // 1. size arena and build offset map
    std::size_t next_idx{0};
    std::vector<MemcpyInfo> offsets;
    offsets.reserve(tensors.size());

    for (auto const& tensor : tensors) {
        offsets.emplace_back(tensor.data(), 0, next_idx, tensor.size_bytes());
        next_idx += align_up(tensor.size_bytes(), 64);
    }

    assert(offsets.size() == tensors.size());

    // 2. build it
    std::vector<Tensor> new_tensors;
    auto arena = Arena::alloc_anonymous(next_idx, device_type);
    arena->bulk_memcpy(offsets);

    for (auto [tensor, offset] : std::views::zip(tensors, offsets)) {
        new_tensors.push_back(Tensor::from_ptr(
            tensor.name(), tensor.dtype(),
            arena->byte_span().subspan(offset.new_offset, offset.size), tensor.shape()));
    }

    return std::make_tuple(std::move(arena), std::move(new_tensors));
}

} // namespace toby::tensors::detail
