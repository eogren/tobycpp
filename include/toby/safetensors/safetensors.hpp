#ifndef SAFETENSORS_SAFETENSORS_HPP
#define SAFETENSORS_SAFETENSORS_HPP

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/file.hpp"
#include "toby/safetensors/tensor.hpp"

#include <concepts>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <ranges>
#include <string_view>
#include <tuple>
#include <vector>

namespace toby::tensors {

namespace detail {
std::tuple<std::unique_ptr<Arena>, std::vector<Tensor>>
parse_safetensors(const std::filesystem::path& in);

std::tuple<std::unique_ptr<Arena>, std::vector<Tensor>>
safetensors_to_arena(const std::vector<Tensor>& tensors, DeviceType device_type);
} // namespace detail

/**
    Parse the given safe tensors file and hand back a set of tensors +
    the arena they are mapped to.

    Func is used to exclude given tensors from loading.
*/
template <typename Func>
    requires std::predicate<Func, std::string_view>
std::tuple<std::unique_ptr<Arena>, std::vector<Tensor>>
load_safetensors(const std::filesystem::path& in, DeviceType device_type, Func pred) {
    // parse safetensor, get list ofe tensor
    auto [arena, parsed_tensors] = detail::parse_safetensors(in);
    auto the_pred = [pred](const Tensor& t) { return pred(t.name()); };
    auto filtered_tensors =
        std::ranges::to<std::vector>(parsed_tensors | std::views::filter(the_pred));

    return detail::safetensors_to_arena(filtered_tensors, device_type);
}
} // namespace toby::tensors
#endif
