#ifndef TOBY_TOKENIZE_UTF8_DETAIL_HPP
#define TOBY_TOKENIZE_UTF8_DETAIL_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unicode/umachine.h>
#include <unicode/utf8.h>

namespace toby::tokenize::detail {

struct Utf8CodePoint {
    /// Value of the UTF8 code point
    std::uint32_t value;

    /// What byte in the original string is this code point?
    std::size_t byte_begin;

    /// Ending byte for the codepoint (exclusive)
    std::size_t byte_end;
};

template <typename Callback>
    requires std::invocable<Callback&, Utf8CodePoint>
void for_each_utf8_code_point(std::string_view text, Callback callback) {
    constexpr auto max_icu_length =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());

    if (text.size() > max_icu_length) {
        throw std::length_error{"UTF-8 input is too large"};
    }

    const auto length = static_cast<std::int32_t>(text.size());
    std::int32_t offset = 0;

    while (offset < length) {
        const std::int32_t byte_begin = offset;

        UChar32 code_point{};
        // NOLINTNEXTLINE(readability-simplify-subscript-expr, bugprone-inc-dec-in-conditions)
        U8_NEXT(text.data(), offset, length, code_point);

        if (code_point < 0) {
            throw std::invalid_argument{"ill-formed UTF-8"};
        }

        std::invoke(callback, Utf8CodePoint{
                                  .value = static_cast<std::uint32_t>(code_point),
                                  .byte_begin = static_cast<std::size_t>(byte_begin),
                                  .byte_end = static_cast<std::size_t>(offset),
                              });
    }
}

} // namespace toby::tokenize::detail

#endif // TOBY_TOKENIZE_UTF8_DETAIL_HPP
