#include "catch2/catch_message.hpp"
#include "utf8_detail.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using toby::tokenize::detail::Utf8CodePoint;

void check_decoding(std::string_view text, std::initializer_list<Utf8CodePoint> expected) {
    std::vector<Utf8CodePoint> actual;

    toby::tokenize::detail::for_each_utf8_code_point(
        text, [&](Utf8CodePoint point) { actual.push_back(point); });

    REQUIRE(actual.size() == expected.size());

    std::size_t index = 0;

    for (const auto& [seen, wanted] : std::views::zip(actual, expected)) {
        CAPTURE(index);

        CHECK(seen.value == wanted.value);
        CHECK(seen.byte_begin == wanted.byte_begin);
        CHECK(seen.byte_end == wanted.byte_end);

        ++index;
    }
}
} // namespace

TEST_CASE("for_each_utf8 can decode gpt-2 style codepoints", "[tokenize][utf8]") {
    // U+0120 LATIN CAPITAL LETTER G WITH DOT ABOVE, space, a encoded as UTF-8.
    const std::string gpt_merge = "\xC4\xA0 a";
    check_decoding(gpt_merge, {
                                  {.value = 0x0120, .byte_begin = 0, .byte_end = 2},
                                  {.value = 0x0020, .byte_begin = 2, .byte_end = 3},
                                  {.value = 0x0061, .byte_begin = 3, .byte_end = 4},
                              });
}

TEST_CASE("for_each_utf8 rejects malformed UTF-8", "[tokenize][utf8]") {
    const std::string malformed = "\xC4";

    CHECK_THROWS_AS(
        toby::tokenize::detail::for_each_utf8_code_point(malformed, [](Utf8CodePoint) {}),
        std::invalid_argument);
}
