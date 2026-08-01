
#include "toby/tokenize/vocab.hpp"
#include "token_list_detail.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace {
std::filesystem::path fixtures() {
    return std::filesystem::path{TOBY_TEST_FIXTURE_DIR};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
template <std::size_t N> constexpr std::span<const std::byte> literal_bytes(const char (&text)[N]) {
    return std::as_bytes(std::span{text}.template first<N - 1>());
}
} // namespace

using toby::tokenize::TokenId;
using toby::tokenize::Vocab;
using toby::tokenize::detail::TokenList;

using Catch::Matchers::RangeEquals;

namespace {
auto standard_vocab() {
    return std::make_shared<Vocab>(
        Vocab::load_gpt2({.vocab = fixtures() / "gpt2" / "vocab.json",
                          .merges = fixtures() / "gpt2" / "merges.txt"}));
}

auto build_token_list(std::span<const std::byte> input) {
    auto vocab = standard_vocab();
    return TokenList{vocab, input};
}
} // namespace

TEST_CASE("tokenlist works with empty list", "[tokenize][token_list]") {
    auto list = build_token_list(literal_bytes(""));
    REQUIRE(list.begin() == list.end());
}

TEST_CASE("tokenlist throws on invalid char", "[tokenize][token_list]") {
    REQUIRE_THROWS(build_token_list(literal_bytes(" QQQ")));
}

TEST_CASE("tokenlist converts bytes one-by-one at construct time", "[tokenize][token_list]") {
    auto list = build_token_list(literal_bytes(" a"));

    REQUIRE_THAT(list, RangeEquals(std::array{
                           TokenId{6},
                           TokenId{3},
                       }));
}
