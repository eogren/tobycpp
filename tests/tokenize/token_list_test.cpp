
#include "toby/tokenize/detail/token_list.hpp"
#include "toby/tokenize/vocab.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstddef>
#include <filesystem>
#include <iterator>
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
    return std::make_shared<const Vocab>(
        Vocab::load_gpt2({.vocab = fixtures() / "gpt2" / "vocab.json",
                          .merges = fixtures() / "gpt2" / "merges.txt"}));
}

auto build_token_list(std::span<const std::byte> input) {
    auto vocab = standard_vocab();
    return TokenList{*vocab, input};
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

TEST_CASE("tokenlist merge successful case", "[tokenize][token_list]") {
    auto list = build_token_list(literal_bytes(" ab"));

    REQUIRE_THAT(list, RangeEquals(std::array{
                           TokenId{6},
                           TokenId{3},
                           TokenId{4},
                       }));

    auto it = list.begin();
    std::advance(it, 1);
    it.merge_with_neighbor(TokenId{7});

    REQUIRE_THAT(list, RangeEquals(std::array{
                           TokenId{6},
                           TokenId{7},
                       }));
}

TEST_CASE("tokenlist merge throws at end", "[tokenize][token_list]") {
    auto list = build_token_list(literal_bytes(" a"));
    auto it = list.begin();
    std::advance(it, 1);
    REQUIRE_THROWS(it.merge_with_neighbor(TokenId{7}));
}

TEST_CASE("tokenlist iterator comparisons", "[tokenize][token_list]") {
    auto list = build_token_list(literal_bytes(" a"));
    auto list2 = build_token_list(literal_bytes(" a"));

    auto it = list.begin();
    auto it2 = list.begin();
    auto next = std::next(it);
    auto otherit = list2.begin();

    CHECK(it == it2);
    CHECK(it < next);
    CHECK(it2 < next);
    CHECK(next > it);
    CHECK(next > it2);
    CHECK(otherit != it);
}
