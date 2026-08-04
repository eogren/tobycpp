#include "toby/tokenize/gpt2_encoder.hpp"
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
#include <vector>

using toby::tokenize::Vocab;

namespace {
std::filesystem::path fixtures() {
    return std::filesystem::path{TOBY_TEST_FIXTURE_DIR};
}

auto standard_vocab() {
    return std::make_shared<const Vocab>(
        Vocab::load_gpt2({.vocab = fixtures() / "gpt2" / "vocab.json",
                          .merges = fixtures() / "gpt2" / "merges.txt"}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
template <std::size_t N> constexpr std::span<const std::byte> literal_bytes(const char (&text)[N]) {
    return std::as_bytes(std::span{text}.template first<N - 1>());
}
} // namespace

using Catch::Matchers::RangeEquals;
using toby::tokenize::Gpt2Encoder;
using toby::tokenize::TokenId;

TEST_CASE("Encoding string with no merges only tokenizes it", "[tokenize][gpt2_encoder]") {
    auto encoder = Gpt2Encoder{standard_vocab()};

    std::vector<TokenId> ret{};
    encoder.encode_one(literal_bytes("#!"), std::back_inserter(ret));
    REQUIRE_THAT(ret, RangeEquals(std::array{
                          TokenId{2},
                          TokenId{0},
                      }));
}

TEST_CASE("Encode basic string", "[tokenize][gpt2_encoder]") {
    auto encoder = Gpt2Encoder{standard_vocab()};

    std::vector<TokenId> ret{};
    // Should merge:
    // <space>a -> new token
    // b
    encoder.encode_one(literal_bytes("ab"), std::back_inserter(ret));
    REQUIRE_THAT(ret, RangeEquals(std::array{
                          TokenId{5},
                      }));
}

TEST_CASE("Encode multiple times", "[tokenize][gpt2_encoder]") {
    auto encoder = Gpt2Encoder{standard_vocab()};

    std::vector<TokenId> ret{};
    // Should merge:
    // <space>a -> new token
    // b
    encoder.encode_one(literal_bytes(" ab"), std::back_inserter(ret));
    // should merge:
    // <6, 3, 4> -> <7, 4> -> <9>
    REQUIRE_THAT(ret, RangeEquals(std::array{
                          TokenId{9},
                      }));
}

TEST_CASE("Encode multiple times - left first", "[tokenize][gpt2_encoder]") {
    auto encoder = Gpt2Encoder{standard_vocab()};

    std::vector<TokenId> ret{};
    // Should merge:
    // <space>a -> new token
    // b
    encoder.encode_one(literal_bytes("ab#"), std::back_inserter(ret));
    // should merge:
    // <3, 4, 2> -> <5, 2> -> <10>
    REQUIRE_THAT(ret, RangeEquals(std::array{
                          TokenId{10},
                      }));
}
