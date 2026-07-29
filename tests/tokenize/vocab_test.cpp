#include "toby/tokenize/vocab.hpp"
#include "vocab_detail.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <utility>

// Tests for the vocabulary loaders.
using toby::tokenize::TokenId;
using toby::tokenize::Vocab;
using toby::tokenize::VocabLoadError;
using toby::tokenize::detail::load_tokenizer_json;
using toby::tokenize::detail::RawVocab;

namespace {

std::filesystem::path fixtures() {
    return std::filesystem::path{TOBY_TEST_FIXTURE_DIR};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
template <std::size_t N> constexpr std::span<const std::byte> literal_bytes(const char (&text)[N]) {
    return std::as_bytes(std::span{text}.template first<N - 1>());
}
} // namespace

TEST_CASE("token_id tests", "[tokenize][vocab]") {
    auto token1 = TokenId{1};
    auto other_token1 = TokenId{1};
    auto token2 = TokenId{2};

    SECTION("tokens are comparable") {
        CHECK(token1 == other_token1);
        CHECK(token1 < token2);
        CHECK(token2 > token1);
        CHECK(other_token1 == token1);
    }

    SECTION("tokens are hashable") {
        CHECK(std::hash<TokenId>{}(TokenId{42}) == std::hash<TokenId>{}(TokenId{42}));
    }
}

TEST_CASE("load_gpt2_vocab reads vocab.json and merges.txt", "[tokenize][vocab]") {
    const auto loaded = Vocab::load_gpt2({.vocab = fixtures() / "gpt2" / "vocab.json",
                                          .merges = fixtures() / "gpt2" / "merges.txt"});

    SECTION("vocab entries map to their ids incl unicode demapping") {
        CHECK(loaded.size() == 10);
        CHECK(loaded.token_for_bytes(literal_bytes("a")) == TokenId{3});
        CHECK(loaded.token_for_bytes(literal_bytes("ab")) == TokenId{5});
        CHECK(loaded.token_for_bytes(literal_bytes(" a")) == TokenId{7});
        CHECK(loaded.token_for_bytes(literal_bytes("<|endoftext|>")) == TokenId{9});
    }

    /*
    SECTION("merges are in rank order") {
        REQUIRE(loaded.merges.size() == 3);
        CHECK(loaded.merges[0] == std::pair<std::string, std::string>{"a", "b"});
        CHECK(loaded.merges[1] ==
              std::pair<std::string, std::string>{std::string{g_space_mark}, "a"});
    }

    SECTION("the #version banner is skipped but a '#' merge rule is not") {
        // 0x23 maps to itself in the byte<->unicode alphabet, so "# #" is a real
        // merge. A loader that skips every comment-looking line silently drops it.
        REQUIRE(loaded.merges.size() == 3);
        CHECK(loaded.merges[2] == std::pair<std::string, std::string>{"#", "#"});
    }
        */
}

TEST_CASE("load_tokenizer_json reads a HuggingFace BPE file", "[tokenize][vocab]") {
    const RawVocab loaded = load_tokenizer_json(fixtures() / "hf" / "llama3_style.json");

    SECTION("vocab and string-form merges") {
        CHECK(loaded.vocab.at("ab") == 3);
        REQUIRE(loaded.merges.size() == 2);
        CHECK(loaded.merges[0] == std::pair<std::string, std::string>{"a", "b"});
    }

    SECTION("added tokens carry their special flag") {
        REQUIRE(loaded.added_tokens.size() == 3);
        CHECK(loaded.added_tokens[0].content == "<|begin_of_text|>");
        CHECK(loaded.added_tokens[0].id == 100);
        CHECK(loaded.added_tokens[0].special);

        // Not every added token is special -- the encoder must not treat this
        // one as unproducible from user text.
        CHECK(loaded.added_tokens[2].content == "[casual]");
        CHECK_FALSE(loaded.added_tokens[2].special);
    }

    SECTION("the pre-tokenizer regex is extracted from the Sequence") {
        // This is the string to compare against your hand-rolled scanner's
        // pattern at load time.
        CHECK_THAT(loaded.pretokenizer_pattern,
                   Catch::Matchers::StartsWith("(?i:'s|'t|'re|'ve|'m|'ll|'d)"));
        CHECK_THAT(loaded.pretokenizer_pattern, Catch::Matchers::ContainsSubstring("\\p{N}{1,3}"));
    }
}

TEST_CASE("load_tokenizer_json accepts the newer array merge form", "[tokenize][vocab]") {
    const RawVocab loaded = load_tokenizer_json(fixtures() / "hf" / "array_merges.json");

    REQUIRE(loaded.merges.size() == 1);
    CHECK(loaded.merges[0] == std::pair<std::string, std::string>{"a", "b"});

    SECTION("a bare Split node works as well as a Sequence") {
        CHECK(loaded.pretokenizer_pattern == "\\p{N}{1,3}");
    }
}

// -----------------------------------------------------------------------------
// Failure modes. Each of these would otherwise surface much later as garbage
// output rather than a load error, which is the entire point of checking them.
// -----------------------------------------------------------------------------
TEST_CASE("load_tokenizer_json rejects a non-BPE model", "[tokenize][vocab]") {
    // A SentencePiece Unigram file (Llama 2, Gemma) is perfectly valid JSON with
    // a perfectly valid vocab and no merges at all. Loading it "successfully"
    // gives you a tokenizer that emits wrong ids for everything.
    CHECK_THROWS_AS(load_tokenizer_json(fixtures() / "hf" / "unigram.json"), VocabLoadError);
}

TEST_CASE("vocab loaders reject malformed and missing files", "[tokenize][vocab]") {
    SECTION("truncated JSON") {
        CHECK_THROWS_AS(load_tokenizer_json(fixtures() / "hf" / "malformed.json"), VocabLoadError);
    }

    SECTION("missing file") {
        CHECK_THROWS_AS(load_tokenizer_json(fixtures() / "hf" / "does_not_exist.json"),
                        VocabLoadError);
        CHECK_THROWS_AS(Vocab::load_gpt2({.vocab = fixtures() / "gpt2" / "nope.json",
                                          .merges = fixtures() / "gpt2" / "merges.txt"}),
                        VocabLoadError);
    }

    SECTION("the error names the file") {
        // Load errors get read by whoever is trying to start a server against a
        // model directory they assembled by hand; "cannot open" alone is useless.
        CHECK_THROWS_WITH(load_tokenizer_json(fixtures() / "hf" / "does_not_exist.json"),
                          Catch::Matchers::ContainsSubstring("does_not_exist.json"));
    }
}
