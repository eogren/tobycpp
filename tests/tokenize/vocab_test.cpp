#include "toby/tokenize/vocab.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

// Tests for the vocabulary loaders. Unlike the pre-tokenizer suite, these assert
// exact values freely: this is file-format plumbing the assistant wrote, not the
// algorithm you are here to learn, so pinning it hard costs you nothing.
//
// Fixtures are deliberately tiny and synthetic (tests/fixtures/). A real
// vocab.json is ~1MB and would make a failure unreadable; every branch worth
// testing is reachable from ten entries.

using toby::tokenize::load_gpt2_vocab;
using toby::tokenize::load_tokenizer_json;
using toby::tokenize::RawVocab;
using toby::tokenize::VocabLoadError;

namespace {

std::filesystem::path fixtures() {
    return std::filesystem::path{TOBY_TEST_FIXTURE_DIR};
}

// U+0120, UTF-8 encoded: the byte<->unicode alphabet's stand-in for a space,
// which is why vocab files are full of "Ġhello".
//
// Named rather than spelled inline at each use: writing "\xc4\xa0" "a" needs
// adjacent-literal concatenation to stop the compiler reading "\xa0a" as a
// single (overlong) hex escape, and that is exactly the kind of subtlety a
// formatter is entitled to flatten.
constexpr std::string_view g_space_mark = "\xc4\xa0";

std::string mapped(std::string_view rest) {
    return std::string{g_space_mark} + std::string{rest};
}

} // namespace

TEST_CASE("load_gpt2_vocab reads vocab.json and merges.txt", "[tokenize][vocab]") {
    const RawVocab loaded =
        load_gpt2_vocab(fixtures() / "gpt2" / "vocab.json", fixtures() / "gpt2" / "merges.txt");

    SECTION("vocab entries map to their ids") {
        CHECK(loaded.vocab.size() == 10);
        CHECK(loaded.vocab.at("a") == 3);
        CHECK(loaded.vocab.at("ab") == 5);
        CHECK(loaded.vocab.at("<|endoftext|>") == 9);
    }

    SECTION("tokens keep the file's byte<->unicode spelling, undecoded") {
        // "Ġa" is the file's way of writing " a". If this ever comes back as
        // " a", something started decoding behind the loader's back and the
        // header's contract is broken.
        CHECK(loaded.vocab.contains(mapped("a")));
        CHECK_FALSE(loaded.vocab.contains(" a"));
    }

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

    SECTION("GPT-2 files declare no pattern and no added tokens") {
        CHECK(loaded.pretokenizer_pattern.empty());
        CHECK(loaded.added_tokens.empty());
    }
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
        CHECK_THROWS_AS(
            load_gpt2_vocab(fixtures() / "gpt2" / "nope.json", fixtures() / "gpt2" / "merges.txt"),
            VocabLoadError);
    }

    SECTION("the error names the file") {
        // Load errors get read by whoever is trying to start a server against a
        // model directory they assembled by hand; "cannot open" alone is useless.
        CHECK_THROWS_WITH(load_tokenizer_json(fixtures() / "hf" / "does_not_exist.json"),
                          Catch::Matchers::ContainsSubstring("does_not_exist.json"));
    }
}
