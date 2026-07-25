#include "toby/tokenize/pretokenizer.hpp"

#include <algorithm>
#include <catch2/catch_get_random_seed.hpp> // Catch::getSeed
#include <catch2/catch_message.hpp>         // INFO, CAPTURE
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Tests for the hand-rolled pre-tokenizer. Tests live OUTSIDE the protected
// zone, so this file is scaffolding the assistant set up -- but the *expected
// splits* are yours to fill in from a reference (e.g. run Python `tiktoken` or
// the `regex` module on the same input) so the assertions reflect behavior you
// actually understand and intend.

using toby::tokenize::pretokenize;

namespace {

// Every chunk is supposed to be a *slice of the input*: it points into the same
// buffer and stays in bounds. Check that with pointer arithmetic BEFORE anyone
// reads the bytes. A view with a bogus pointer or a size that underflowed to
// ~2^64 then reports its own offset and length, instead of detonating deep
// inside std::string with libc++'s famously unhelpful "basic_string" (that is
// just std::length_error saying "you asked for more than max_size()").
// Pointer arithmetic only -- deliberately never dereferences, so it is safe to
// call on a chunk that has not been validated yet.
bool chunk_is_slice(std::string_view input, std::string_view chunk) {
    const char* const begin = input.data();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const char* const end = begin + input.size();

    return chunk.data() >= begin && chunk.data() <= end &&
           chunk.size() <= static_cast<std::size_t>(end - chunk.data());
}

void require_chunks_are_slices(std::string_view input,
                               const std::vector<std::string_view>& chunks) {
    const char* const begin = input.data();

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const std::string_view chunk = chunks[i];
        INFO("chunk[" << i << "] of " << chunks.size() << ": data() is " << (chunk.data() - begin)
                      << " bytes into a " << input.size() << "-byte input, size() is "
                      << chunk.size());
        REQUIRE(chunk_is_slice(input, chunk));
    }
}

// Render a validated split as ["a"]["b"] so a failing comparison shows where
// the boundaries actually landed, not just that two strings differ.
std::string describe(const std::vector<std::string_view>& chunks) {
    std::string out;
    for (const std::string_view chunk : chunks) {
        out += '[';
        out += chunk;
        out += ']';
    }
    return out;
}

// Concatenating the chunks must reproduce the input exactly. Chunks are
// non-owning views into the input; std::string has no free operator+ for
// string_view, so build up with += rather than std::accumulate.
std::string concat(const std::vector<std::string_view>& chunks) {
    std::string out;
    for (const std::string_view chunk : chunks) {
        out += chunk;
    }
    return out;
}

} // namespace

// -----------------------------------------------------------------------------
// Property test: reconstruction. This holds no matter *where* you decide the
// chunk boundaries go, so it doesn't give away the algorithm -- but it's a
// strong red/green signal and catches the classic bugs (dropping a leading
// space, eating a trailing newline, off-by-one at a class boundary).
//
// GENERATE rather than a for loop: Catch2 reruns the case once per input and
// names the input in the failure header, so one bad input doesn't hide the
// others behind an early abort.
// -----------------------------------------------------------------------------
TEST_CASE("pretokenize is lossless: chunks concatenate back to the input",
          "[tokenize][pretokenizer]") {
    // The last four are the interesting ones: whitespace that is NOT U+0020
    // sitting *between* two non-whitespace chars. Every input above them either
    // has no interior whitespace or ends its whitespace run at end-of-input,
    // which is the one case the give-back rule skips.
    const std::string_view input =
        GENERATE(as<std::string_view>{}, "", "hello", " hello world", "don't stop", "abc123def",
                 "hi!!!  there\n", "   leading and trailing   ", "a\tb", "a\t\tb", "a\nb", "a \nb");
    CAPTURE(input);

    // Separate step so a throw from inside pretokenize is attributed to
    // pretokenize (with its own message) rather than to the comparison.
    std::vector<std::string_view> chunks;
    REQUIRE_NOTHROW(chunks = pretokenize(input));

    require_chunks_are_slices(input, chunks);

    INFO("split: " << describe(chunks));
    CHECK(concat(chunks) == std::string{input});
}

// -----------------------------------------------------------------------------
// Exact-split cases: one section per rule the GPT-2/cl100k pattern encodes.
// Each is a checklist item for your implementation. Fill in the expected chunks
// from a reference, delete the SKIP, and turn it green.
//
// (SKIP keeps these honestly "not yet verified" instead of falsely passing.)
// -----------------------------------------------------------------------------
TEST_CASE("pretokenize splits match the GPT-2 rules", "[tokenize][pretokenizer]") {
    SECTION("a leading space attaches to the following word") {
        CHECK(pretokenize(" hello world") == std::vector<std::string_view>{" hello", " world"});
    }

    SECTION("letters and digits are separate runs") {
        CHECK(pretokenize("abc123def") == std::vector<std::string_view>{"abc", "123", "def"});
    }

    SECTION("contractions split off") {
        CHECK(pretokenize("don't") == std::vector<std::string_view>{"don", "\'t"});
    }

    SECTION("runs of punctuation are their own 'other' chunk") {
        CHECK(pretokenize("hi!!!") == std::vector<std::string_view>{"hi", "!!!"});
    }

    SECTION("whitespace runs and the trailing-space rule") {
        CHECK(pretokenize("a   b") == std::vector<std::string_view>{"a", "  ", " b"});
    }

    SECTION("trailing whitespace is accounted for") {
        CHECK(pretokenize("a   ") == std::vector<std::string_view>{"a", "   "});
    }

    SECTION("leading whitespace is attributed correctly") {
        CHECK(pretokenize("  a") == std::vector<std::string_view>{" ", " a"});
    }
}

// Once the contraction alternatives have failed to match, `'` is just another
// [^\s\p{L}\p{N}] char, so it shares a chunk with neighboring punctuation rather
// than standing alone. `end_quotation_block` runs before the class scan, so the
// contraction cases are already settled by the time this rule applies.
TEST_CASE("pretokenize groups a quote with adjacent punctuation", "[tokenize][pretokenizer]") {
    SECTION("string and quotes") {
        CHECK(pretokenize("hi!'") == std::vector<std::string_view>{"hi", "!'"});
    }

    SECTION("lots of punctuation") {
        CHECK(pretokenize("!'!") == std::vector<std::string_view>{"!'!"});
    }
}

// -----------------------------------------------------------------------------
// Characterization test: this pins CURRENT behavior, not desired behavior.
//
// The reference pattern treats "café" as one letter run; toby throws instead.
// `pretokenize` is not declared noexcept, so on a server this is reachable from
// untrusted input -- which is the reason to pin it rather than leave it implicit.
//
// When Unicode handling lands, this test should change or be deleted. It exists
// so the throw is a documented choice rather than a surprise.
// -----------------------------------------------------------------------------
TEST_CASE("pretokenize currently rejects non-ASCII input", "[tokenize][pretokenizer]") {
    CHECK_THROWS_AS(pretokenize("café"), std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Randomized property fuzz. Deliberately oracle-FREE: it asserts only the three
// things that must hold no matter where the boundaries land, so it never encodes
// an expected split. That is enough to catch the failure modes that hurt --
// a cursor that stops advancing (caught by the CTest TIMEOUT, since a stalled
// scan hangs rather than returning), a dropped or duplicated byte, and a view
// pointing outside the input.
//
// For exact-split checking against the real GPT-2 pattern, see
// tools/pretokenizer_diff.py -- that needs a Python reference, which is why it
// lives outside the C++ suite.
//
// Seeded from Catch2's RNG, so the seed is fixed by default (no flaky suite),
// printed in the failure header, and overridable: `--rng-seed 12345` explores
// new inputs and reproduces whatever it finds. Assertion count is kept to two
// per iteration on the happy path -- the detailed per-chunk diagnostics only
// fire once something is actually wrong.
// -----------------------------------------------------------------------------
TEST_CASE("pretokenize holds its invariants on random input", "[tokenize][pretokenizer][fuzz]") {
    // ASCII only: a byte >= 0x80 throws by design, see the test above.
    constexpr std::string_view alphabet = "aBz19 \t\n\r\v\f'\"!.,-_#*()[]{}<>/@$%^&+=~;:?";
    constexpr std::size_t iterations = 20000;
    constexpr std::size_t max_len = 64;

    std::mt19937 rng{Catch::getSeed()};
    std::uniform_int_distribution<std::size_t> len_dist{0, max_len};
    std::uniform_int_distribution<std::size_t> char_dist{0, alphabet.size() - 1};

    for (std::size_t n = 0; n < iterations; ++n) {
        std::string input;
        const std::size_t len = len_dist(rng);
        for (std::size_t i = 0; i < len; ++i) {
            input.push_back(alphabet[char_dist(rng)]);
        }

        INFO("iteration " << n << " of " << iterations << " (rerun with --rng-seed "
                          << Catch::getSeed() << ")");
        CAPTURE(input);

        std::vector<std::string_view> chunks;
        REQUIRE_NOTHROW(chunks = pretokenize(input));

        // Validate every view before anything reads its bytes: concat() below
        // would happily walk off the end of a chunk whose size underflowed.
        const bool all_slices = std::ranges::all_of(
            chunks, [&input](const std::string_view c) { return chunk_is_slice(input, c); });
        if (!all_slices) {
            require_chunks_are_slices(input, chunks); // reports which chunk, then aborts
        }

        REQUIRE(concat(chunks) == input);
    }
}
