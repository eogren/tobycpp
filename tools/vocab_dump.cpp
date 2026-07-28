// vocab_dump: load a vocabulary file and print what the loader actually saw.
//
// The unit tests prove the parsers handle their branches; this is for pointing
// them at a real 50k-entry file and eyeballing the result.
//
//   $ ./tools/fetch_vocab.py gpt2
//   $ ./build/clang-debug/bin/vocab_dump gpt2 vocab/gpt2/vocab.json vocab/gpt2/merges.txt
//   $ ./build/clang-debug/bin/vocab_dump hf vocab/qwen3/tokenizer.json
//
// Also the quickest way to see the byte<->unicode representation for yourself:
// the sample entries print raw, so a space really does show up as "Ġ".

#include "toby/tokenize/vocab.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Sorted by id so successive runs print the same thing: RawVocab::vocab is an
// unordered_map, and iterating it raw would reshuffle the sample every run and
// make two dumps impossible to diff.
void print_sample_tokens(const toby::tokenize::RawVocab& loaded, const std::size_t count) {
    std::vector<std::pair<std::int32_t, std::string_view>> by_id;
    by_id.reserve(loaded.vocab.size());
    for (const auto& [token, id] : loaded.vocab) {
        by_id.emplace_back(id, token);
    }
    std::ranges::sort(by_id);

    for (const auto& [id, token] : std::span{by_id}.first(std::min(count, by_id.size()))) {
        std::println("    {:>6}  {}", id, token);
    }
}

void dump(const toby::tokenize::RawVocab& loaded) {
    std::println("  vocab entries : {}", loaded.vocab.size());
    std::println("  merge rules   : {}", loaded.merges.size());
    std::println("  added tokens  : {}", loaded.added_tokens.size());

    if (loaded.pretokenizer_pattern.empty()) {
        std::println("  pattern       : (none declared)");
    } else {
        std::println("  pattern       : {}", loaded.pretokenizer_pattern);
    }

    std::println("\n  first tokens by id:");
    print_sample_tokens(loaded, 8);

    if (!loaded.merges.empty()) {
        std::println("\n  highest-priority merges:");
        for (const auto& [left, right] :
             std::span{loaded.merges}.first(std::min<std::size_t>(5, loaded.merges.size()))) {
            std::println("    {} + {}", left, right);
        }
    }

    if (!loaded.added_tokens.empty()) {
        std::println("\n  added tokens:");
        for (const auto& token : std::span{loaded.added_tokens}.first(
                 std::min<std::size_t>(12, loaded.added_tokens.size()))) {
            // The special flag is the one that decides whether user text may
            // ever produce this id, so make it the loud part of the line.
            std::println("    {:>6}  {:<9} {}", token.id, token.special ? "SPECIAL" : "ordinary",
                         token.content);
        }
    }
}

int run(const std::span<const std::string_view> args) {
    if (args.size() == 4 && args[1] == "gpt2") {
        std::println("gpt2: {} + {}", args[2], args[3]);
        dump(toby::tokenize::load_gpt2_vocab(args[2], args[3]));
        return 0;
    }

    if (args.size() == 3 && args[1] == "hf") {
        std::println("hf: {}", args[2]);
        dump(toby::tokenize::load_tokenizer_json(args[2]));
        return 0;
    }

    std::println(stderr, "usage: vocab_dump gpt2 <vocab.json> <merges.txt>");
    std::println(stderr, "       vocab_dump hf   <tokenizer.json>");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string_view> args;
        args.reserve(static_cast<std::size_t>(argc));
        for (const char* arg : std::span{argv, static_cast<std::size_t>(argc)}) {
            args.emplace_back(arg);
        }
        return run(args);
    } catch (const std::exception& e) {
        // A VocabLoadError here is the normal outcome of pointing this at the
        // wrong file, so report it as a message rather than a crash.
        std::fputs("vocab_dump: ", stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    }
}
