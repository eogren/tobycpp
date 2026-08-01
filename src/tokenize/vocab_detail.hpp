#ifndef TOBY_TOKENIZE_VOCAB_DETAIL_HPP
#define TOBY_TOKENIZE_VOCAB_DETAIL_HPP

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Private vocabulary-loader representation.
//
// This header is shared only with loader tests and the vocab_dump developer
// tool. Production consumers should include toby/tokenize/vocab.hpp; the raw
// on-disk alphabet is not a usable tokenizer vocabulary.

namespace toby::tokenize::detail {

struct AddedToken {
    std::int32_t id{};
    std::string content;
    bool special{};
};

struct RawVocab {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
    std::vector<AddedToken> added_tokens;
    std::string pretokenizer_pattern;
};

[[nodiscard]] RawVocab load_gpt2_vocab(const std::filesystem::path& vocab_json,
                                       const std::filesystem::path& merges_txt);

[[nodiscard]] RawVocab load_tokenizer_json(const std::filesystem::path& tokenizer_json);

} // namespace toby::tokenize::detail

#endif // TOBY_TOKENIZE_VOCAB_DETAIL_HPP
