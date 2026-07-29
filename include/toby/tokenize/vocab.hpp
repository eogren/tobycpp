#pragma once

#include <filesystem>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Vocabulary functions
// -----------------------------------------------------------------------------

namespace toby::tokenize {

/// Thrown for anything wrong with a vocabulary file: missing, unreadable,
/// malformed JSON, or structurally valid JSON that is not a vocabulary.
///
/// One type for all of it on purpose -- every case is "this file is unusable,
/// refuse to load the model", and a caller has no useful way to react
/// differently. The message carries the detail.
class VocabLoadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Gpt2VocabFiles {
    std::filesystem::path vocab;
    std::filesystem::path merges;
};

class Vocab {
public:
    /// Load a GPT-2 style vocab from the given files.
    [[nodiscard]] static Vocab load_gpt2(const Gpt2VocabFiles& files);
};
} // namespace toby::tokenize
