#pragma once

#include <stdexcept>

// -----------------------------------------------------------------------------
// Vocabulary (SCAFFOLDING -- file-format plumbing, not the algorithm).
//
// The parsed on-disk representation is deliberately private: its strings use
// GPT-2's printable byte alphabet and are not usable as tokenizer lookup keys.
// This public header will hold the decoded Vocab API once that representation
// exists. For now it exposes only the common load-error type.
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

} // namespace toby::tokenize
