#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Vocabulary loading (SCAFFOLDING -- file-format plumbing, not the algorithm).
//
// Reads the two shapes of BPE vocabulary file you will actually meet:
//
//   * GPT-2 style: `vocab.json` (token -> id) plus `merges.txt` (one merge pair
//     per line, ordered by rank). This is what HuggingFace's `openai-community/gpt2`
//     repo ships.
//   * HuggingFace `tokenizer.json`: one file holding vocab, merges, added tokens
//     and the pre-tokenizer spec. This is what Llama 3 and Qwen ship.
//
// Both produce a `RawVocab`. Same merge engine, different parsers -- which is
// the whole reason these are two free functions returning one type rather than
// a loader interface with virtual dispatch.
//
// -----------------------------------------------------------------------------
// WHAT THIS DELIBERATELY DOES NOT DO
//
// It does not decode the byte<->unicode representation.
//
// Byte-level BPE cannot store raw bytes in a JSON string -- 0x00 and friends are
// not valid UTF-8 text. So GPT-2 defines a bijection from all 256 byte values
// onto printable codepoints, and the vocab file stores tokens in *that* alphabet.
// It is why you see "Ġhello" in a vocab file: U+0120 is the stand-in for 0x20,
// a space. Llama 3 and Qwen use the identical mapping.
//
// The strings below are therefore spelled exactly as the file spells them --
// UTF-8 of the mapped codepoints, NOT the token's real bytes. Turning "Ġhello"
// back into " hello" is the decode step, and it is yours to write: it is a small
// function, it is the thing that makes the vocab mean something, and building it
// is how the representation stops being magic. The reference is `bytes_to_unicode()`
// in OpenAI's gpt-2 encoder.py (and in HF's ByteLevel pre-tokenizer).
//
// Concretely: `vocab` keys here are NOT the byte sequences you will look up
// during merging. Decode them once at load and build your own lookup table.
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

/// A token added outside the BPE vocabulary proper.
///
/// These matter for the reason discussed at length elsewhere: `special` tokens
/// (`<|eot_id|>`, `<|im_start|>`) must never be producible from user text. This
/// struct only reports what the file declares -- enforcing that split is the
/// encoder's job, and it needs this flag to do it.
struct AddedToken {
    std::int32_t id{};
    std::string content;
    bool special{};
};

/// A vocabulary exactly as spelled on disk.
///
/// "Raw" is the load-bearing word: nothing here has been byte-decoded, and the
/// strings are in the file's mapped alphabet (see the header comment). This is a
/// faithful parse, not a tokenizer.
struct RawVocab {
    /// Token text -> id. Roughly 50k entries for GPT-2, ~128k for Llama 3.
    std::unordered_map<std::string, std::int32_t> vocab;

    /// Merge pairs in rank order: `merges[0]` is the highest-priority merge.
    /// The index IS the rank -- BPE picks the lowest-ranked applicable pair, so
    /// preserving file order is the entire contract of this field.
    std::vector<std::pair<std::string, std::string>> merges;

    /// Specials and other out-of-band tokens. Empty for plain GPT-2 files, which
    /// carry `<|endoftext|>` inside `vocab` itself rather than listing it here.
    std::vector<AddedToken> added_tokens;

    /// The pre-tokenizer regex declared by a `tokenizer.json`, verbatim, or empty
    /// if the file did not declare one (always empty for the GPT-2 pair, which
    /// has no machine-readable pattern -- it is implied by the format).
    ///
    /// You are not going to run this. It is here so you can *compare* it against
    /// the pattern you hand-rolled and refuse to load on a mismatch. That turns
    /// "silently tokenizes differently from what the model was trained on" --
    /// which degrades output in ways that look like a bad model, not a bug --
    /// into a loud error at startup.
    std::string pretokenizer_pattern;
};

/// Load a GPT-2 style pair. `merges_txt` is ranked by line order; a leading
/// `#version:` line is ignored.
///
/// Throws VocabLoadError.
[[nodiscard]] RawVocab load_gpt2_vocab(const std::filesystem::path& vocab_json,
                                       const std::filesystem::path& merges_txt);

/// Load a HuggingFace `tokenizer.json` (Llama 3, Qwen, and anything else built
/// with the `tokenizers` library).
///
/// Accepts both merge encodings in the wild: the older `"a b"` strings and the
/// `["a", "b"]` arrays that tokenizers >= 0.20 emits.
///
/// Throws VocabLoadError. In particular it throws when `model.type` is not BPE:
/// a SentencePiece Unigram file (Llama 2, Gemma) parses as JSON perfectly well
/// and would otherwise load as a vocabulary with no merges.
[[nodiscard]] RawVocab load_tokenizer_json(const std::filesystem::path& tokenizer_json);

} // namespace toby::tokenize
