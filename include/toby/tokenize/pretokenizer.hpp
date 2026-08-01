#ifndef TOBY_TOKENIZE_PRETOKENIZER_HPP
#define TOBY_TOKENIZE_PRETOKENIZER_HPP

#include <string_view>
#include <vector>

// -----------------------------------------------------------------------------
// Pre-tokenizer (NON-core, NON-protected): the first stage of the tokenizer.
//
// It splits raw input text into the coarse chunks that BPE then merges -- the
// job the GPT-2 / cl100k regex does. Because that pattern is fixed and known,
// you are hand-rolling it as a codepoint scan instead of pulling a regex engine
// (no \p{L} support in std::regex, no lookaround in re2). See notes in the .cpp.
//
// This header is SCAFFOLDING. The signature below is a *starting suggestion* so
// the test harness has something to link against -- it is yours to change.
// -----------------------------------------------------------------------------

namespace toby::tokenize {

/// Split `text` into pre-token chunks (letters / digits / "other" runs, with
/// the GPT-2 leading-space and whitespace rules). The concatenation of the
/// returned chunks should reproduce `text` exactly (no bytes dropped or added).
///
/// Each chunk is a contiguous slice of the input (the pre-tokenizer only slices,
/// it never rewrites bytes), so we hand back non-owning views to avoid copying.
///
/// LIFETIME: the returned views borrow `text`. They are valid only while the
/// buffer behind `text` outlives them -- do not return them past `text`'s scope.
///
/// Scope for now: ASCII only. A byte >= 0x80 is REJECTED -- the function throws
/// std::invalid_argument rather than classifying it. That is a placeholder, not
/// a design: the reference pattern treats "café" as a single letter run, so real
/// Unicode category tables (or at least a UTF-8 decode that lumps non-ASCII into
/// "other") still have to land here.
///
/// Note the consequence while it stands: this is reachable from any caller that
/// forwards untrusted bytes, and the throw is not part of the signature.
[[nodiscard]] std::vector<std::string_view> pretokenize(std::string_view text);

} // namespace toby::tokenize

#endif // TOBY_TOKENIZE_PRETOKENIZER_HPP
