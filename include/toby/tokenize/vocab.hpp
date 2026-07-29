#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

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

struct TokenId {
    std::uint32_t value;

    TokenId() = delete;

    constexpr explicit TokenId(std::uint32_t val) noexcept : value{val} {}

    auto operator<=>(const TokenId&) const = default;
};

static_assert(std::constructible_from<TokenId, std::uint32_t>);
static_assert(!std::default_initializable<TokenId>);
static_assert(!std::convertible_to<std::uint32_t, TokenId>);

class Vocab {
public:
    /// Load a GPT-2 style vocab from the given files.
    [[nodiscard]] static Vocab load_gpt2(const Gpt2VocabFiles& files);

    /// Size of the vocab
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return all_tokens_.size();
    }

    /// @brief  Lookup token by the given ID. Returns the bytes associated with it.
    /// @param id Token id
    /// @return Bytes associated with the token id or nullopt if the token doesn't exist
    [[nodiscard]] std::optional<std::span<const std::byte>> lookup_token(TokenId id) const;

    [[nodiscard]] std::optional<TokenId> token_for_bytes(std::span<const std::byte> bytes) const noexcept;

private:
    /// @brief  All tokens contained in the vocab file. String_views returned by the public
    /// API end up pointing into here.
    std::vector<std::byte> all_tokens_;

    std::vector<std::span<const std::byte>> token_to_bytes_;
    std::vector<std::pair<std::span<std::byte>, TokenId>> bytes_to_token_id_;
};

} // namespace toby::tokenize

template <> struct std::hash<toby::tokenize::TokenId> {
    std::size_t operator()(toby::tokenize::TokenId id) const noexcept {
        return std::hash<std::uint32_t>{}(id.value);
    }
};
