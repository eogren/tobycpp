#include "toby/tokenize/vocab.hpp"

#include "utf8_detail.hpp"
#include "vocab_detail.hpp"

#include <algorithm>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_map_fwd.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// json.hpp is nlohmann's umbrella header; include-cleaner wants the internal
// detail header that actually declares this, which is not ours to depend on.
// NOLINTNEXTLINE(misc-include-cleaner)
using nlohmann::json;
using toby::tokenize::VocabLoadError;

// Opened in binary mode deliberately: merges.txt is line-oriented, but letting
// the platform rewrite line endings would silently corrupt a token whose bytes
// happen to map to \r. Trailing \r is stripped explicitly below instead.
std::string read_file(const std::filesystem::path& path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        throw VocabLoadError{std::format("cannot open {}", path.string())};
    }

    std::string contents{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
    if (in.bad()) {
        throw VocabLoadError{std::format("error reading {}", path.string())};
    }

    return contents;
}

json parse_json(const std::filesystem::path& path) {
    const std::string text = read_file(path);
    try {
        return json::parse(text);
    } catch (const json::exception& e) {
        throw VocabLoadError{std::format("{}: malformed JSON: {}", path.string(), e.what())};
    }
}

// Token ids index into the embedding matrix, so they must be non-negative and
// must fit the type we hand back.
std::int32_t to_token_id(const json& value, const std::filesystem::path& path,
                         std::string_view what) {
    if (!value.is_number_integer()) {
        throw VocabLoadError{std::format("{}: {}: expected an integer id, got {}", path.string(),
                                         what, value.type_name())};
    }

    const auto raw = value.get<std::int64_t>();
    if (raw < 0 || raw > std::numeric_limits<std::int32_t>::max()) {
        throw VocabLoadError{
            std::format("{}: {}: token id {} out of range", path.string(), what, raw)};
    }

    return static_cast<std::int32_t>(raw);
}

std::unordered_map<std::string, std::int32_t>
parse_vocab_object(const json& vocab, const std::filesystem::path& path) {
    if (!vocab.is_object()) {
        throw VocabLoadError{std::format("{}: vocab is not a JSON object", path.string())};
    }

    std::unordered_map<std::string, std::int32_t> out;
    out.reserve(vocab.size());

    for (const auto& [token, id] : vocab.items()) {
        out.emplace(token, to_token_id(id, path, std::format("vocab entry '{}'", token)));
    }

    return out;
}

// Split "left right" into its two halves.
//
// Splitting on the first space is unambiguous: in the byte<->unicode alphabet a
// real space is spelled U+0120, so neither half can contain U+0020.
std::pair<std::string, std::string> split_merge(std::string_view line,
                                                const std::filesystem::path& path,
                                                const std::size_t line_number) {
    const std::size_t space = line.find(' ');
    if (space == std::string_view::npos) {
        throw VocabLoadError{std::format("{}:{}: merge rule has no space separator: '{}'",
                                         path.string(), line_number, line)};
    }

    const std::string_view left = line.substr(0, space);
    const std::string_view right = line.substr(space + 1);

    if (left.empty() || right.empty() || right.contains(' ')) {
        throw VocabLoadError{std::format("{}:{}: merge rule is not exactly two parts: '{}'",
                                         path.string(), line_number, line)};
    }

    return {std::string{left}, std::string{right}};
}

// Find the first Regex pattern in a Split or nested Sequence.
// Literal String patterns are intentionally ignored.
std::string find_split_regex(const json& root) {
    // Depth is 2 in every real file; this only exists so a pathological one
    // terminates rather than walking forever.
    constexpr int max_nodes = 256;

    std::vector<const json*> worklist{&root};
    int visited = 0;

    while (!worklist.empty() && visited < max_nodes) {
        const json* const node = worklist.back();
        worklist.pop_back();
        ++visited;

        if (!node->is_object()) {
            continue;
        }

        const auto type = node->value("type", std::string{});

        if (type == "Split") {
            const auto pattern = node->find("pattern");
            if (pattern != node->end() && pattern->is_object()) {
                const auto regex = pattern->find("Regex");
                if (regex != pattern->end() && regex->is_string()) {
                    return regex->get<std::string>();
                }
            }
            continue;
        }

        if (type == "Sequence") {
            const auto nested = node->find("pretokenizers");
            if (nested != node->end() && nested->is_array()) {
                // Pushed in reverse so the stack pops them in document order:
                // the first Split wins, and "first" should mean first in the
                // file rather than an artifact of the traversal.
                for (auto child = nested->rbegin(); child != nested->rend(); ++child) {
                    worklist.push_back(&*child);
                }
            }
        }
    }

    return {};
}

std::vector<toby::tokenize::detail::AddedToken>
parse_added_tokens(const json& root, const std::filesystem::path& path) {
    const auto added = root.find("added_tokens");
    if (added == root.end() || !added->is_array()) {
        return {};
    }

    std::vector<toby::tokenize::detail::AddedToken> out;
    out.reserve(added->size());

    for (const auto& entry : *added) {
        if (!entry.is_object()) {
            throw VocabLoadError{
                std::format("{}: added_tokens entry is not an object", path.string())};
        }

        const auto id = entry.find("id");
        const auto content = entry.find("content");
        if (id == entry.end() || content == entry.end() || !content->is_string()) {
            throw VocabLoadError{
                std::format("{}: added_tokens entry missing id or content", path.string())};
        }

        out.push_back({.id = to_token_id(*id, path, "added_tokens entry"),
                       .content = content->get<std::string>(),
                       // Absent means ordinary: HF omits the flag on non-specials.
                       .special = entry.value("special", false)});
    }

    return out;
}

// Helper for byte_to_printable_map -- can a byte be encoded directly?
// Derived from bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"),
// ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
[[nodiscard]] constexpr bool can_encode_directly(std::uint8_t byte) {
    return ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
            (byte >= 174 && byte <= 255));
}

// Generate an array that maps a raw byte to a uint16_t representation.
// Equivalent of bytes_to_unicode() in https://github.com/openai/gpt-2/blob/master/src/encoder.py.
//
// We need this for the reverse map (given a codepoint in a vocab or merge file, translate it back
// to raw byte).
[[nodiscard]] constexpr auto byte_to_printable_map() {
    std::array<std::uint16_t, 256> ret{};
    static_assert(ret.size() <= 256);

    std::uint16_t to_add = 0;

    for (std::uint16_t i = 0; i < ret.size(); i++) { // NOLINT(bugprone-too-small-loop-variable)
        if (can_encode_directly(static_cast<std::uint8_t>(i))) {
            ret[i] = i; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index): static_assert
                        // guarantees this
        } else {
            ret[i] = 256 + to_add; // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            to_add++;
        }
    }

    return ret;
}

// Reverse of byte_to_printable_map.
[[nodiscard]] constexpr auto printable_to_byte_map() {
    std::array<std::optional<std::byte>, 324> ret{};

    constexpr auto forward_map = byte_to_printable_map();
    for (std::size_t i = 0; i < forward_map.size(); i++) {
        ret[forward_map[i]] = // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index)
            static_cast<std::byte>(i);
    }
    return ret;
}

static_assert(printable_to_byte_map()[0x20] == std::nullopt);
static_assert(printable_to_byte_map()[0x20 + 256] == std::byte{0x20});

template <typename T, std::size_t N> constexpr bool all_unique(const std::array<T, N>& values) {
    for (auto it = values.begin(); it != values.end();
         ++it) { // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic): bounded array iterator
        if (std::find(std::next(it), values.end(), *it) != values.end()) {
            return false;
        }
    }

    return true;
}

static_assert(all_unique(byte_to_printable_map()));
static_assert(byte_to_printable_map()[0x20] == 288);

[[nodiscard]] toby::tokenize::TokenId
find_token_or_throw(const toby::tokenize::detail::RawVocab& raw_vocab, const std::string& token) {
    auto it = raw_vocab.vocab.find(token);
    if (it == raw_vocab.vocab.end()) {
        throw VocabLoadError{std::format("Could not find tokenId for \"{}\"", token)};
    }

    return toby::tokenize::TokenId{static_cast<uint32_t>(it->second)};
}

[[nodiscard]] auto merges_to_map(const toby::tokenize::detail::RawVocab& raw_vocab) {
    std::uint32_t rank = 0;
    boost::unordered_flat_map<std::pair<toby::tokenize::TokenId, toby::tokenize::TokenId>,
                              toby::tokenize::Vocab::MergeTarget>
        ret{};
    for (const auto& merge : raw_vocab.merges) {
        auto first_token = find_token_or_throw(raw_vocab, merge.first);
        auto second_token = find_token_or_throw(raw_vocab, merge.second);
        auto new_token = find_token_or_throw(raw_vocab, merge.first + merge.second);

        auto merge_info = toby::tokenize::Vocab::MergeTarget{
            .new_token = new_token,
            .rank = rank,
        };

        rank++;

        ret.insert_or_assign(std::make_pair(first_token, second_token), merge_info);
    }

    return ret;
}
} // namespace

namespace toby::tokenize::detail {

// Two paths in a fixed order is the honest signature; swapping them fails loudly
// at parse time with the offending filename in the message, so a wrapper type per
// path would be ceremony without a payoff.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
RawVocab load_gpt2_vocab(const std::filesystem::path& vocab_json,
                         const std::filesystem::path& merges_txt) {
    RawVocab out;
    out.vocab = parse_vocab_object(parse_json(vocab_json), vocab_json);

    const std::string merges = read_file(merges_txt);

    std::size_t line_number = 0;
    std::size_t pos = 0;
    while (pos <= merges.size()) {
        const std::size_t newline = merges.find('\n', pos);
        const std::size_t end = newline == std::string::npos ? merges.size() : newline;

        std::string_view line{merges};
        line = line.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        ++line_number;

        // Only line 1 may be the version banner. A later '#' is a real token:
        // 0x23 maps to itself in the byte<->unicode alphabet, so "# #" is a
        // legitimate merge rule and skipping every '#' line would drop it.
        const bool is_version_banner = line_number == 1 && line.starts_with("#version");

        if (!line.empty() && !is_version_banner) {
            out.merges.push_back(split_merge(line, merges_txt, line_number));
        }

        if (newline == std::string::npos) {
            break;
        }
        pos = newline + 1;
    }

    return out;
}

RawVocab load_tokenizer_json(const std::filesystem::path& tokenizer_json) {
    const json root = parse_json(tokenizer_json);

    const auto model = root.find("model");
    if (model == root.end() || !model->is_object()) {
        throw VocabLoadError{std::format("{}: no \"model\" object", tokenizer_json.string())};
    }

    // Reject non-BPE up front. A Unigram/SentencePiece file (Llama 2, Gemma) is
    // valid JSON with a valid vocab and simply no merges, so without this check
    // it loads "successfully" and then tokenizes everything wrong.
    const auto type = model->value("type", std::string{});
    if (type != "BPE") {
        throw VocabLoadError{std::format(
            "{}: model.type is '{}', expected 'BPE' -- this tokenizer is not byte-level BPE "
            "and its vocabulary is not usable by this merge engine",
            tokenizer_json.string(), type)};
    }

    RawVocab out;

    const auto vocab = model->find("vocab");
    if (vocab == model->end()) {
        throw VocabLoadError{std::format("{}: no model.vocab", tokenizer_json.string())};
    }
    out.vocab = parse_vocab_object(*vocab, tokenizer_json);

    const auto merges = model->find("merges");
    if (merges == model->end() || !merges->is_array()) {
        throw VocabLoadError{std::format("{}: no model.merges array", tokenizer_json.string())};
    }

    out.merges.reserve(merges->size());
    std::size_t index = 0;
    for (const auto& entry : *merges) {
        ++index;

        // Two encodings in the wild for the same data: tokenizers < 0.20 wrote
        // "a b"; newer versions write ["a", "b"]. The array form is also the
        // only one that could represent a token containing a space, so prefer
        // it structurally rather than re-joining and re-splitting.
        if (entry.is_string()) {
            out.merges.push_back(split_merge(entry.get<std::string>(), tokenizer_json, index));
        } else if (entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
                   entry[1].is_string()) {
            out.merges.emplace_back(entry[0].get<std::string>(), entry[1].get<std::string>());
        } else {
            throw VocabLoadError{std::format("{}: model.merges[{}] is neither \"a b\" nor [a, b]",
                                             tokenizer_json.string(), index - 1)};
        }
    }

    out.added_tokens = parse_added_tokens(root, tokenizer_json);

    const auto pre = root.find("pre_tokenizer");
    if (pre != root.end()) {
        out.pretokenizer_pattern = find_split_regex(*pre);
    }

    return out;
}

} // namespace toby::tokenize::detail

namespace toby::tokenize {
struct Vocab::MergeIndex {
    MergeIndex(boost::unordered_flat_map<std::pair<TokenId, TokenId>, MergeTarget> map_in)
        : map(std::move(map_in)) {}

    boost::unordered_flat_map<std::pair<TokenId, TokenId>, MergeTarget> map;
};

Vocab::Vocab() = default;
Vocab::~Vocab() = default;

Vocab::Vocab(Vocab&&) noexcept = default;
Vocab& Vocab::operator=(Vocab&&) noexcept = default;

std::span<const std::byte> Vocab::to_span(const ByteRange& range) const {
    auto it = all_tokens_.begin();
    std::advance(it, range.offset);
    return std::span{it, range.length};
}

std::size_t hash_value(TokenId const& t) noexcept {
    const boost::hash<std::uint32_t> hasher;
    return hasher(t.value);
}

std::ostream& operator<<(std::ostream& out, TokenId id) {
    out << "TokenId{" << id.value << "}";
    return out;
}

Vocab Vocab::load_gpt2(const Gpt2VocabFiles& files) {

    using toby::tokenize::detail::for_each_utf8_code_point;
    using toby::tokenize::detail::Utf8CodePoint;

    auto raw_vocab = detail::load_gpt2_vocab(files.vocab, files.merges);
    auto ret = Vocab{};
    ret.merges_ = std::make_unique<MergeIndex>(merges_to_map(raw_vocab));

    auto vocab_size = raw_vocab.vocab.size() + raw_vocab.added_tokens.size();
    if (vocab_size == 0) {
        throw VocabLoadError{"No entries found in vocab file"};
    }

    auto max_expected_id = vocab_size - 1;

    try {
        constexpr auto printable_to_bytes = printable_to_byte_map();

        for (const auto& [key, value] : raw_vocab.vocab) {

            std::vector<std::byte> vocab_bytes;

            if (std::cmp_greater(value, max_expected_id)) {
                throw VocabLoadError{std::format("{}: Token id {} is greater than expected max {}",
                                                 files.vocab.string(), value, max_expected_id)};
            }

            // Validate each codepoint and append to vocab_bytes. This guards us from adding corrupt
            // data into the the internal state of Vocab (even though in practice a parse error here
            // is probably fatal)

            for_each_utf8_code_point(key, [&printable_to_bytes,
                                           &vocab_bytes](const Utf8CodePoint cp) {
                if (cp.value >= printable_to_bytes.size()) {
                    throw std::range_error{
                        std::format("vocab contains out-of-range code point U+{:04X}", cp.value)};
                }

                auto byte = printable_to_bytes.at(cp.value);
                if (!byte) {
                    throw std::range_error{std::format(
                        "vocab contained codepoint that doesn't map to printable: U+{:04X}",
                        cp.value)};
                }

                vocab_bytes.push_back(*byte);
            });

            auto it = ret.all_tokens_.insert_range(ret.all_tokens_.end(), vocab_bytes);
            auto byte_range = ByteRange{
                .offset = static_cast<std::size_t>(std::distance(ret.all_tokens_.begin(), it)),
                .length = vocab_bytes.size(),
            };
            auto token_id = TokenId{static_cast<uint32_t>(value)};

            if (ret.token_to_bytes_.size() <= token_id.value) {
                ret.token_to_bytes_.resize(token_id.value + 1);
            }

            if (ret.token_to_bytes_.at(token_id.value) != std::nullopt) {
                throw VocabLoadError{std::format("duplicate token id {} in vocab", token_id.value)};
            }
            ret.token_to_bytes_[token_id.value] = byte_range;
            ret.bytes_to_token_id_.emplace_back(byte_range, token_id);
        }

        bool missing_token = false;
        std::string missing_token_ids{"Token ids missing in vocabulary: "};
        for (size_t i = 0; i < ret.token_to_bytes_.size(); i++) {
            if (!ret.token_to_bytes_.at(i).has_value()) {
                std::format_to(std::back_inserter(missing_token_ids), "{}{}",
                               missing_token ? ", " : "", i);
                missing_token = true;
            }
        }

        if (missing_token) {
            throw VocabLoadError{missing_token_ids};
        }

        const auto byte_less = [](const auto lhs, const auto rhs) {
            return std::ranges::lexicographical_compare(lhs, rhs);
        };

        std::ranges::sort(ret.bytes_to_token_id_, byte_less,
                          [&ret](const auto& entry) { return ret.to_span(entry.first); });
    } catch (const std::invalid_argument& e) {
        throw VocabLoadError{std::format("failed loading {}: {}", files.vocab.string(), e.what())};
    } catch (const std::range_error& e) {
        throw VocabLoadError{std::format("failed loading {}: {}", files.vocab.string(), e.what())};
    }

    return ret;
}

[[nodiscard]] std::optional<TokenId>
Vocab::token_for_bytes(std::span<const std::byte> bytes) const noexcept {
    const auto byte_less = [](const auto lhs, const auto rhs) {
        return std::ranges::lexicographical_compare(lhs, rhs);
    };

    auto entry =
        std::ranges::lower_bound(bytes_to_token_id_, bytes, byte_less,
                                 [this](const auto& entry) { return to_span(entry.first); });

    if (entry != bytes_to_token_id_.end() && std::ranges::equal(to_span(entry->first), bytes)) {
        return entry->second;
    }

    return {};
}

[[nodiscard]] std::optional<std::span<const std::byte>> Vocab::lookup_token(TokenId id) const {
    if (id.value >= token_to_bytes_.size()) {
        throw std::out_of_range{"TokenId larger than vocab size"};
    }

    return token_to_bytes_[id.value].transform([this](const ByteRange& b) { return to_span(b); });
}

[[nodiscard]] std::optional<Vocab::MergeTarget>
Vocab::find_merge_target(std::pair<TokenId, TokenId> pair) const noexcept {
    if (merges_ == nullptr) {
        return {};
    }

    auto it = merges_->map.find(pair);
    if (it == merges_->map.end()) {
        return {};
    }

    return it->second;
}
} // namespace toby::tokenize
