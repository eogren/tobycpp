#include "toby/tokenize/pretokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

enum class CharClass : std::uint8_t {
    Quote,
    Space,
    OtherWhitespace,
    Number,
    LittleD,
    LittleE,
    LittleL,
    LittleM,
    LittleR,
    LittleT,
    LittleS,
    LittleV,
    OtherLetter,
    Other,
};

constexpr bool is_whitespace(const CharClass char_class) {
    return char_class == CharClass::Space || char_class == CharClass::OtherWhitespace;
}

constexpr bool is_letter(const CharClass char_class) {
    return char_class == CharClass::OtherLetter || char_class == CharClass::LittleD ||
           char_class == CharClass::LittleE || char_class == CharClass::LittleL ||
           char_class == CharClass::LittleM || char_class == CharClass::LittleR ||
           char_class == CharClass::LittleS || char_class == CharClass::LittleT ||
           char_class == CharClass::LittleV;
}

constexpr CharClass characterize(const uint32_t codepoint) {
    if (codepoint >= 128) {
        throw std::invalid_argument{"character >= 128"};
    }

    if (codepoint == 'd') {
        return CharClass::LittleD;
    }

    if (codepoint == 'e') {
        return CharClass::LittleE;
    }

    if (codepoint == 'l') {
        return CharClass::LittleL;
    }

    if (codepoint == 'm') {
        return CharClass::LittleM;
    }

    if (codepoint == 'r') {
        return CharClass::LittleR;
    }

    if (codepoint == 's') {
        return CharClass::LittleS;
    }

    if (codepoint == 't') {
        return CharClass::LittleT;
    }

    if (codepoint == 'v') {
        return CharClass::LittleV;
    }

    if ((codepoint >= 'A' && codepoint <= 'Z') || (codepoint >= 'a' && codepoint <= 'z')) {
        return CharClass::OtherLetter;
    }

    if (codepoint == ' ') {
        return CharClass::Space;
    }

    if (codepoint == '\t' || codepoint == '\n' || codepoint == '\v' || codepoint == '\f' ||
        codepoint == '\r') {
        return CharClass::OtherWhitespace;
    }

    if (codepoint >= '0' && codepoint <= '9') {
        return CharClass::Number;
    }

    if (codepoint == '\'') {
        return CharClass::Quote;
    }

    return CharClass::Other;
}

static_assert(characterize('5') == CharClass::Number);
static_assert(characterize('C') == CharClass::OtherLetter);
static_assert(characterize('c') == CharClass::OtherLetter);

constexpr std::vector<CharClass> characterize_str(std::string_view view) {
    std::vector<CharClass> ret{};
    ret.reserve(view.length());

    // todo: simd version
    std::ranges::transform(view, std::back_inserter(ret), [](const char c) -> CharClass {
        return characterize(static_cast<uint32_t>(c));
    });

    return ret;
}

/**
    Look for the end of a quotation_block. Return some(end) if one is found; none if not.
*/
constexpr std::optional<std::ptrdiff_t>
end_quotation_block(std::span<const CharClass> text_classes) noexcept {
    if (text_classes.empty()) {
        return {};
    }

    if (text_classes[0] != CharClass::Quote) {
        return {};
    }

    if (text_classes.size() > 1 &&
        (text_classes[1] == CharClass::LittleS || text_classes[1] == CharClass::LittleD ||
         text_classes[1] == CharClass::LittleM || text_classes[1] == CharClass::LittleT)) {
        return 1;
    }

    if (text_classes.size() > 2) {
        if (text_classes[2] == CharClass::LittleE) {
            if (text_classes[1] == CharClass::LittleR || text_classes[1] == CharClass::LittleV) {
                return 2;
            }
        } else if (text_classes[1] == CharClass::LittleL && text_classes[2] == CharClass::LittleL) {
            return 2;
        }
    }

    return {};
}

constexpr std::array test_strings = {
    CharClass::Quote,       CharClass::LittleT, CharClass::Quote,   CharClass::OtherLetter,
    CharClass::OtherLetter, CharClass::Quote,   CharClass::LittleL, CharClass::LittleL,
    CharClass::Quote,       CharClass::LittleD, CharClass::Quote,   CharClass::LittleV,
    CharClass::LittleE};

constexpr std::span<const CharClass> test_span = test_strings;

static_assert(end_quotation_block(test_span) == 1);
static_assert(!end_quotation_block(test_span.subspan(2)));
static_assert(end_quotation_block(test_span.subspan(5)) == 2);
static_assert(end_quotation_block(test_span.subspan(8)) == 1);
static_assert(end_quotation_block(test_span.subspan(10)) == 2);

constexpr CharClass standardize_char_class(CharClass char_class) noexcept {
    if (is_letter(char_class)) {
        char_class = CharClass::OtherLetter;
    } else if (is_whitespace(char_class)) {
        char_class = CharClass::OtherWhitespace;
    } else if (char_class == CharClass::Quote) {
        char_class = CharClass::Other;
    }

    return char_class;
}
} // namespace

namespace toby::tokenize {
std::vector<std::string_view> pretokenize(std::string_view text) {
    std::vector<std::string_view> ret{};

    /*
        '(?:[sdmt]|ll|ve|re)   # English contractions such as 'm and 've
         ?\p{L}+              # Optional space + one or more letters
         ?\p{N}+              # Optional space + one of more numbers
         ?[^\s\p{L}\p{N}]+    # Optional space + one or more punctuation(-ish)
        \s+(?!\S)            # Whitespace not followed by non-whitespace
        \s+"""               # Whitespace
    */
    auto classes = characterize_str(text);

    auto it = classes.begin();

    while (it != classes.end()) {
        // Check quote class first.
        auto quote_offset = end_quotation_block({it, classes.end()});
        if (quote_offset) {
            auto idx_in_string = std::distance(classes.begin(), it);
            const auto* str_it = text.begin();
            std::advance(str_it, idx_in_string);

            ret.emplace_back(str_it, *quote_offset + 1);
            std::advance(it, *quote_offset + 1);

            continue;
        }

        auto class_selector = it;

        if (*it == CharClass::Space) {
            std::advance(class_selector, 1);
            if (class_selector == classes.end()) {
                auto idx_in_string = std::distance(classes.begin(), it);
                const auto* str_it = text.begin();
                std::advance(str_it, idx_in_string);

                ret.emplace_back(str_it, 1);
                break;
            }
        }

        // then check to see if class switches
        auto char_class = standardize_char_class(*class_selector);
        auto end_char =
            std::find_if(class_selector, classes.end(), [char_class](const CharClass x) {
                return standardize_char_class(x) != char_class;
            });

        auto idx_in_string = std::distance(classes.begin(), it);
        const auto* str_it = text.begin();
        std::advance(str_it, idx_in_string);

        if (is_whitespace(char_class) && end_char != classes.end()) {
            if (!is_whitespace(*end_char) && std::distance(it, end_char) != 1) {
                std::advance(end_char, -1);
            }
        }

        ret.emplace_back(str_it, std::distance(it, end_char));
        it = end_char;
    }

    return ret;
}
} // namespace toby::tokenize
