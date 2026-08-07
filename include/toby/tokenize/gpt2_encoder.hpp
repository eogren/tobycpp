#ifndef TOBY_TOKENIZE_GPT2_ENCODER_HPP
#define TOBY_TOKENIZE_GPT2_ENCODER_HPP

#include "toby/tokenize/detail/token_list.hpp"
#include "toby/tokenize/pretokenizer.hpp"
#include "toby/tokenize/vocab.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <span>
#include <string_view>

namespace toby::tokenize {

class Gpt2Encoder {
public:
    explicit Gpt2Encoder(std::shared_ptr<const Vocab> vocab);

    template <typename OutIter>
    void encode(std::string_view text, OutIter inserter)
        requires std::output_iterator<OutIter, TokenId>
    {
        auto pieces = pretokenize(text);
        for (auto piece : pieces) {
            auto piece_bytes = std::as_bytes(std::span{piece.data(), piece.size()});
            encode_one(piece_bytes, inserter);
        }
    }

    template <typename OutIter>
    void encode_one(std::span<const std::byte> bytes, OutIter inserter)
        requires std::output_iterator<OutIter, TokenId>
    {
        detail::TokenList token_list{*vocab_, bytes};
        do_merge(token_list);
        std::ranges::copy(token_list, inserter);
    }

private:
    void do_merge(detail::TokenList& token_list) const;

    std::shared_ptr<const Vocab> vocab_;
};
} // namespace toby::tokenize
#endif
