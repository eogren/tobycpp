#include "token_list_detail.hpp"

#include "toby/tokenize/vocab.hpp"

#include <cassert>
#include <cstddef>
#include <format>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>

using toby::tokenize::detail::TokenList;

static_assert(std::forward_iterator<TokenList::ConstIterator>);
static_assert(std::ranges::forward_range<const TokenList>);

using toby::tokenize::Vocab;

namespace toby::tokenize::detail {
TokenList::TokenList(std::shared_ptr<const Vocab> vocab, std::span<const std::byte> bytes_in)
    : vocab_(std::move(vocab)) {
    nodes_.reserve(bytes_in.size());
    for (const std::byte& b : bytes_in) {
        auto token = vocab_->token_for_bytes(std::span{&b, 1});
        if (!token.has_value()) {
            throw std::invalid_argument{
                std::format("No token found for {:02x}", std::to_integer<unsigned int>(b))};
        }

        nodes_.emplace_back(*token, nodes_.size() + 1);
    }
}

TokenList::ConstIterator TokenList::begin() const {
    return {this, 0};
}

TokenList::ConstIterator TokenList::end() const {
    return {this, nodes_.size()};
}

TokenList::ConstIterator::ConstIterator(const TokenList* owner, std::size_t node)
    : owner_(owner), node_(node) {}

TokenList::ConstIterator& TokenList::ConstIterator::operator++() {
    assert(owner_ != nullptr);
    assert(node_ < owner_->nodes_.size());

    node_ = owner_->nodes_[node_].next;

    return *this;
}

const TokenId& TokenList::ConstIterator::operator*() const {
    assert(node_ < owner_->nodes_.size());

    return owner_->nodes_.at(node_).token;
}
} // namespace toby::tokenize::detail
