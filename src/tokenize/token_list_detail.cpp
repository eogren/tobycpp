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
static_assert(std::forward_iterator<TokenList::Iterator>);
static_assert(std::convertible_to<TokenList::Iterator, TokenList::ConstIterator>);
static_assert(!std::convertible_to<TokenList::ConstIterator, TokenList::Iterator>);
static_assert(std::ranges::forward_range<TokenList>);
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

TokenList::Iterator TokenList::begin() {
    return {this, 0};
}

TokenList::Iterator TokenList::end() {
    return {this, nodes_.size()};
}

TokenList::ConstIterator TokenList::begin() const {
    return {this, 0};
}

TokenList::ConstIterator TokenList::end() const {
    return {this, nodes_.size()};
}

template <bool IsConst>
TokenList::BasicIterator<IsConst>::BasicIterator(Owner* owner, std::size_t node)
    : owner_(owner), node_(node) {}

template <bool IsConst>
TokenList::BasicIterator<IsConst>::BasicIterator(const BasicIterator<false>& other)
    requires IsConst
    : owner_(other.owner_), node_(other.node_) {}

template <bool IsConst>
TokenList::BasicIterator<IsConst>& TokenList::BasicIterator<IsConst>::operator++() {
    assert(owner_ != nullptr);
    assert(node_ < owner_->nodes_.size());

    node_ = owner_->nodes_[node_].next;

    return *this;
}

template <bool IsConst>
TokenList::BasicIterator<IsConst> TokenList::BasicIterator<IsConst>::operator++(int) {
    auto previous = *this;
    ++(*this);
    return previous;
}

template <bool IsConst> const TokenId& TokenList::BasicIterator<IsConst>::operator*() const {
    assert(node_ < owner_->nodes_.size());

    return owner_->nodes_.at(node_).token;
}

template <bool IsConst> const TokenId* TokenList::BasicIterator<IsConst>::operator->() const {
    return std::addressof(operator*());
}

template class TokenList::BasicIterator<false>;
template class TokenList::BasicIterator<true>;
} // namespace toby::tokenize::detail
