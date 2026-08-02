#include "token_list_detail.hpp"

#include "toby/tokenize/vocab.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <format>
#include <iterator>
#include <limits>
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
TokenList::TokenList(const Vocab& vocab, std::span<const std::byte> bytes_in) {
    if (bytes_in.size() >= std::numeric_limits<std::ptrdiff_t>::max()) {
        throw std::runtime_error{"bytes_in has way too many elements, bailing"};
    }
    if (vocab.size() == 0) {
        throw std::invalid_argument{"vocab must not be null or empty"};
    }

    nodes_.reserve(bytes_in.size());
    for (const std::byte& b : bytes_in) {
        auto token = vocab.token_for_bytes(std::span{&b, 1});
        if (!token.has_value()) {
            throw std::invalid_argument{
                std::format("No token found for {:02x}", std::to_integer<unsigned int>(b))};
        }

        nodes_.emplace_back(*token, nodes_.size() + 1);
    }

    check_integrity();
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

template <bool IsConst>
void TokenList::BasicIterator<IsConst>::merge_with_neighbor(TokenId new_token)
    requires(!IsConst)
{
    auto& node = owner_->nodes_.at(node_);
#ifndef NDEBUG
    if (!node.active) {
        throw std::runtime_error{"Iterator pointing at inactive node"};
    }
#endif

    if (node.next == owner_->nodes_.size()) {
        throw std::runtime_error{"Cannot merge at end of sequence"};
    }

    // me.next = two nodes from now
    node.token = new_token;
    node.version += 1;
    auto& next_node = owner_->nodes_.at(node.next);
    node.next = next_node.next;

    // me.new_next.prev = me
    if (node.next != owner_->nodes_.size()) {
        auto& new_next = owner_->nodes_.at(node.next);
        new_next.prev = static_cast<std::ptrdiff_t>(node_);
    }

#ifndef NDEBUG
    next_node.active = false;
#endif

    owner_->check_integrity();
}

template class TokenList::BasicIterator<false>;
template class TokenList::BasicIterator<true>;

void TokenList::check_integrity() const {
#ifndef NDEBUG
    // if not active: next
    // if active:
    // next should be active
    // next->prev should be this if not at end
    // if at beginning prev should be -1
    if (nodes_.empty()) {
        return;
    }

    const auto& first = nodes_.front();
    if (!first.active) {
        throw std::logic_error{"First node should be active"};
    }

    if (first.prev != -1) {
        throw std::logic_error{"Previous elem of first node should be -1"};
    }

    for (size_t cur_index = 0; cur_index < nodes_.size(); cur_index++) {
        const auto& node = nodes_.at(cur_index);
        // If node not active, we have nothing to verify
        if (!node.active) {
            continue;
        }

        // If at the end, we don't need to do this next ptr check
        if (node.next == nodes_.size()) {
            continue;
        }

        const auto& next = nodes_.at(node.next);
        if (!next.active) {
            throw std::logic_error{std ::format(
                "Consistency failure: Node {} points at inactive node {}", cur_index, node.next)};
        }

        const auto& next_prev = nodes_.at(node.next).prev;
        if (next_prev < 0 || std::cmp_not_equal(next_prev, cur_index)) {
            throw std::logic_error{
                std::format("Consistency failure: Node {} has next {}, but next->prev is {}",
                            cur_index, node.next, next_prev)};
        }
    }
#endif
}
} // namespace toby::tokenize::detail
