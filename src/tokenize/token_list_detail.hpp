#pragma once

#include "toby/tokenize/vocab.hpp"

#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace toby::tokenize::detail {
/**
    TokenList
*/
class TokenList {
public:
    /**
      Initialize the token list with the given byte string and vocab. It will first convert every
      byte in the input string into its corresponding token and throw an exception if an unknown
      token is found.
     */
    TokenList(const Vocab& vocab, std::span<const std::byte> bytes_in);

    template <bool IsConst> class BasicIterator {
    public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = TokenId;
        using difference_type = std::ptrdiff_t;
        using reference = const TokenId&;
        using pointer = const TokenId*;

        BasicIterator() = default;
        ~BasicIterator() = default;
        BasicIterator(const BasicIterator&) = default;
        BasicIterator(BasicIterator&&) = default;
        BasicIterator& operator=(const BasicIterator&) = default;
        BasicIterator& operator=(BasicIterator&&) = default;

        BasicIterator(const BasicIterator<false>& other)
            requires IsConst;

        reference operator*() const;
        pointer operator->() const;

        BasicIterator& operator++();
        BasicIterator operator++(int);

        friend bool operator==(const BasicIterator&, const BasicIterator&) = default;

        /** Merge the current token with its neighbor, replacing both with new_token. */
        void merge_with_neighbor(TokenId new_token)
            requires(!IsConst);

    private:
        friend class TokenList;
        template <bool> friend class BasicIterator;

        using Owner = std::conditional_t<IsConst, const TokenList, TokenList>;

        BasicIterator(Owner* owner, std::size_t node);

        Owner* owner_{};
        std::size_t node_{npos};
    };

    using Iterator = BasicIterator<false>;
    using ConstIterator = BasicIterator<true>;

    [[nodiscard]] Iterator begin();
    [[nodiscard]] Iterator end();
    [[nodiscard]] ConstIterator begin() const;
    [[nodiscard]] ConstIterator end() const;

private:
    /**
        Check internal integrity of the list and throw
        an exception if it's invalid
    */
    void check_integrity() const;

    struct Node {
        Node(TokenId token_id, size_t next)
            : token(token_id), next(next), prev(static_cast<std::ptrdiff_t>(next) - 2) {}

        TokenId token;
        std::size_t next;
        // signed so prev can be -1 on the head node
        std::ptrdiff_t prev;
        size_t version{};
#ifndef NDEBUG
        bool active = true;
#endif
    };

    std::vector<Node> nodes_;

    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
};
} // namespace toby::tokenize::detail
