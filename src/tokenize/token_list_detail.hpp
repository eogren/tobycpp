#pragma once

#include "toby/tokenize/vocab.hpp"

#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
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
    TokenList(std::shared_ptr<const Vocab> vocab, std::span<const std::byte> bytes_in);

    [[nodiscard]] TokenId at(std::size_t n) const;

    /**
    Merge the token at the given position with its neighbor, replacing with new_token.
    Eg the token list is 0 -> 1 -> 2, and we call `merge_with_neighbor(0, 100)`, the new list
    is 100 -> 2. [replaced 0 with 100 and 1 is no longer part of the list].
     */
    void merge_with_neighbor(std::size_t position, TokenId new_token);

    class ConstIterator {
    public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = TokenId;
        using difference_type = std::ptrdiff_t;
        using reference = const TokenId&;
        using pointer = const TokenId*;

        ConstIterator() = default;

        reference operator*() const;
        pointer operator->() const;

        ConstIterator& operator++();
        ConstIterator operator++(int);

        friend bool operator==(const ConstIterator&, const ConstIterator&) = default;

    private:
        friend class TokenList;

        ConstIterator(const TokenList* owner, std::size_t node);

        const TokenList* owner_{};
        std::size_t node_{npos};
    };

    [[nodiscard]] ConstIterator begin() const;
    [[nodiscard]] ConstIterator end() const;

private:
    struct Node {
        Node(TokenId token_id, size_t next) : token(token_id), next(next) {}

        TokenId token;
        size_t next;
    };

    std::vector<Node> nodes_;
    std::shared_ptr<const Vocab> vocab_;

    static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
};
} // namespace toby::tokenize::detail
