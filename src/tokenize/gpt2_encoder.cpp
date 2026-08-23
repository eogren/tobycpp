#include "toby/tokenize/gpt2_encoder.hpp"

#include "toby/tokenize/detail/token_list.hpp"
#include "toby/tokenize/vocab.hpp"

#include <cassert>
#include <compare>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

using toby::tokenize::TokenId;

namespace {
struct MergeCandidate {
    toby::tokenize::detail::TokenList::Iterator position_first;
    std::size_t version;
    std::size_t next_version;
    std::size_t rank;
    TokenId new_token;
#ifndef NDEBUG
    TokenId expected_next;
#endif

    std::strong_ordering operator<=>(const MergeCandidate& other) const {
        // rank first
        if (rank < other.rank) {
            return std::strong_ordering::less;
        }
        if (rank > other.rank) {
            return std::strong_ordering::greater;
        }

        // then version
        if (version < other.version) {
            return std::strong_ordering::less;
        }
        if (version > other.version) {
            return std::strong_ordering::greater;
        }

        // then version
        if (next_version < other.next_version) {
            return std::strong_ordering::less;
        }
        if (next_version > other.next_version) {
            return std::strong_ordering::greater;
        }

        // the new token
        if (new_token < other.new_token) {
            return std::strong_ordering::less;
        }
        if (new_token > other.new_token) {
            return std::strong_ordering::greater;
        }

        return position_first <=> other.position_first;
    }

    bool operator==(const MergeCandidate& other) const {
        return rank == other.rank && version == other.version &&
               next_version == other.next_version && new_token == other.new_token &&
               position_first == other.position_first;
    }
};

template <typename T>
bool maybe_emplace_merge(const toby::tokenize::Vocab& vocab,
                         const toby::tokenize::detail::TokenList::Iterator& it, T& container) {
    auto next = std::next(it);

    if (const auto target = vocab.find_merge_target(std::make_pair(*it, *next))) {
        container.emplace(MergeCandidate{
            .position_first = it,
            .version = it.version(),
            .next_version = next.version(),
            .rank = target->rank,
            .new_token = target->new_token,
#ifndef NDEBUG
            .expected_next = *next,
#endif

        });
        return true;
    }

    return false;
}
} // namespace

namespace toby::tokenize {
Gpt2Encoder::Gpt2Encoder(std::shared_ptr<const Vocab> vocab) : vocab_(std::move(vocab)) {}

void Gpt2Encoder::do_merge(detail::TokenList& token_list) const {
    auto it = token_list.begin();
    auto end = token_list.end();
    auto candidates =
        std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<>>{};

    while (true) {
        // Last element in the list has no suitable merges so bail out
        if (std::distance(it, end) <= 1) {
            break;
        }

        maybe_emplace_merge(*vocab_, it, candidates);
        std::advance(it, 1);
    }

    while (!candidates.empty()) {
        auto next = candidates.top();
        candidates.pop();

        if (!next.position_first.active()) {
            continue;
        }

        auto next_token = std::next(next.position_first);

        // yank and check out version - do merge if still avail
        if (next.position_first.version() == next.version &&
            next_token.version() == next.next_version) {
            next.position_first.merge_with_neighbor(next.new_token);

            if (next.position_first != token_list.begin()) {
                // (prev, new this)
                auto prev_it = std::prev(next.position_first);
                maybe_emplace_merge(*vocab_, prev_it, candidates);
            }

            // (new this, new next)
            auto new_next = std::next(next.position_first);
            if (new_next != token_list.end()) {
                maybe_emplace_merge(*vocab_, next.position_first, candidates);
            }
        }
    }
}
} // namespace toby::tokenize
