#ifndef GPT2_KERNELS_H
#define GPT2_KERNELS_H

#include "toby/safetensors/arena.hpp"
#include "toby/safetensors/tensor.hpp"

namespace toby::gpt2 {
/**
 * TOOD: not gpt2 specific, but I don't want to create a utils library yet
 *
 * Check whether two tensors are equal. To be equal they must:
 *  - Have the same shape
 *  - Have the same dtype
 *  - Have every element compare equal in a pair-wise comparison
 *
 * If the device type of the two tensors is different, throw an exception.
 */
bool tensors_equal(const tensors::Tensor& t1, const tensors::Tensor& t2);

/**
 * Run the embedding part of GPT2 and return a new tensor that is the embedded representation of the
 * input tokens. It should be of shape [len(in_tokens), hidden_dimension]. Memory to store it will
 * be taken from output_arena. It will have the same dtype as the `in_wte` tensor.
 *
 * in_tokens: 1D vector of tokens. Must be an integer. For each token, 0 <= position < vocab_size.
 * in_position: 1D vector of positions for the token. Must be a 1D vector the same shape as
 * in_tokens. For each position, 0 <= position < max_seq_len (1024 for GPT2Medium). in_wte: Weights
 * for the embedding layer. Should be 2D tensor of <vocab_size, hidden_dimension>. Eg for GPT2
 * medium [50257, 1024]. in_wpe: Weights for positional embedding layer. In GPT2 the embeddings are
 * only indexed by position; no RoPE. Should be 2D tensor of <max_seq_len, hidden_dimension>. Eg for
 * GPT2 medium [1024, 1024].
 *
 * Other invariant: all tensors plus the arena must live on the same device. If they don't an
 * exception will be thrown. in_wte/in_wpe must have the same dtype. in_tokens/in_position must be
 * an integral dtype.
 */
tensors::Tensor run_embedding(tensors::Arena& output_arena, const tensors::Tensor& in_tokens,
                              const tensors::Tensor& in_position, const tensors::Tensor& in_wte,
                              const tensors::Tensor& in_wpe);
} // namespace toby::gpt2
#endif
