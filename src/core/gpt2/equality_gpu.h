#ifndef GPT2_EQUALITY_GPU_H
#define GPT2_EQUALITY_GPU_H

#include "toby/safetensors/tensor.hpp"

namespace toby::gpt2::detail {
bool tensors_equal_gpu(const toby::tensors::Tensor& t1, const toby::tensors::Tensor& t2);
}
#endif
