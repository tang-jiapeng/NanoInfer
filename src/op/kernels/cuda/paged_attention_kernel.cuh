#ifndef PAGED_ATTENTION_KERNEL_CUH
#define PAGED_ATTENTION_KERNEL_CUH

#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void paged_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& context_lens,
    int32_t max_context_len, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_size, int32_t block_size, float scale, void* stream);
}

#endif