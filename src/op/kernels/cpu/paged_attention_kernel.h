#ifndef NANO_INFER_PAGED_ATTENTION_KERNEL_CPU_H
#define NANO_INFER_PAGED_ATTENTION_KERNEL_CPU_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void paged_attention_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& output,
                                const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                const tensor::Tensor& block_table,
                                const tensor::Tensor& context_lens, int32_t max_context_len,
                                int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                int32_t block_size, float scale, void* stream);
}  // namespace kernel
#endif
