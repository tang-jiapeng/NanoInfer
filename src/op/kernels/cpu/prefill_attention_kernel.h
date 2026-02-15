#ifndef NANO_INFER_PREFILL_ATTENTION_KERNEL_CPU_H
#define NANO_INFER_PREFILL_ATTENTION_KERNEL_CPU_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
/**
 * @brief CPU Chunked Prefill Attention
 *
 * 与 CUDA 版本相同的语义:
 * 1. 将 chunk K/V 写入 Paged Cache
 * 2. 从 Cache 读取 [0, context_len) 的全部 K/V
 * 3. Q @ K^T → causal softmax → Scores @ V → output
 *
 * @param context_len  本 chunk 结束后的总上下文长度 (start_pos + chunk_len)
 */
void prefill_attention_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& key,
                                  const tensor::Tensor& value, const tensor::Tensor& output,
                                  const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                  const tensor::Tensor& block_table,
                                  const tensor::Tensor& positions, int32_t num_heads,
                                  int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                                  int32_t context_len);
}  // namespace kernel
#endif
