#ifndef PREFILL_ATTENTION_KERNEL_CUH
#define PREFILL_ATTENTION_KERNEL_CUH

#include "nanoinfer/tensor/tensor.h"

namespace kernel {

/**
 * @brief Prefill Causal Attention Kernel (cuBLAS + Custom Softmax)
 *
 * 用于 Prefill 阶段，对所有 prompt tokens 做一次性因果注意力。
 * 同时将 K/V 写入 Paged KV Cache。
 *
 * @param query       [total_tokens, num_heads * head_size]
 * @param key         [total_tokens, num_kv_heads * head_size]
 * @param value       [total_tokens, num_kv_heads * head_size]
 * @param output      [total_tokens, num_heads * head_size]
 * @param k_cache     全局 Key Cache (Paged)
 * @param v_cache     全局 Value Cache (Paged)
 * @param block_table [batch_size, max_blocks_per_seq]
 * @param positions   [total_tokens] 每个 token 的位置
 * @param num_heads   Query 头数
 * @param num_kv_heads KV 头数
 * @param head_size   每个头的维度
 * @param block_size  KV Cache block 大小
 * @param config      CudaConfig (含 cublas handle 和 stream)
 */
void prefill_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& key,
    const tensor::Tensor& value, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& positions,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, const CudaConfig* config);

}  // namespace kernel

#endif  // PREFILL_ATTENTION_KERNEL_CUH
