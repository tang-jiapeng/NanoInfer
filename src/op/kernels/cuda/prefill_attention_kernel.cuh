#ifndef PREFILL_ATTENTION_KERNEL_CUH
#define PREFILL_ATTENTION_KERNEL_CUH

#include "nanoinfer/tensor/tensor.h"

namespace kernel {

/**
 * @brief Chunked Prefill Attention Kernel (vLLM 风格)
 *
 * 支持分块处理 prompt tokens (Chunked Prefill):
 * 1. 将当前 chunk 的 K/V 写入 Paged Cache
 * 2. 从 Paged Cache Gather 全部已缓存 K/V [0, context_len)
 * 3. cuBLAS batched GEMM + chunked causal softmax
 *
 * Score 矩阵: [num_heads, chunk_len, context_len] — O(chunk × context)
 * 避免了 O(seq_len²) 的显存爆炸
 *
 * @param query       [chunk_len, num_heads * head_size]
 * @param key         [chunk_len, num_kv_heads * head_size]
 * @param value       [chunk_len, num_kv_heads * head_size]
 * @param output      [chunk_len, num_heads * head_size]
 * @param k_cache     全局 Key Cache (Paged)
 * @param v_cache     全局 Value Cache (Paged)
 * @param block_table [1, max_blocks_per_seq] (单序列 prefill)
 * @param positions   [chunk_len] 每个 token 的绝对位置
 * @param num_heads   Query 头数
 * @param num_kv_heads KV 头数
 * @param head_size   每个头的维度
 * @param block_size  KV Cache block 大小
 * @param context_len 写入 chunk 后的总上下文长度 (start_pos + chunk_len)
 * @param config      CudaConfig (含 cublas handle 和 stream)
 */
void prefill_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& key,
    const tensor::Tensor& value, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& positions,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, int32_t context_len, const CudaConfig* config);

}  // namespace kernel

#endif  // PREFILL_ATTENTION_KERNEL_CUH
