/**
 * @file paged_attention_kernel.cpp
 * @brief CPU Paged Attention 算子（Decode 阶段）
 *
 * 基于 Block Table 的注意力计算，适用于 Decode 阶段（每请求单 Token Query）：
 *   1. Q @ K^T → scores (scaled dot-product)，Key 从 Paged Cache 中按 Block 查找
 *   2. Softmax(scores)
 *   3. scores @ V → Output，Value 同样从 Paged Cache 查找
 *
 * Cache 布局: [num_blocks, block_size, num_kv_heads, head_size]
 */
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU Paged Attention（Decode 阶段）
 *
 * 对每个 (sequence, head) 计算完整的 Scaled Dot-Product Attention：
 *   1. Q · K^T → scores（K 从 Paged Cache 中按 Block Table 查找）
 *   2. Softmax(scores × scale)
 *   3. 加权求和 V → output（V 同样从 Paged Cache 查找）
 *
 * 支持 GQA：多个 Q Head 共享同一个 KV Head。
 *
 * @param query            输入 Query [batch_size, num_heads × head_size]
 * @param output           输出 Tensor，shape 同 query
 * @param k_cache          Key Cache [num_blocks, block_size, num_kv_heads, head_size]
 * @param v_cache          Value Cache，布局同 k_cache
 * @param block_table      Block Table [batch_size, max_blocks_per_seq]，Int32
 * @param context_lens     每个序列的上下文长度 [batch_size]，Int32
 * @param max_context_len  未使用（保留接口兼容）
 * @param num_heads        Q Head 数
 * @param num_kv_heads     KV Head 数（GQA 时 < num_heads）
 * @param head_size        单个 Head 维度
 * @param block_size       每个物理 Block 容纳的 Token 数
 * @param scale            注意力缩放因子（通常 = 1/√head_size）
 * @param stream           未使用
 */
void paged_attention_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& output,
                                const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                const tensor::Tensor& block_table,
                                const tensor::Tensor& context_lens, int32_t max_context_len,
                                int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                int32_t block_size, float scale, [[maybe_unused]] void* stream) {
    UNUSED(max_context_len);

    int32_t batch_size = static_cast<int32_t>(query.get_dim(0));
    int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));
    int32_t heads_per_kv = num_heads / num_kv_heads;

    // Cache strides: [num_blocks, block_size, num_kv_heads,
    // head_size]
    int32_t stride_head = head_size;
    int32_t stride_token = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token;

    const float* q_ptr = query.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    const float* k_cache_ptr = k_cache.ptr<float>();
    const float* v_cache_ptr = v_cache.ptr<float>();
    const int32_t* bt_ptr = block_table.ptr<int32_t>();
    const int32_t* cl_ptr = context_lens.ptr<int32_t>();

    // 临时 buffer 用于 softmax scores
    std::vector<float> scores;

    for (int32_t seq = 0; seq < batch_size; ++seq) {
        int32_t seq_len = cl_ptr[seq];
        if (seq_len == 0) continue;

        scores.resize(seq_len);

        for (int32_t head = 0; head < num_heads; ++head) {
            int32_t kv_head = head / heads_per_kv;

            // Q 指针: [batch_size, num_heads * head_size]
            const float* q_head = q_ptr + seq * num_heads * head_size + head * head_size;
            float* o_head = out_ptr + seq * num_heads * head_size + head * head_size;

            // Step 1: Q @ K^T → scores[t] for t in [0, seq_len)
            for (int32_t t = 0; t < seq_len; ++t) {
                int32_t log_block = t / block_size;
                int32_t blk_offset = t % block_size;
                int32_t phys_block = bt_ptr[seq * max_blocks_per_seq + log_block];

                const float* k_vec = k_cache_ptr + static_cast<int64_t>(phys_block) * stride_block +
                                     static_cast<int64_t>(blk_offset) * stride_token +
                                     static_cast<int64_t>(kv_head) * stride_head;

                float dot = 0.0f;
                for (int32_t d = 0; d < head_size; ++d) {
                    dot += q_head[d] * k_vec[d];
                }
                scores[t] = dot * scale;
            }

            // Step 2: Softmax
            float max_val = -FLT_MAX;
            for (int32_t t = 0; t < seq_len; ++t) {
                max_val = std::max(max_val, scores[t]);
            }
            float sum_exp = 0.0f;
            for (int32_t t = 0; t < seq_len; ++t) {
                scores[t] = std::exp(scores[t] - max_val);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-6f);
            for (int32_t t = 0; t < seq_len; ++t) {
                scores[t] *= inv_sum;
            }

            // Step 3: Weighted sum of V
            std::memset(o_head, 0, head_size * sizeof(float));
            for (int32_t t = 0; t < seq_len; ++t) {
                int32_t log_block = t / block_size;
                int32_t blk_offset = t % block_size;
                int32_t phys_block = bt_ptr[seq * max_blocks_per_seq + log_block];

                const float* v_vec = v_cache_ptr + static_cast<int64_t>(phys_block) * stride_block +
                                     static_cast<int64_t>(blk_offset) * stride_token +
                                     static_cast<int64_t>(kv_head) * stride_head;

                float prob = scores[t];
                for (int32_t d = 0; d < head_size; ++d) {
                    o_head[d] += prob * v_vec[d];
                }
            }
        }
    }
}

REGISTER_KERNEL(paged_attention, kDeviceCPU, paged_attention_kernel_cpu)

}  // namespace kernel
