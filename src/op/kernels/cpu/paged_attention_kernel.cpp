#include "paged_attention_kernel.h"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace kernel {

void paged_attention_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& output,
                                const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                const tensor::Tensor& block_table,
                                const tensor::Tensor& context_lens, int32_t max_context_len,
                                int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                int32_t block_size, float scale, void* stream) {
    UNUSED(stream);
    UNUSED(max_context_len);

    int32_t batch_size = static_cast<int32_t>(query.get_dim(0));
    int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));
    int32_t heads_per_kv = num_heads / num_kv_heads;

    // Cache strides: [num_blocks, block_size, num_kv_heads, head_size]
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

}  // namespace kernel
