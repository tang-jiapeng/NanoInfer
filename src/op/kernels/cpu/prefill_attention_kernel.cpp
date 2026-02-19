#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>
#include "../kernel_registry.h"

namespace kernel {

void prefill_attention_kernel_cpu(const tensor::Tensor& query, const tensor::Tensor& key,
                                  const tensor::Tensor& value, const tensor::Tensor& output,
                                  const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                  const tensor::Tensor& block_table,
                                  const tensor::Tensor& positions, int32_t num_heads,
                                  int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                                  int32_t context_len, [[maybe_unused]] void* stream) {
    int32_t chunk_len = static_cast<int32_t>(query.get_dim(0));
    int32_t kv_dim = num_kv_heads * head_size;
    int32_t start_pos = context_len - chunk_len;
    int32_t heads_per_kv = num_heads / num_kv_heads;

    // Cache strides: [num_blocks, block_size, num_kv_heads, head_size]
    int32_t stride_head = head_size;
    int32_t stride_token = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token;

    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    float* k_cache_ptr = const_cast<float*>(k_cache.ptr<float>());
    float* v_cache_ptr = const_cast<float*>(v_cache.ptr<float>());
    const int32_t* bt = block_table.ptr<int32_t>();
    const int32_t* pos_ptr = positions.ptr<int32_t>();

    // ==== Step 0: Write chunk K/V to Paged Cache ====
    for (int32_t t = 0; t < chunk_len; ++t) {
        int32_t pos = pos_ptr[t];
        int32_t log_block = pos / block_size;
        int32_t blk_offset = pos % block_size;
        int32_t phys_block = bt[log_block];

        for (int32_t h = 0; h < num_kv_heads; ++h) {
            int64_t src_off =
                static_cast<int64_t>(t) * kv_dim + static_cast<int64_t>(h) * head_size;
            int64_t cache_off = static_cast<int64_t>(phys_block) * stride_block +
                                static_cast<int64_t>(blk_offset) * stride_token +
                                static_cast<int64_t>(h) * stride_head;
            std::memcpy(k_cache_ptr + cache_off, k_ptr + src_off, head_size * sizeof(float));
            std::memcpy(v_cache_ptr + cache_off, v_ptr + src_off, head_size * sizeof(float));
        }
    }

    // ==== Step 1-3: 对每个 query token 做 attention over [0, context_len) ====
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_size));
    std::vector<float> scores(context_len);

    for (int32_t qi = 0; qi < chunk_len; ++qi) {
        // 绝对位置 = start_pos + qi → causal mask 允许看到 [0, start_pos + qi]
        int32_t valid_len = start_pos + qi + 1;

        for (int32_t head = 0; head < num_heads; ++head) {
            int32_t kv_head = head / heads_per_kv;

            // Q 向量: query[qi, head * head_size .. (head+1) * head_size]
            const float* q_vec = q_ptr + static_cast<int64_t>(qi) * num_heads * head_size +
                                 static_cast<int64_t>(head) * head_size;

            // Step 1: Q @ K^T for all positions [0, valid_len)
            for (int32_t t = 0; t < valid_len; ++t) {
                int32_t log_block = t / block_size;
                int32_t blk_offset = t % block_size;
                int32_t phys_block = bt[log_block];

                const float* k_vec = k_cache_ptr + static_cast<int64_t>(phys_block) * stride_block +
                                     static_cast<int64_t>(blk_offset) * stride_token +
                                     static_cast<int64_t>(kv_head) * stride_head;

                float dot = 0.0f;
                for (int32_t d = 0; d < head_size; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[t] = dot * inv_sqrt_d;
            }

            // Step 2: Softmax over [0, valid_len)
            float max_val = -FLT_MAX;
            for (int32_t t = 0; t < valid_len; ++t) {
                max_val = std::max(max_val, scores[t]);
            }
            float sum_exp = 0.0f;
            for (int32_t t = 0; t < valid_len; ++t) {
                scores[t] = std::exp(scores[t] - max_val);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-6f);
            for (int32_t t = 0; t < valid_len; ++t) {
                scores[t] *= inv_sum;
            }

            // Step 3: Weighted sum of V → output[qi, head * head_size ..]
            float* o_vec = out_ptr + static_cast<int64_t>(qi) * num_heads * head_size +
                           static_cast<int64_t>(head) * head_size;
            std::memset(o_vec, 0, head_size * sizeof(float));

            for (int32_t t = 0; t < valid_len; ++t) {
                int32_t log_block = t / block_size;
                int32_t blk_offset = t % block_size;
                int32_t phys_block = bt[log_block];

                const float* v_vec = v_cache_ptr + static_cast<int64_t>(phys_block) * stride_block +
                                     static_cast<int64_t>(blk_offset) * stride_token +
                                     static_cast<int64_t>(kv_head) * stride_head;

                float prob = scores[t];
                for (int32_t d = 0; d < head_size; ++d) {
                    o_vec[d] += prob * v_vec[d];
                }
            }
        }
    }
}

REGISTER_KERNEL(prefill_attention, kDeviceCPU, prefill_attention_kernel_cpu)

}  // namespace kernel
