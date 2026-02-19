/**
 * @file rope_kernel.cpp
 * @brief CPU 旋转位置编码（RoPE）算子
 *
 * 对 Query 和 Key 进行原地 (In-place) 旋转编码：
 *   x' = x * cos(θ) - y * sin(θ)
 *   y' = x * sin(θ) + y * cos(θ)
 *
 * 每个 Token 根据其位置索引从预计算的 Sin/Cos Cache 中查表。
 * 支持 GQA/MQA：K 只旋转前 kv_dim 元素（而 Q 旋转全部 dim）。
 */
#include <armadillo>
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU 旋转位置编码（RoPE）
 *
 * 对 Q 和 K 施加 In-place 旋转编码，每对相邻元素 (x, y) 执行：
 *   x' = x × cos(θ) - y × sin(θ)
 *   y' = x × sin(θ) + y × cos(θ)
 * Sin/Cos 值从预计算 Cache 中按 (pos, head_dim) 索引查表。
 *
 * GQA 支持：Q 旋转全部 dim 维度，K 只旋转前 kv_dim 维度。
 *
 * @param dim         Q 的总维度（= num_heads × head_size）
 * @param kv_dim      K 的总维度（= num_kv_heads × head_size，GQA 时 < dim）
 * @param head_size   单个 Head 维度（用于频率取模）
 * @param input_q     Query Tensor [total_tokens, dim]，in-place 修改
 * @param input_k     Key Tensor [total_tokens, kv_dim]，in-place 修改
 * @param input_pos   位置索引 Tensor [total_tokens]，Int32
 * @param sin_cache   预计算 Sin Cache [max_seq_len, head_size]
 * @param cos_cache   预计算 Cos Cache [max_seq_len, head_size]
 * @param stream      未使用
 */
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     [[maybe_unused]] void* stream) {
    CHECK(!input_q.is_empty());
    CHECK(!input_pos.is_empty());

    // 1. 获取 Batch 信息
    // input_q 形状: [total_tokens, dim]
    int32_t total_tokens = static_cast<int32_t>(input_pos.size());

    // 2. 获取数据指针 (非 const 指针用于写入修改)
    float* q_ptr = const_cast<float*>(input_q.ptr<float>());
    float* k_ptr = const_cast<float*>(input_k.ptr<float>());
    const int32_t* pos_ptr = input_pos.ptr<int32_t>();

    const float* sin_ptr = sin_cache.ptr<float>();
    const float* cos_ptr = cos_cache.ptr<float>();

    // 3. 逐 Token 处理 (Batched 循环)
    for (int i = 0; i < total_tokens; ++i) {
        // 当前 Token 的位置索引 (例如: Token 0 pos=10, Token 1 pos=11)
        int32_t pos = pos_ptr[i];

        // 定位当前 Token 在 Q 和 K 中的起始内存地址
        float* q_row = q_ptr + i * dim;
        float* k_row = k_ptr + i * kv_dim;

        // 4. 遍历 Hidden Dim 进行旋转 (步长为 2)
        for (int j = 0; j < dim; j += 2) {
            int32_t head_dim = j % head_size;

            // 从 Cache 中查表 (根据 pos 和 head_dim)
            // Cache 布局通常是 [max_seq_len, head_size]
            int32_t cache_idx = pos * head_size + head_dim;
            float sin_val = sin_ptr[cache_idx];
            float cos_val = cos_ptr[cache_idx];

            // --- Rotate Query (Q) ---
            float q0 = q_row[j];
            float q1 = q_row[j + 1];
            // 旋转公式:
            // x' = x * cos - y * sin
            // y' = x * sin + y * cos
            q_row[j] = q0 * cos_val - q1 * sin_val;
            q_row[j + 1] = q0 * sin_val + q1 * cos_val;

            // --- Rotate Key (K) ---
            // 只有当 j 在 kv_dim 范围内时才旋转 K (处理 GQA/MQA 情况)
            if (j < kv_dim) {
                float k0 = k_row[j];
                float k1 = k_row[j + 1];
                k_row[j] = k0 * cos_val - k1 * sin_val;
                k_row[j + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

REGISTER_KERNEL(rope, kDeviceCPU, rope_kernel_cpu)

}  // namespace kernel