/**
 * @file attention.cpp
 * @brief Attention 算子实现：编排 RoPE → KV Cache Write → Attention 的完整流程
 *
 * 本文件是 Attention的调度层，不包含实际计算 Kernel。
 * 根据 is_prefill_ 标志分流到两条路径：
 *
 *   【Prefill 路径】(处理 prompt 阶段，一次性输入多个 token)
 *     1. RoPE：为 Q/K 施加旋转位置编码
 *     2. PrefillAttention Kernel：内部完成 KV Write + cuBLAS GEMM Attention
 *        （详见 prefill_attention_kernel.cu）
 *
 *   【Decode 路径】(自回归生成阶段，每次输入 1 个 token)
 *     1. RoPE：旋转位置编码
 *     2. PagedKVWrite Kernel：将新 K/V 写入 Paged Cache 对应物理块
 *     3. PagedAttention Kernel：扫描整个 Cache 计算 Attention
 *        （详见 paged_attention_kernel.cu）
 *
 * 输入 Tensor 约定（6 个）：
 *   [0] query        : [total_tokens, num_heads * head_size]
 *   [1] key          : [total_tokens, num_kv_heads * head_size]
 *   [2] value        : [total_tokens, num_kv_heads * head_size]
 *   [3] block_table  : [batch_size, max_blocks_per_seq]  (int32, 逻辑→物理块映射)
 *   [4] context_lens : [batch_size]  (int32, 每个序列的上下文长度)
 *   [5] input_pos    : [total_tokens]  (int32, 每个 token 的绝对位置)
 *
 * 外部注入资源（由 Model 在每层 forward 前设置）：
 *   - key_cache_ / value_cache_ : 当前层的 Paged KV Cache
 *   - sin_cache_ / cos_cache_   : 预计算的 RoPE 三角函数表
 */
#include "nanoinfer/op/attention.h"
#include "kernels/kernel_registry.h"
#include "kernels/kernel_types.h"

namespace op {
AttentionLayer::AttentionLayer(base::DeviceType device_type, int32_t layer_index, int32_t kv_mul,
                               int32_t kv_dim, int32_t head_num, int32_t head_size,
                               int32_t block_size)
    : Layer(device_type, LayerType::kLayerAttention, "Attention"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      head_num_(head_num),
      head_size_(head_size),
      block_size_(block_size) {
    reset_input_size(6);
    reset_output_size(1);
}

base::Status AttentionLayer::check() const {
    for (int i = 0; i < 6; ++i) {
        if (get_input(i).is_empty()) {
            return base::Status(base::StatusCode::kInternalError,
                                "Input " + std::to_string(i) + " is empty in AttentionLayer");
        }
    }
    if (key_cache_.is_empty() || value_cache_.is_empty()) {
        return base::Status(base::StatusCode::kInvalidArgument,
                            "KV Cache not set in AttentionLayer");
    }
    if (sin_cache_.is_empty() || cos_cache_.is_empty()) {
        return base::Status(base::StatusCode::kInvalidArgument,
                            "RoPE Cache not set in AttentionLayer");
    }
    return base::error::Success();
}

void AttentionLayer::set_kv_cache(const tensor::Tensor& key_cache,
                                  const tensor::Tensor& value_cache) {
    key_cache_ = key_cache;
    value_cache_ = value_cache;
}

void AttentionLayer::set_rope_cache(const tensor::Tensor& sin_cache,
                                    const tensor::Tensor& cos_cache) {
    sin_cache_ = sin_cache;
    cos_cache_ = cos_cache;
}

void AttentionLayer::set_prefill(bool is_prefill) {
    is_prefill_ = is_prefill;
}

void AttentionLayer::set_context_len(int32_t context_len) {
    context_len_ = context_len;
}

/**
 * @brief Attention Forward 核心流程
 *
 * 流程概览：
 *   1. RoPE：对 Q、K 施加旋转位置编码（使模型感知 Token 位置）
 *   2. 根据 is_prefill_ 分支：
 *      · Prefill  → 调用 PrefillAttention Kernel（内含 KV Write + cuBLAS GEMM）
 *      · Decode   → 先 PagedKVWrite（写入新 K/V），再 PagedAttention（全序列 Attention）
 *
 * Prefill vs Decode 的关键区别：
 *   - Prefill 一次处理 chunk_len 个 Query token，用 cuBLAS BatchedGEMM 做矩阵乘
 *   - Decode 每次只有 1 个 Query token，用自定义 Kernel 逐 Block 扫描 KV Cache
 */
base::Status AttentionLayer::forward() {
    auto status = check();
    if (!status) return status;

    // ---- 取出 6 个输入 Tensor ----
    const auto& query = get_input(0);         // [total_tokens, num_heads * head_size]
    const auto& key = get_input(1);           // [total_tokens, num_kv_heads * head_size]
    const auto& value = get_input(2);         // [total_tokens, num_kv_heads * head_size]
    const auto& block_table = get_input(3);   // [batch_size, max_blocks_per_seq]
    const auto& context_lens = get_input(4);  // [batch_size] (int32)
    const auto& input_pos = get_input(5);     // [total_tokens] (int32)

    auto& output = get_output(0);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }

    int32_t q_dim = head_num_ * head_size_;
    int32_t num_kv_heads = kv_dim_ / head_size_;

    // ==================================================================
    // Step 1: RoPE (Rotary Positional Embeddings)
    // ------------------------------------------------------------------
    // 对 Q 和 K 同时施加旋转编码，使点积 Q·K 自然包含相对位置信息。
    // 旋转公式：对每对 (q[2i], q[2i+1]) 做二维旋转 θ = pos * freq_i
    //   q'[2i]   = q[2i]·cos(θ) - q[2i+1]·sin(θ)
    //   q'[2i+1] = q[2i]·sin(θ) + q[2i+1]·cos(θ)
    // sin_cache_ / cos_cache_ 就是预计算好的 sin(θ) / cos(θ) 表。
    // ==================================================================
    auto rope_kernel =
        kernel::KernelRegistry::instance().get<kernel::RoPEKernelFn>("rope", device_type_);
    if (!rope_kernel) {
        return base::error::InternalError("RoPE kernel not found for device: " +
                                          std::to_string(static_cast<int>(device_type_)));
    }
    rope_kernel(q_dim, kv_dim_, head_size_, query, key, input_pos, sin_cache_, cos_cache_,
                cuda_config_ ? cuda_config_->stream : nullptr);

    // ==================================================================
    //  Step 2: 分支 — Prefill vs Decode
    // ==================================================================
    if (is_prefill_) {
        // ---- Prefill: 使用 cuBLAS BatchedGEMM 做高效矩阵乘法 ----
        // 完整流程在 prefill_attention_kernel.cu 中：
        //   a. 将当前 chunk 的 K/V 写入 Paged Cache
        //   b. 从 Cache Gather 出 [0, context_len) 的全部 K/V
        //   c. Scores = Q @ K^T   (cuBLAS BatchedGEMM)
        //   d. Chunked Causal Softmax（带 start_pos 偏移的因果 Mask）
        //   e. Output = Scores @ V (cuBLAS BatchedGEMM)
        auto prefill_attention_kernel =
            kernel::KernelRegistry::instance().get<kernel::PrefillAttentionKernelFn>(
                "prefill_attention", device_type_);
        if (!prefill_attention_kernel) {
            return base::error::InternalError("PrefillAttention kernel not found for device: " +
                                              std::to_string(static_cast<int>(device_type_)));
        }
        prefill_attention_kernel(query, key, value, output, key_cache_, value_cache_, block_table,
                                 input_pos, head_num_, num_kv_heads, head_size_, block_size_,
                                 context_len_, cuda_config_ ? cuda_config_.get() : nullptr);
    } else {
        // ---- Decode: PagedAttention (每次只有 1 个新 Query token) ----

        // Step 2a: 将新生成的 K/V 写入 Paged Cache 对应的物理块
        //   根据 input_pos 计算出逻辑 Block → 查 block_table 得物理 Block → 写入
        auto kv_write_kernel = kernel::KernelRegistry::instance().get<kernel::PagedKVWriteKernelFn>(
            "paged_kv_write", device_type_);
        if (!kv_write_kernel) {
            return base::error::InternalError("PagedKVWrite kernel not found for device: " +
                                              std::to_string(static_cast<int>(device_type_)));
        }
        kv_write_kernel(key, value, key_cache_, value_cache_, block_table, input_pos, num_kv_heads,
                        head_size_, block_size_, cuda_config_ ? cuda_config_->stream : nullptr);

        // Step 2b: 计算 Attention
        //   遍历当前序列的整个 KV Cache：
        //   score[t] = Q · K[t] / √head_size   (t = 0 ... context_len-1)
        //   prob = Softmax(score)
        //   output = Σ prob[t] * V[t]
        auto paged_attention_kernel =
            kernel::KernelRegistry::instance().get<kernel::PagedAttentionKernelFn>(
                "paged_attention", device_type_);
        if (!paged_attention_kernel) {
            return base::error::InternalError("PagedAttention kernel not found for device: " +
                                              std::to_string(static_cast<int>(device_type_)));
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(head_size_));
        int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));
        int32_t max_context_len_estimate = max_blocks_per_seq * block_size_;

        paged_attention_kernel(query, output, key_cache_, value_cache_, block_table, context_lens,
                               max_context_len_estimate, head_num_, num_kv_heads, head_size_,
                               block_size_, scale, cuda_config_ ? cuda_config_->stream : nullptr);
    }

    return base::error::Success();
}

}  // namespace op