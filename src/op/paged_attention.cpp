/**
 * @file paged_attention.cpp
 * @brief PagedAttention 层实现（RoPE + KV Cache + Attention）
 *
 * 统一的 Attention 层，组合多个子算子完成完整的注意力计算：
 *
 *   Prefill 流程：
 *     RoPE(Q, K) → PrefillAttention（内含 KV Cache 写入 + 加掩码 Attention）
 *
 *   Decode 流程：
 *     RoPE(Q, K) → PagedKVWrite（写入当前 Token）→ PagedAttention（基于 Block Table）
 *
 * 输入：Q/K/V + block_table + context_lens + input_pos（6 个输入）
 * 全局资源：key_cache_ / value_cache_ / sin_cache_ / cos_cache_
 */
#include "nanoinfer/op/paged_attention.h"
#include "kernels/kernel_registry.h"
#include "kernels/kernel_types.h"

namespace op {
PagedAttention::PagedAttention(base::DeviceType device_type, int32_t layer_index, int32_t kv_mul,
                               int32_t kv_dim, int32_t head_num, int32_t head_size,
                               int32_t block_size)
    : Layer(device_type, LayerType::kLayerPagedAttention, "PagedAttention"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      head_num_(head_num),
      head_size_(head_size),
      block_size_(block_size) {
    // Paged Attention 需要 6 个动态输入
    reset_input_size(6);
    reset_output_size(1);
}

/** @brief 输入校验：检查 6 个输入非空 + KV Cache / RoPE Cache 已设置 */
base::Status PagedAttention::check() const {
    // 检查输入是否齐全
    for (int i = 0; i < 6; ++i) {
        if (get_input(i).is_empty()) {
            return base::Status(base::StatusCode::kInternalError,
                                "Input " + std::to_string(i) + " is empty in PagedAttention");
        }
    }
    // 检查全局资源是否设置
    if (key_cache_.is_empty() || value_cache_.is_empty()) {
        return base::Status(base::StatusCode::kInvalidArgument,
                            "KV Cache not set in PagedAttention");
    }
    if (sin_cache_.is_empty() || cos_cache_.is_empty()) {
        return base::Status(base::StatusCode::kInvalidArgument,
                            "RoPE Cache not set in PagedAttention");
    }
    return base::error::Success();
}

void PagedAttention::set_kv_cache(const tensor::Tensor& key_cache,
                                  const tensor::Tensor& value_cache) {
    // 仅仅持有引用/浅拷贝，不分配新内存
    key_cache_ = key_cache;
    value_cache_ = value_cache;
}

void PagedAttention::set_rope_cache(const tensor::Tensor& sin_cache,
                                    const tensor::Tensor& cos_cache) {
    sin_cache_ = sin_cache;
    cos_cache_ = cos_cache;
}

void PagedAttention::set_prefill(bool is_prefill) {
    is_prefill_ = is_prefill;
}

void PagedAttention::set_context_len(int32_t context_len) {
    context_len_ = context_len;
}

/**
 * @brief 前向计算：完整的 Attention 流程
 *
 * 执行流程：
 *   1. RoPE: 对 Q/K 施加旋转位置编码
 *   2a. Prefill: 调用 prefill_attention 算子（含 KV Cache 写入 + Causal Mask）
 *   2b. Decode: PagedKVWrite 写入当前 Token → PagedAttention 基于 Block Table 查询
 */
base::Status PagedAttention::forward() {
    auto status = check();
    if (!status) return status;

    // 1. 获取输入 Tensors
    const auto& query = get_input(0);         // [total_tokens, num_heads * head_dim]
    const auto& key = get_input(1);           // [total_tokens, num_kv_heads * head_dim]
    const auto& value = get_input(2);         // [total_tokens, num_kv_heads * head_dim]
    const auto& block_table = get_input(3);   // [batch, max_blocks]
    const auto& context_lens = get_input(4);  // [batch] (int32)
    const auto& input_pos = get_input(5);     // [total_tokens] (int32)

    auto& output = get_output(0);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }

    // 计算辅助参数
    int32_t q_dim = head_num_ * head_size_;
    int32_t num_kv_heads = kv_dim_ / head_size_;

    // ==========================================================================
    // Step 1: RoPE (Rotary Positional Embeddings)
    // ==========================================================================
    auto rope_kernel =
        kernel::KernelRegistry::instance().get<kernel::RoPEKernelFn>("rope", device_type_);
    if (!rope_kernel) {
        return base::error::InternalError("RoPE kernel not found for device: " +
                                          std::to_string(static_cast<int>(device_type_)));
    }
    rope_kernel(q_dim, kv_dim_, head_size_, query, key, input_pos, sin_cache_, cos_cache_,
                cuda_config_ ? cuda_config_->stream : nullptr);

    // ==========================================================================
    // Prefill vs Decode 分支
    // ==========================================================================
    if (is_prefill_) {
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
        // ---- Decode: Paged Attention (单 token 查询) ----

        // Step 2: KV Cache Write
        auto kv_write_kernel = kernel::KernelRegistry::instance().get<kernel::PagedKVWriteKernelFn>(
            "paged_kv_write", device_type_);
        if (!kv_write_kernel) {
            return base::error::InternalError("PagedKVWrite kernel not found for device: " +
                                              std::to_string(static_cast<int>(device_type_)));
        }
        kv_write_kernel(key, value, key_cache_, value_cache_, block_table, input_pos, num_kv_heads,
                        head_size_, block_size_, cuda_config_ ? cuda_config_->stream : nullptr);

        // Step 3: Paged Attention
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