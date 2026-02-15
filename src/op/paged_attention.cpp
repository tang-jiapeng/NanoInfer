#include "nanoinfer/op/paged_attention.h"
#include "kernels/cuda/prefill_attention_kernel.cuh"
#include "kernels/kernels_interface.h"

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

base::Status PagedAttention::init() {
    if (device_type_ != base::DeviceType::kDeviceCUDA) {
        return base::Status(base::StatusCode::kFunctionUnImplement,
                            "PagedAttention only supports CUDA currently.");
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

    // 获取 CUDA 流
    void* cuda_stream = cuda_config_ ? cuda_config_->stream : nullptr;

    // 计算辅助参数
    int32_t q_dim = head_num_ * head_size_;
    int32_t num_kv_heads = kv_dim_ / head_size_;

    // ==========================================================================
    // Step 1: RoPE (Rotary Positional Embeddings)
    // ==========================================================================
    kernel::RoPEKernel rope_kernel = kernel::get_rope_kernel(device_type_);
    if (!rope_kernel) {
        return base::Status(base::StatusCode::kFunctionUnImplement, "RoPE kernel missing");
    }

    rope_kernel(q_dim, kv_dim_, head_size_, query, key, input_pos, sin_cache_, cos_cache_,
                cuda_stream);

    // ==========================================================================
    // Prefill vs Decode 分支
    // ==========================================================================
    if (is_prefill_) {
        // ---- Chunked Prefill: Write to cache + Gather + cuBLAS ----
        // Score 矩阵: [num_heads, chunk_len, context_len] — 不再是 O(seq_len²)
        kernel::prefill_attention_kernel(query, key, value, output, key_cache_, value_cache_,
                                         block_table, input_pos, head_num_, num_kv_heads,
                                         head_size_, block_size_, context_len_,
                                         cuda_config_ ? cuda_config_.get() : nullptr);
    } else {
        // ---- Decode: Paged Attention (单 token 查询) ----

        // Step 2: KV Cache Write
        kernel::PagedKVWriteKernel kv_write_kernel =
            kernel::get_paged_kv_write_kernel(device_type_);
        if (!kv_write_kernel) {
            return base::Status(base::StatusCode::kFunctionUnImplement,
                                "PagedKVWrite kernel missing");
        }
        kv_write_kernel(key, value, key_cache_, value_cache_, block_table, input_pos, num_kv_heads,
                        head_size_, block_size_, cuda_stream);

        // Step 3: Paged Attention
        kernel::PagedAttentionKernel pa_kernel = kernel::get_paged_attention_kernel(device_type_);
        if (!pa_kernel) {
            return base::Status(base::StatusCode::kFunctionUnImplement,
                                "PagedAttention kernel missing");
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(head_size_));
        int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));
        int32_t max_context_len_estimate = max_blocks_per_seq * block_size_;

        pa_kernel(query, output, key_cache_, value_cache_, block_table, context_lens,
                  max_context_len_estimate, head_num_, num_kv_heads, head_size_, block_size_, scale,
                  cuda_stream);
    }

    return base::error::Success();
}

}  // namespace op