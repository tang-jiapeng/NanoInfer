/**
 * @file attention.h
 * @brief Attention 层 (Chunked Prefill + Paged Decode)
 */
#ifndef NANO_INFER_ATTENTION_H
#define NANO_INFER_ATTENTION_H

#include "layer.h"
#include "nanoinfer/base/base.h"

namespace op {

/**
 * @brief Attention 层
 *
 * Prefill 使用 Chunked Prefill Attention，Decode 使用 Paged Attention。
 * 不自行维护 KV Cache，而是操作全局 KV Cache Block。
 *
 * 输入约定 (set_input idx):
 * - 0: Query  [batch_size, num_heads, head_dim]
 * - 1: Key    [batch_size, num_kv_heads, head_dim]
 * - 2: Value  [batch_size, num_kv_heads, head_dim]
 * - 3: Block Table [batch_size, max_blocks_per_seq]
 * - 4: Context Lens [batch_size]
 * - 5: Input Pos    [batch_size]
 */
class AttentionLayer : public Layer {
   public:
    explicit AttentionLayer(base::DeviceType device_type, int32_t layer_index, int32_t kv_mul,
                            int32_t kv_dim, int32_t head_num, int32_t head_size,
                            int32_t block_size);

    base::Status check() const override;

    base::Status forward() override;

    using Layer::forward;

    void set_kv_cache(const tensor::Tensor& key_cache, const tensor::Tensor& value_cache) override;

    void set_rope_cache(const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache) override;

    void set_prefill(bool is_prefill) override;

    void set_context_len(int32_t context_len) override;

   private:
    bool is_prefill_ = false;
    int32_t context_len_ = 0;
    int32_t layer_index_ = 0;
    int32_t kv_mul_ = 0;
    int32_t kv_dim_ = 0;
    int32_t head_num_ = 0;
    int32_t head_size_ = 0;
    int32_t block_size_ = 0;

    tensor::Tensor key_cache_;
    tensor::Tensor value_cache_;
    tensor::Tensor sin_cache_;
    tensor::Tensor cos_cache_;
};
}  // namespace op

#endif  // NANO_INFER_ATTENTION_H