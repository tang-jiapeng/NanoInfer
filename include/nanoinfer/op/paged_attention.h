#ifndef NANO_INFER_PAGED_ATTENTION_H
#define NANO_INFER_PAGED_ATTENTION_H

#include "layer.h"
#include "nanoinfer/base/base.h"

namespace op {

/**
 * @brief Paged Attention Layer (基于 vLLM 理念设计)
 *
 * 这是支持 Continuous Batching 的核心 Attention 层。
 * 它不维护自己的 KV Cache 内存，而是操作全局的 KV Cache Block。
 *
 * 输入 Tensor 约定 (set_input):
 * 0: Query [batch_size, num_heads, head_dim]
 * 1: Key   [batch_size, num_kv_heads, head_dim]
 * 2: Value [batch_size, num_kv_heads, head_dim]
 * 3: Block Table [batch_size, max_blocks_per_seq] - 逻辑块到物理块的映射
 * 4: Context Lens [batch_size] - 每个请求的实际上下文长度
 * 5: Input Pos [batch_size] - 每个 Token 的当前生成位置 (用于 RoPE 和 Cache 写入)
 */

class PagedAttention : public Layer {
   public:
    explicit PagedAttention(base::DeviceType device_type, int32_t layer_index, int32_t kv_mul,
                            int32_t kv_dim, int32_t head_num, int32_t head_size,
                            int32_t block_size);

    base::Status check() const override;

    /**
     * @brief 执行 Paged Attention 前向传播
     *
     * 流程：
     * 1. RoPE: 对输入的 Q 和 K 进行旋转位置编码 (使用 input_pos)。
     * 2. KV Cache Write: 将旋转后的 K 和原始 V 写入到全局 KV Cache 的非连续物理块中 (使用
     * block_table)。
     * 3. Paged Attention: 使用 Paged Attention Kernel 计算注意力分数并聚合 Value。
     */
    base::Status forward() override;

    /**
     * @brief 设置全局 KV Cache (由 CacheManager 分配)
     * @param key_cache [num_blocks, block_size, num_kv_heads, head_dim]
     * @param value_cache [num_blocks, block_size, num_kv_heads, head_dim]
     */
    void set_kv_cache(const tensor::Tensor& key_cache, const tensor::Tensor& value_cache);

    /**
     * @brief 设置预计算的 RoPE 表
     */
    void set_rope_cache(const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache);

    /**
     * @brief 设置是否为 Prefill 模式
     * Prefill 时使用 Chunked Prefill (cuBLAS + Gather from cache)
     * Decode 时使用 PagedAttention kernel 做单 token 注意力
     */
    void set_prefill(bool is_prefill);

    /**
     * @brief 设置当前 chunk 完成后的总上下文长度 (host side)
     * 仅 Prefill 模式使用, = start_pos + chunk_len
     */
    void set_context_len(int32_t context_len);

   private:
    bool is_prefill_ = false;
    int32_t context_len_ = 0;
    int32_t layer_index_ = 0;
    int32_t kv_mul_ = 0;     // GQA: query_heads / kv_heads
    int32_t kv_dim_ = 0;     // num_kv_heads * head_size
    int32_t head_num_ = 0;   // num_heads
    int32_t head_size_ = 0;  // head_dim
    int32_t block_size_ = 0;

    tensor::Tensor key_cache_;
    tensor::Tensor value_cache_;
    tensor::Tensor sin_cache_;
    tensor::Tensor cos_cache_;
};
}  // namespace op

#endif