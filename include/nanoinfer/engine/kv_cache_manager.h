/**
 * @file kv_cache_manager.h
 * @brief KV Cache 管理器：物理显存分配 + 逻辑映射 + Cache Tensor 持有
 */
#ifndef NANO_INFER_KV_CACHE_MANAGER_H
#define NANO_INFER_KV_CACHE_MANAGER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "nanoinfer/engine/block_manager.h"
#include "nanoinfer/engine/block_table.h"
#include "nanoinfer/tensor/tensor.h"

namespace engine {

/**
 * @brief KV Cache 管理器 (PagedAttention 核心)
 *
 * 物理布局: [num_blocks, num_kv_heads, block_size, head_size]
 * 逻辑视图: 每个 Sequence 由若干非连续 Block 组成。
 */
class KVCacheManager {
   public:
    KVCacheManager(int32_t num_blocks, int32_t block_size, int32_t num_layers, int32_t num_kv_heads,
                   int32_t head_size);

    ~KVCacheManager() = default;

    /// @brief 分配 GPU 显存，初始化 BlockManager / BlockTable
    base::Status init(std::shared_ptr<base::DeviceAllocator> allocator);

    /// @brief 为新序列分配初始 Block
    base::Status allocate_sequence(int32_t seq_id, int32_t num_tokens);

    /**
     * @brief 扩展序列（Decode 阶段动态增长）
     * @param num_allocated_blocks 输出实际新分配的 Block 数
     */
    base::Status extend_sequence(int32_t seq_id, int32_t additional_tokens,
                                 int32_t* num_allocated_blocks = nullptr);

    base::Status free_sequence(int32_t seq_id);

    tensor::Tensor& get_key_cache(int32_t layer_idx);

    tensor::Tensor& get_value_cache(int32_t layer_idx);

    /// @brief 生成 GPU 格式 Block Table [num_seqs, max_blocks_per_seq]
    base::Status get_block_table_tensor(const std::vector<int32_t>& seq_ids,
                                        tensor::Tensor& block_table_tensor);

    int32_t get_num_blocks(int32_t seq_id) const;

    int32_t get_sequence_capacity(int32_t seq_id) const;

    bool is_sequence_allocated(int32_t seq_id) const;

    int32_t get_free_block_num() const;

    int32_t get_total_block_num() const;

    float get_utilization() const;

    int32_t get_max_blocks_per_seq() const;

    void reset();

    int32_t layer_num() const {
        return num_layers_;
    }
    int32_t kv_heads_num() const {
        return num_kv_heads_;
    }
    int32_t head_size() const {
        return head_size_;
    }
    int32_t block_size() const {
        return block_size_;
    }

   private:
    int32_t calculate_blocks_needed(int32_t num_tokens) const {
        return (num_tokens + block_size_ - 1) / block_size_;
    }

    int32_t num_blocks_;
    int32_t block_size_;
    int32_t num_layers_;
    int32_t num_kv_heads_;
    int32_t head_size_;

    bool initialized_ = false;

    std::unique_ptr<BlockManager> block_manager_;
    std::unique_ptr<BlockTable> block_table_;

    std::vector<tensor::Tensor> key_caches_;
    std::vector<tensor::Tensor> value_caches_;

    std::unordered_map<int32_t, int32_t> seq_num_tokens_;
};
}  // namespace engine

#endif