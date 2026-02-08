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
 * @brief 基于块的 KV Cache 管理器 (PagedAttention 核心)
 *
 * 职责：
 * 1. 管理物理显存 (通过 BlockManager)
 * 2. 维护逻辑映射 (通过 BlockTable)
 * 3. 持有实际的 KV Cache Tensor 数据
 *
 * 内存布局 (Physical):
 * [num_blocks, num_kv_heads, block_size, head_size]
 *
 * 逻辑视图 (Logical):
 * 每个 Sequence 由若干个非连续的 Block 组成。
 */
class KVCacheManager {
   public:
    /**
     * @brief 构造函数
     *
     * @param num_blocks 物理块总数
     * @param block_size 每个块的 token 容量 (e.g. 16)
     * @param num_layers Transformer 层数
     * @param num_kv_heads KV Head 数量 (GQA/MQA 下通常小于 Q Head)
     * @param head_size 每个 Head 的维度
     */
    KVCacheManager(int32_t num_blocks, int32_t block_size, int32_t num_layers,
                   int32_t num_kv_heads, int32_t head_size);

    ~KVCacheManager() = default;

    /**
     * @brief 初始化 KV Cache
     * 分配 GPU 显存，初始化 BlockManager 和 BlockTable
     */
    base::Status init(std::shared_ptr<base::DeviceAllocator> allocator);

    /**
     * @brief 为新序列分配初始 Block
     *
     * @param seq_id 序列 ID
     * @param num_tokens 初始 token 数量
     * @return base::Status
     */
    base::Status allocate_sequence(int32_t seq_id, int32_t num_tokens);

    /**
     * @brief 扩展序列 (用于生成过程中动态增加 Token)
     * @param seq_id 序列 ID
     * @param additional_tokens 新增的 token 数量
     * @param num_allocated_blocks [输出] 实际新分配的 block 数量
     */
    base::Status extend_sequence(int32_t seq_id, int32_t additional_tokens,
                                 int32_t* num_allocated_blocks = nullptr);

    /**
     * @brief 释放序列占用的所有资源
     */
    base::Status free_sequence(int32_t seq_id);

    /**
     * @brief 获取指定层的 Key Cache Tensor
     * @return Tensor& (Shape: [num_blocks, num_kv_heads, block_size, head_size])
     */
    tensor::Tensor& get_key_cache(int32_t layer_idx);

    /**
     * @brief 获取指定层的 Value Cache Tensor
     */
    tensor::Tensor& get_value_cache(int32_t layer_idx);

    /**
     * @brief 生成 GPU 格式的 Block Table
     *
     * @param seq_ids 需要包含的序列 ID 列表
     * @param block_table_tensor [输出] 生成的 Tensor (Shape: [num_seqs,
     * max_blocks_per_seq])
     */
    base::Status get_block_table_tensor(const std::vector<int32_t>& seq_ids,
                                        tensor::Tensor& block_table_tensor);

    /**
     * @brief 获取序列当前的 Block 数量
     */
    int32_t get_num_blocks(int32_t seq_id) const;

    int32_t get_sequence_capacity(int32_t seq_id) const;

    /**
     * @brief 检查序列是否已分配
     */
    bool is_sequence_allocated(int32_t seq_id) const;

    /**
     * @brief 获取空闲 Block 数量
     */
    int32_t get_free_block_num() const;

    /**
     * @brief 获取总 Block 数量
     */
    int32_t get_total_block_num() const;

    /**
     * @brief 获取显存利用率
     */
    float get_utilization() const;

    /**
     * @brief 获取为了 Kernel 配置所需的最大 Padding 长度
     */
    int32_t get_max_blocks_per_seq() const;

    /**
     * @brief 重置所有状态
     */
    void reset();

    // Accessors
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
    // 计算需要的 Block 数量 (向上取整)
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

    // Shape: [num_blocks, num_kv_heads, block_size, head_size]
    std::vector<tensor::Tensor> key_caches_;
    std::vector<tensor::Tensor> value_caches_;

    // 跟踪每个序列当前的 Token 数量，用于计算是否需要扩展 Block
    std::unordered_map<int32_t, int32_t> seq_num_tokens_;
};
}  // namespace engine

#endif