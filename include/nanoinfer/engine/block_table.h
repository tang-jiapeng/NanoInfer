/**
 * @file block_table.h
 * @brief 序列 → 物理 Block 映射表，支持转换为 GPU 格式供 PagedAttention Kernel 使用
 */
#ifndef NANO_INFER_BLOCK_TABLE_H
#define NANO_INFER_BLOCK_TABLE_H

#include <unordered_map>
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

namespace engine {

/**
 * @brief 序列块表 (Block Table)
 *
 * 维护 seq_id → [block_0, block_1, ...] 映射；
 * to_gpu_format() 输出 [num_seqs, max_blocks_per_seq] Tensor 供 Kernel 使用。
 */
class BlockTable {
   public:
    explicit BlockTable(bool thread_safe = false);

    ~BlockTable() = default;

    /// @brief 为新序列分配初始 Block 列表
    base::Status allocate_sequence(int32_t seq_id, const std::vector<int32_t>& block_ids);

    /// @brief 向序列追加单个 Block
    base::Status append_block(int32_t seq_id, int32_t block_id);

    /// @brief 向序列追加多个 Block
    base::Status append_blocks(int32_t seq_id, const std::vector<int32_t>& block_ids);

    base::Status get_blocks(int32_t seq_id, std::vector<int32_t>& block_ids) const;

    int32_t get_num_blocks(int32_t seq_id) const;

    /// @brief 释放序列，返回其持有的 Block ID 列表
    base::Status free_sequence(int32_t seq_id, std::vector<int32_t>& freed_blocks);

    bool has_sequence(int32_t seq_id) const;

    std::vector<int32_t> get_sequence_ids() const;

    int32_t get_num_sequences() const;

    /**
     * @brief 转换为 GPU 格式 [num_seqs, max_blocks_per_seq]，不足填 -1
     * @param seq_ids 当前 Batch 的序列 ID
     * @param max_blocks_per_seq Padding 长度
     * @param tensor 输出 CPU Tensor（需调用者拷贝到 GPU）
     */
    base::Status to_gpu_format(const std::vector<int32_t>& seq_ids, int32_t max_blocks_per_seq,
                               tensor::Tensor& tensor) const;

    void reset();

   private:
    bool thread_safe_;
    std::unordered_map<int32_t, std::vector<int32_t>> seq_to_blocks_;
    mutable std::mutex mutex_;

    class LockGuard {
       public:
        LockGuard(std::mutex& mutex, bool enabled) : mutex_(mutex), enabled_(enabled) {
            if (enabled_) mutex_.lock();
        }
        ~LockGuard() {
            if (enabled_) mutex_.unlock();
        }

       private:
        std::mutex& mutex_;
        bool enabled_;
    };
};

}  // namespace engine

#endif