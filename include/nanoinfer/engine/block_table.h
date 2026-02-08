#ifndef NANO_INFER_BLOCK_TABLE_H
#define NANO_INFER_BLOCK_TABLE_H

#include <unordered_map>
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

namespace engine {

/**
 * @brief 序列块表管理器 (Block Table)
 *
 * 维护逻辑序列 (Sequence) 到物理块 (Physical Blocks) 的映射关系。
 * 支持动态追加块、查询块列表，并将映射表转换为 GPU 可读的 Tensor 格式 (PagedAttention
 * 需要)。
 *
 * Example:
 *   Sequence 0: [block 5, block 12, block 3]   (3 blocks, non-contiguous)
 *   Sequence 1: [block 0, block 7]             (2 blocks)
 *   Sequence 2: [block 9, block 2, block 11, block 6]  (4 blocks)
 *
 * GPU Format (flat array with padding):
 *   [5, 12, 3, -1,     // seq 0 (3 blocks, padded to max_blocks_per_seq)
 *    0, 7, -1, -1,     // seq 1 (2 blocks)
 *    9, 2, 11, 6]      // seq 2 (4 blocks)
 */
class BlockTable {
   public:
    /**
     * @brief 构造函数
     * @param thread_safe 是否开启线程安全保护 (默认 false)
     */
    explicit BlockTable(bool thread_safe = false);

    ~BlockTable() = default;

    /**
     * @brief 为新序列分配初始块
     *
     * @param seq_id 序列 ID
     * @param block_ids 初始物理块 ID 列表
     * @return base::Status 如果 seq_id 已存在则返回错误
     */
    base::Status allocate_sequence(int32_t seq_id, const std::vector<int32_t>& block_ids);

    /**
     * @brief 向现有序列追加一个物理块
     * 通常用于解码阶段生成新 token 后动态扩展
     *
     * @param seq_id 序列 ID
     * @param block_id 要追加的物理块 ID
     * @return base::Status 如果序列不存在则返回错误
     */
    base::Status append_block(int32_t seq_id, int32_t block_id);

    /**
     * @brief 向现有序列追加多个物理块
     *
     * @param seq_id 序列 ID
     * @param block_ids 要追加的物理块 ID 列表
     */
    base::Status append_blocks(int32_t seq_id, const std::vector<int32_t>& block_ids);

    /**
     * @brief 获取序列的所有物理块
     *
     * @param seq_id 序列 ID
     * @param blocks [输出] 物理块 ID 列表
     * @return base::Status 如果序列不存在则返回错误
     */
    base::Status get_blocks(int32_t seq_id, std::vector<int32_t>& block_ids) const;

    /**
     * @brief 获取序列已分配的块数量
     *
     * @param seq_id 序列 ID
     * @return int32_t 块数量 (如果序列不存在返回 -1 或抛出日志)
     */
    int32_t get_num_blocks(int32_t seq_id) const;

    /**
     * @brief 释放序列并移除记录
     *
     * @param seq_id 序列 ID
     * @param freed_blocks [输出] 该序列持有的所有物理块 ID (用于归还给 BlockManager)
     * @return base::Status 如果序列不存在则返回错误
     */
    base::Status free_sequence(int32_t seq_id, std::vector<int32_t>& freed_blocks);

    /**
     * @brief 检查序列是否存在
     */
    bool has_sequence(int32_t seq_id) const;

    /**
     * @brief 获取所有活跃的序列 ID
     */
    std::vector<int32_t> get_sequence_ids() const;

    /**
     * @brief 获取当前活跃序列总数
     */
    int32_t get_num_sequences() const;

    /**
     * @brief 转换为 GPU 格式的 Block Table
     *
     * 生成一个 Shape 为 [num_seqs, max_blocks_per_seq] 的 Tensor，
     * 用于传给 PagedAttention Kernel。
     *
     * @param seq_ids 需要包含的序列 ID 列表 (通常是当前 Batch 中的请求)
     * @param max_blocks_per_seq 每个序列的最大块数 (用于 Padding，不足填 -1)
     * @param tensor [输出] 生成的 CPU Tensor (后续需由调用者拷贝到 GPU)
     * @return base::Status
     */
    base::Status to_gpu_format(const std::vector<int32_t>& seq_ids,
                               int32_t max_blocks_per_seq, tensor::Tensor& tensor) const;

    /**
     * @brief 重置清空所有数据
     * @warning 调用前需确保物理块已通过 BlockManager 释放
     */
    void reset();

   private:
    bool thread_safe_;  ///< 线程安全标志

    /// 映射表: seq_id -> [physical_block_0, physical_block_1, ...]
    std::unordered_map<int32_t, std::vector<int32_t>> seq_to_blocks_;

    mutable std::mutex mutex_;  ///< 保护 seq_to_blocks_ 的互斥锁

    /// 内部使用的条件锁
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