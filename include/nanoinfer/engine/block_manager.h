/**
 * @file block_manager.h
 * @brief 物理显存块管理器（PagedAttention 核心组件）
 */
#ifndef NANO_INFER_BLOCK_MANAGER_H
#define NANO_INFER_BLOCK_MANAGER_H

#include <mutex>
#include <vector>
#include "nanoinfer/base/base.h"

namespace engine {

/**
 * @brief 物理 Block 分配/释放管理器
 *
 * 维护 Block 空闲栈与分配位图，支持可选线程安全。
 */
class BlockManager {
   public:
    /**
     * @brief 构造
     * @param num_blocks 物理块总数
     * @param block_size 每块容纳的 Token 数
     * @param thread_safe 是否启用线程安全
     */
    explicit BlockManager(int32_t num_blocks, int32_t block_size, bool thread_safe = false);

    ~BlockManager() = default;

    /// @brief 分配单个 Block，block_id 为输出
    base::Status allocate(int32_t& block_id);

    /// @brief 批量分配 Block
    base::Status allocate(int32_t num_blocks_needed, std::vector<int32_t>& allocated_blocks);

    base::Status free(int32_t block_id);

    base::Status free(const std::vector<int32_t>& block_ids);

    int32_t get_free_block_num() const;

    int32_t get_total_block_num() const {
        return num_blocks_;
    }

    int32_t get_block_size() const {
        return block_size_;
    }

    int32_t get_allocated_block_num() const;

    bool is_allocated(int32_t block_id) const;

    void reset();

    float get_utilization() const;

   private:
    class LockGuard {
       public:
        explicit LockGuard(std::mutex& mutex, bool enabled) : mutex_(mutex), enabled_(enabled) {
            if (enabled_) {
                mutex_.lock();
            }
        }
        ~LockGuard() {
            if (enabled_) {
                mutex_.unlock();
            }
        }

       private:
        std::mutex& mutex_;
        bool enabled_;
    };

    bool is_valid_block_id(int32_t block_id) const {
        return block_id >= 0 && block_id < num_blocks_;
    }

    int32_t num_blocks_;
    int32_t block_size_;
    bool thread_safe_;
    std::vector<int32_t> free_blocks_;
    std::vector<bool> allocated_;
    mutable std::mutex mutex_;
};

}  // namespace engine

#endif