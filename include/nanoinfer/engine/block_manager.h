#ifndef NANO_INFER_BLOCK_MANAGER_H
#define NANO_INFER_BLOCK_MANAGER_H

#include <mutex>
#include <vector>
#include "nanoinfer/base/base.h"

namespace engine {

/**
 * @brief 显存块管理器 (Paged Attention 核心组件)
 * 管理物理显存块的分配与释放，维护逻辑 Block 到物理 Block 的映射池
 */
class BlockManager {
   public:
    /**
     * @brief 构造函数
     *
     * @param num_blocks 物理块总数
     * @param block_size 每个块包含的 token 数 (通常为 16 或 32)
     * @param thread_safe 是否开启线程安全保护 (默认 false)
     */
    explicit BlockManager(int32_t num_blocks, int32_t block_size,
                          bool thread_safe = false);

    ~BlockManager() = default;

    /**
     * @brief 分配单个物理块
     * @param block_id [输出] 分配到的 Block ID
     * @return base::Status 如果显存耗尽返回错误
     */
    base::Status allocate(int32_t& block_id);

    /**
     * @brief 分配多个物理块
     * @param num_blocks_needed 需要分配的块数量
     * @param allocated_blocks [输出] 分配到的 Block ID 列表
     */
    base::Status allocate(int32_t num_blocks_needed,
                          std::vector<int32_t>& allocated_blocks);

    /**
     * @brief 释放单个物理块
     */
    base::Status free(int32_t block_id);

    /**
     * @brief 批量释放物理块
     */
    base::Status free(const std::vector<int32_t>& block_ids);

    /**
     * @brief 获取当前空闲块数量
     */
    int32_t get_free_block_num() const;

    /**
     * @brief 获取总块数
     */
    int32_t get_total_block_num() const {
        return num_blocks_;
    }

    /**
     * @brief 获取块大小 (token 数)
     */
    int32_t get_block_size() const {
        return block_size_;
    }

    /**
     * @brief 获取已分配块数量
     */
    int32_t get_allocated_block_num() const;

    /**
     * @brief 检查指定 Block ID 是否已被分配
     */
    bool is_allocated(int32_t block_id) const;

    /**
     * @brief 重置所有块为空闲状态
     * @warning 仅应在没有活跃请求时调用
     */
    void reset();

    /**
     * @brief 获取显存利用率 (0.0 ~ 1.0)
     */
    float get_utilization() const;

   private:
    /**
     * @brief 简单的内部 LockGuard，用于根据 thread_safe_ 标志条件加锁
     */
    class LockGuard {
       public:
        explicit LockGuard(std::mutex& mutex, bool enabled)
            : mutex_(mutex), enabled_(enabled) {
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

    int32_t num_blocks_;                ///< 总块数
    int32_t block_size_;                ///< 块大小
    bool thread_safe_;                  ///< 线程安全标记
    std::vector<int32_t> free_blocks_;  ///< 空闲块栈 (Stack)
    std::vector<bool> allocated_;       ///< 分配状态位图
    mutable std::mutex mutex_;          ///< 互斥锁
};

}  // namespace engine

#endif