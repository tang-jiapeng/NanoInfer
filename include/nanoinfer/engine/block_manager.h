/**
 * @file block_manager.h
 * @brief 物理显存块管理器（PagedAttention 核心组件）
 */
#ifndef NANO_INFER_BLOCK_MANAGER_H
#define NANO_INFER_BLOCK_MANAGER_H

#include <list>
#include <mutex>
#include <unordered_map>
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

    // ========== Prefix Caching API ==========

    /// @brief 尝试分配一个与 hash 匹配的缓存 Block（Prefix Caching）
    /// @param block_hash 该 Block 的内容哈希（含链式前缀信息）
    /// @param[out] block_id 分配到的 Block ID
    /// @param[out] cache_hit 是否命中缓存（true=KV 数据有效，无需重算）
    base::Status allocate_cached(uint64_t block_hash, int32_t& block_id, bool& cache_hit);

    /// @brief 释放 Block（引用计数模式：有 hash → 加入驱逐候选；无 hash → 直接释放）
    void release(int32_t block_id);

    /// @brief 获取指定 Block 的引用计数
    int32_t get_ref_count(int32_t block_id) const;

    /// @brief 获取可驱逐 Block 数（ref_count=0 但保留 KV 数据的缓存 Block）
    int32_t get_evictable_block_num() const;

    /// @brief 获取可用 Block 数（空闲 + 可驱逐）
    int32_t get_available_block_num() const;

    /// @brief Prefix Cache 命中/未命中统计
    int64_t get_cache_hits() const {
        return cache_hits_;
    }
    int64_t get_cache_misses() const {
        return cache_misses_;
    }

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

    /// @brief 从空闲栈或驱逐列表中分配一个 Block
    base::Status allocate_or_evict(int32_t& block_id);

    int32_t num_blocks_;
    int32_t block_size_;
    bool thread_safe_;
    std::vector<int32_t> free_blocks_;
    std::vector<bool> allocated_;
    mutable std::mutex mutex_;

    // ---- Prefix Caching 数据结构 ----
    std::vector<int32_t> ref_counts_;                      ///< 每个 Block 的引用计数
    std::unordered_map<uint64_t, int32_t> hash_to_block_;  ///< Hash → Block ID
    std::unordered_map<int32_t, uint64_t> block_to_hash_;  ///< Block ID → Hash（反向映射）
    std::list<int32_t> eviction_order_;                    ///< LRU 驱逐队列 (front=最旧)
    std::unordered_map<int32_t, std::list<int32_t>::iterator> eviction_map_;  ///< Block → 迭代器
    int64_t cache_hits_ = 0;
    int64_t cache_misses_ = 0;
};

}  // namespace engine

#endif