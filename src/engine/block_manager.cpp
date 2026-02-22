/**
 * @file block_manager.cpp
 * @brief 物理 Block 分配管理器实现（基于栈的空闲列表）
 *
 * BlockManager 维护固定数量的物理 Block，提供：
 *   - allocate()：从空闲栈弹出 Block ID（支持单个和批量）
 *   - free()：将 Block ID 压回空闲栈（带重复释放检测）
 *   - get_utilization()：内存利用率 = 已分配 / 总数
 *   - reset()：全量释放并重置栈
 *
 * 可选线程安全模式（通过 LockGuard 条件加锁）。
 */
#include "nanoinfer/engine/block_manager.h"
#include <glog/logging.h>

namespace engine {
BlockManager::BlockManager(int32_t num_blocks, int32_t block_size, bool thread_safe)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      thread_safe_(thread_safe),
      allocated_(num_blocks, false),
      ref_counts_(num_blocks, 0) {
    // 初始化空闲块栈。
    // 使用逆序压栈 (N-1 -> 0)，这样 allocate 时 pop_back()取出的第一个块是 block_id 0
    free_blocks_.reserve(num_blocks);
    for (int32_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(i);
    }

    VLOG(1) << "BlockManager initialized:";
    VLOG(1) << "  Total blocks: " << num_blocks;
    VLOG(1) << "  Block size: " << block_size << " tokens";
    VLOG(1) << "  Total capacity: " << (static_cast<size_t>(num_blocks) * block_size) << " tokens";
    VLOG(1) << "  Thread-safe: " << (thread_safe ? "yes" : "no");
}

/** @brief 分配单个 Block，从空闲栈弹出 Block ID */
base::Status BlockManager::allocate(int32_t& block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (free_blocks_.empty()) {
        return base::error::InternalError("No free blocks available (OOM). Total blocks: " +
                                          std::to_string(num_blocks_) + ", All allocated.");
    }

    block_id = free_blocks_.back();
    free_blocks_.pop_back();

    allocated_[block_id] = true;

    VLOG(2) << "Allocated block " << block_id << " (free blocks remaining: " << free_blocks_.size()
            << ")";

    return base::error::Success();
}

/** @brief 批量分配 Block，连续弹出 num_blocks_needed 个 Block ID */
base::Status BlockManager::allocate(int32_t num_blocks_needed,
                                    std::vector<int32_t>& allocated_blocks) {
    LockGuard lock(mutex_, thread_safe_);

    if (num_blocks_needed <= 0) {
        return base::error::InvalidArgument("Number of blocks must be positive");
    }

    if (static_cast<int32_t>(free_blocks_.size()) < num_blocks_needed) {
        return base::error::InternalError(
            "Insufficient free blocks. Requested: " + std::to_string(num_blocks_needed) +
            ", Available: " + std::to_string(free_blocks_.size()));
    }

    allocated_blocks.clear();
    allocated_blocks.reserve(num_blocks_needed);

    for (int32_t i = 0; i < num_blocks_needed; ++i) {
        int32_t block_id = free_blocks_.back();
        free_blocks_.pop_back();
        allocated_[block_id] = true;
        allocated_blocks.push_back(block_id);
    }

    VLOG(2) << "Allocated batch of " << num_blocks_needed << " blocks"
            << " (free blocks remaining: " << free_blocks_.size() << ")";

    return base::error::Success();
}

/** @brief 释放单个 Block，压回空闲栈（带重复释放检测） */
base::Status BlockManager::free(int32_t block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (!is_valid_block_id(block_id)) {
        return base::error::InvalidArgument("Invalid block ID: " + std::to_string(block_id));
    }

    if (!allocated_[block_id]) {
        return base::error::InvalidArgument("Block " + std::to_string(block_id) +
                                            " is not allocated");
    }

    allocated_[block_id] = false;
    free_blocks_.push_back(block_id);

    VLOG(2) << "Freed block " << block_id << " (free blocks: " << free_blocks_.size() << ")";

    return base::error::Success();
}

/** @brief 批量释放 Block（先校验全部合法，再逐个归还） */
base::Status BlockManager::free(const std::vector<int32_t>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    for (int32_t block_id : block_ids) {
        if (!is_valid_block_id(block_id)) {
            return base::error::InvalidArgument("Invalid block ID in batch: " +
                                                std::to_string(block_id));
        }
        if (!allocated_[block_id]) {
            return base::error::InvalidArgument("Block " + std::to_string(block_id) +
                                                " is not allocated");
        }
    }

    for (int32_t block_id : block_ids) {
        allocated_[block_id] = false;
        free_blocks_.push_back(block_id);
    }

    VLOG(2) << "Freed batch of " << block_ids.size() << " blocks"
            << " (free blocks: " << free_blocks_.size() << ")";

    return base::error::Success();
}

int32_t BlockManager::get_free_block_num() const {
    LockGuard lock(mutex_, thread_safe_);
    return static_cast<int32_t>(free_blocks_.size());
}

int32_t BlockManager::get_allocated_block_num() const {
    LockGuard lock(mutex_, thread_safe_);
    return num_blocks_ - static_cast<int32_t>(free_blocks_.size());
}

bool BlockManager::is_allocated(int32_t block_id) const {
    LockGuard lock(mutex_, thread_safe_);
    if (!is_valid_block_id(block_id)) {
        return false;
    }
    return allocated_[block_id];
}

/** @brief 重置所有 Block 为空闲状态（含 Prefix Cache 清理） */
void BlockManager::reset() {
    LockGuard lock(mutex_, thread_safe_);

    free_blocks_.clear();
    free_blocks_.reserve(num_blocks_);

    for (int32_t i = num_blocks_ - 1; i >= 0; --i) {
        allocated_[i] = false;
        ref_counts_[i] = 0;
        free_blocks_.push_back(i);
    }

    // 清理 Prefix Cache 数据结构
    hash_to_block_.clear();
    block_to_hash_.clear();
    eviction_order_.clear();
    eviction_map_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;

    LOG(INFO) << "BlockManager reset: all " << num_blocks_ << " blocks freed (cache cleared)";
}

float BlockManager::get_utilization() const {
    LockGuard lock(mutex_, thread_safe_);
    if (num_blocks_ == 0) return 0.0f;
    int32_t num_allocated = num_blocks_ - static_cast<int32_t>(free_blocks_.size());
    return static_cast<float>(num_allocated) / static_cast<float>(num_blocks_);
}

// ========== Prefix Caching 实现 ==========

/**
 * @brief 从空闲栈或 LRU 驱逐列表中分配一个 Block
 *
 * 优先使用空闲栈；若空闲栈为空，则驱逐 LRU 队列中最旧的缓存 Block。
 */
base::Status BlockManager::allocate_or_evict(int32_t& block_id) {
    // 注意：调用者已持有锁
    if (!free_blocks_.empty()) {
        block_id = free_blocks_.back();
        free_blocks_.pop_back();
        allocated_[block_id] = true;
        ref_counts_[block_id] = 0;
        return base::error::Success();
    }

    if (!eviction_order_.empty()) {
        // 驱逐最旧的缓存 Block（LRU: front = 最旧）
        block_id = eviction_order_.front();
        eviction_order_.pop_front();
        eviction_map_.erase(block_id);

        // 清除旧的 hash 映射
        auto hash_it = block_to_hash_.find(block_id);
        if (hash_it != block_to_hash_.end()) {
            hash_to_block_.erase(hash_it->second);
            block_to_hash_.erase(hash_it);
        }

        // Block 已是 allocated 状态，直接复用
        ref_counts_[block_id] = 0;
        VLOG(2) << "Evicted cached block " << block_id << " for reuse";
        return base::error::Success();
    }

    return base::error::InternalError(
        "No free or evictable blocks available (OOM). Total blocks: " +
        std::to_string(num_blocks_));
}

/**
 * @brief 尝试分配一个与 hash 匹配的缓存 Block（Prefix Caching 核心方法）
 *
 * 查找顺序：
 *   1. hash_to_block_ 命中 → 复用现有 Block（incref）
 *   2. 未命中 → 从 free_blocks_ 或 eviction_list 分配新 Block
 */
base::Status BlockManager::allocate_cached(uint64_t block_hash, int32_t& block_id,
                                           bool& cache_hit) {
    LockGuard lock(mutex_, thread_safe_);

    // 查找 hash 是否已缓存
    auto it = hash_to_block_.find(block_hash);
    if (it != hash_to_block_.end()) {
        block_id = it->second;

        // 如果该 Block 在驱逐列表中 (ref_count=0)，从中移除并重新激活
        if (ref_counts_[block_id] == 0) {
            auto eit = eviction_map_.find(block_id);
            if (eit != eviction_map_.end()) {
                eviction_order_.erase(eit->second);
                eviction_map_.erase(eit);
            }
        }
        ref_counts_[block_id]++;
        cache_hit = true;
        cache_hits_++;

        VLOG(2) << "Prefix cache HIT: hash=" << block_hash << " → block " << block_id
                << " (ref_count=" << ref_counts_[block_id] << ")";
        return base::error::Success();
    }

    // Cache miss: 分配新 Block
    base::Status status = allocate_or_evict(block_id);
    if (!status) {
        return status;
    }

    // 注册 hash 映射
    ref_counts_[block_id] = 1;
    hash_to_block_[block_hash] = block_id;
    block_to_hash_[block_id] = block_hash;
    cache_hit = false;
    cache_misses_++;

    VLOG(2) << "Prefix cache MISS: hash=" << block_hash << " → new block " << block_id;
    return base::error::Success();
}

/**
 * @brief 释放 Block（引用计数模式）
 *
 * - ref_count > 0 的 Block：decref；若变为 0 且有 hash → 加入 LRU 驱逐列表
 * - ref_count = 0 的 Block（非缓存 Block）：直接释放回空闲栈
 */
void BlockManager::release(int32_t block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (!is_valid_block_id(block_id) || !allocated_[block_id]) {
        LOG(WARNING) << "Attempt to release invalid or unallocated block " << block_id;
        return;
    }

    if (ref_counts_[block_id] > 0) {
        ref_counts_[block_id]--;
        if (ref_counts_[block_id] == 0) {
            if (block_to_hash_.count(block_id)) {
                // 有 hash → 保留 KV 数据，加入驱逐候选列表
                eviction_order_.push_back(block_id);
                eviction_map_[block_id] = std::prev(eviction_order_.end());
                VLOG(2) << "Block " << block_id
                        << " added to eviction list (ref_count=0, hash preserved)";
            } else {
                // 无 hash → 直接释放
                allocated_[block_id] = false;
                free_blocks_.push_back(block_id);
                VLOG(2) << "Block " << block_id << " freed directly (no hash)";
            }
        }
    } else {
        // ref_count = 0: 这是通过普通 allocate() 分配的非缓存 Block
        allocated_[block_id] = false;
        free_blocks_.push_back(block_id);
        VLOG(2) << "Block " << block_id << " freed (non-cached block)";
    }
}

int32_t BlockManager::get_ref_count(int32_t block_id) const {
    LockGuard lock(mutex_, thread_safe_);
    if (!is_valid_block_id(block_id)) return 0;
    return ref_counts_[block_id];
}

int32_t BlockManager::get_evictable_block_num() const {
    LockGuard lock(mutex_, thread_safe_);
    return static_cast<int32_t>(eviction_order_.size());
}

int32_t BlockManager::get_available_block_num() const {
    LockGuard lock(mutex_, thread_safe_);
    return static_cast<int32_t>(free_blocks_.size() + eviction_order_.size());
}

}  // namespace engine
