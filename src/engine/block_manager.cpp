#include "nanoinfer/engine/block_manager.h"
#include <glog/logging.h>

namespace engine {
BlockManager::BlockManager(int32_t num_blocks, int32_t block_size, bool thread_safe)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      thread_safe_(thread_safe),
      allocated_(num_blocks, false) {
    // 初始化空闲块栈。
    // 使用逆序压栈 (N-1 -> 0)，这样 allocate 时 pop_back()取出的第一个块是 block_id 0
    free_blocks_.reserve(num_blocks);
    for (int32_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(i);
    }

    VLOG(1) << "BlockManager initialized:";
    VLOG(1) << "  Total blocks: " << num_blocks;
    VLOG(1) << "  Block size: " << block_size << " tokens";
    VLOG(1) << "  Total capacity: " << (static_cast<size_t>(num_blocks) * block_size)
            << " tokens";
    VLOG(1) << "  Thread-safe: " << (thread_safe ? "yes" : "no");
}

base::Status BlockManager::allocate(int32_t& block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (free_blocks_.empty()) {
        return base::error::InternalError(
            "No free blocks available (OOM). Total blocks: " +
            std::to_string(num_blocks_) + ", All allocated.");
    }

    block_id = free_blocks_.back();
    free_blocks_.pop_back();

    allocated_[block_id] = true;

    VLOG(2) << "Allocated block " << block_id
            << " (free blocks remaining: " << free_blocks_.size() << ")";

    return base::error::Success();
}

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

base::Status BlockManager::free(int32_t block_id) {
    LockGuard lock(mutex_, thread_safe_);

    if (!is_valid_block_id(block_id)) {
        return base::error::InvalidArgument("Invalid block ID: " +
                                            std::to_string(block_id));
    }

    if (!allocated_[block_id]) {
        return base::error::InvalidArgument("Block " + std::to_string(block_id) +
                                            " is not allocated");
    }

    allocated_[block_id] = false;
    free_blocks_.push_back(block_id);

    VLOG(2) << "Freed block " << block_id << " (free blocks: " << free_blocks_.size()
            << ")";

    return base::error::Success();
}

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

void BlockManager::reset() {
    LockGuard lock(mutex_, thread_safe_);

    free_blocks_.clear();
    free_blocks_.reserve(num_blocks_);

    for (int32_t i = num_blocks_ - 1; i >= 0; --i) {
        allocated_[i] = false;
        free_blocks_.push_back(i);
    }

    LOG(INFO) << "BlockManager reset: all " << num_blocks_ << " blocks freed";
}

float BlockManager::get_utilization() const {
    LockGuard lock(mutex_, thread_safe_);
    if (num_blocks_ == 0) return 0.0f;
    int32_t num_allocated = num_blocks_ - static_cast<int32_t>(free_blocks_.size());
    return static_cast<float>(num_allocated) / static_cast<float>(num_blocks_);
}

}  // namespace engine
