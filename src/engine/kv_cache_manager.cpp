/**
 * @file kv_cache_manager.cpp
 * @brief KV Cache 统一管理器实现（显存分配 + Block 映射 + 序列生命周期）
 *
 * KVCacheManager 组合 BlockManager 和 BlockTable，提供序列级别的 KV Cache 操作：
 *   - init()：为所有层分配 Key/Value Cache Tensor
 *     Shape: [num_blocks, num_kv_heads, block_size, head_size] × 2 × num_layers
 *   - allocate_sequence()：计算需要的 Block 数 → BlockManager 分配 → BlockTable 注册
 *   - extend_sequence()：追加 Token 时按需扩展 Block
 *   - free_sequence()：释放并归还 Block
 */
#include "nanoinfer/engine/kv_cache_manager.h"

namespace engine {

KVCacheManager::KVCacheManager(int32_t num_blocks, int32_t block_size, int32_t num_layers,
                               int32_t num_kv_heads, int32_t head_size)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size) {
    block_manager_ = std::make_unique<BlockManager>(num_blocks, block_size);
    block_table_ = std::make_unique<BlockTable>();

    LOG(INFO) << "KVCacheManager created:";
    LOG(INFO) << "  Num blocks: " << num_blocks;
    LOG(INFO) << "  Block size: " << block_size << " tokens";
    LOG(INFO) << "  Total capacity: " << (static_cast<size_t>(num_blocks) * block_size)
              << " tokens";
    LOG(INFO) << "  Num layers: " << num_layers;
    LOG(INFO) << "  KV heads: " << num_kv_heads;
    LOG(INFO) << "  Head size: " << head_size;
}

/**
 * @brief 初始化 KV Cache 显存
 *
 * 为每一层分配 Key/Value Cache Tensor，Shape 均为
 * [num_blocks, num_kv_heads, block_size, head_size]，共 num_layers × 2 个 Tensor。
 */
base::Status KVCacheManager::init(std::shared_ptr<base::DeviceAllocator> allocator) {
    if (initialized_) {
        return base::error::InvalidArgument("KVCacheManager is already initialized");
    }
    if (!allocator) {
        return base::error::InvalidArgument("Allocator cannot be null");
    }

    // 计算总显存占用
    // Shape: [num_blocks, num_kv_heads, block_size, head_size]
    size_t elements_per_block = static_cast<size_t>(num_kv_heads_) * block_size_ * head_size_;
    size_t elements_per_layer = elements_per_block * num_blocks_;
    size_t bytes_per_layer = elements_per_layer * sizeof(float);
    size_t total_bytes = bytes_per_layer * num_layers_ * 2;  // Key + Value

    LOG(INFO) << "Initializing block-based KV cache:";
    LOG(INFO) << "  Bytes per layer: " << (bytes_per_layer / (1024.0 * 1024.0)) << " MB";
    LOG(INFO) << "  Total cache size: " << (total_bytes / (1024.0 * 1024.0)) << " MB";

    key_caches_.reserve(num_layers_);
    value_caches_.reserve(num_layers_);

    for (int32_t i = 0; i < num_layers_; ++i) {
        // Allocate Key Cache
        tensor::Tensor k_cache(base::DataType::kDataTypeFp32, num_blocks_, num_kv_heads_,
                               block_size_, head_size_, true, allocator);
        if (k_cache.ptr<float>() == nullptr) {
            return base::error::InternalError("Failed to allocate Key Cache for layer " +
                                              std::to_string(i));
        }
        key_caches_.push_back(std::move(k_cache));

        tensor::Tensor v_cache(base::DataType::kDataTypeFp32, num_blocks_, num_kv_heads_,
                               block_size_, head_size_, true, allocator);
        if (v_cache.ptr<float>() == nullptr) {
            return base::error::InternalError("Failed to allocate Value Cache for layer " +
                                              std::to_string(i));
        }
        value_caches_.push_back(std::move(v_cache));
    }

    initialized_ = true;
    LOG(INFO) << "KV cache initialization complete";
    return base::error::Success();
}

/**
 * @brief 为新序列分配 KV Cache Block
 *
 * 流程：计算 Block 数 → BlockManager 分配物理 Block → BlockTable 注册映射。
 * 失败时自动回滚已分配的 Block。
 */
base::Status KVCacheManager::allocate_sequence(int32_t seq_id, int32_t num_tokens) {
    if (!initialized_) {
        return base::error::InvalidArgument("KVCacheManager is not initialized");
    }
    if (block_table_->has_sequence(seq_id)) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " already allocated");
    }
    // 计算需要的 Block 数
    int32_t num_blocks_needed = calculate_blocks_needed(num_tokens);

    // 从 BlockManager 分配物理 Block
    std::vector<int32_t> block_ids;
    base::Status status = block_manager_->allocate(num_blocks_needed, block_ids);
    if (!status) {
        return status;
    }
    // 在 BlockTable 注册映射
    status = block_table_->allocate_sequence(seq_id, block_ids);
    if (!status) {
        // 回滚：释放已分配的 Block
        block_manager_->free(block_ids);
        return status;
    }

    // 记录 Token 数量
    seq_num_tokens_[seq_id] = num_tokens;

    VLOG(1) << "Allocated sequence " << seq_id << ": " << num_tokens << " tokens, "
            << num_blocks_needed << " blocks";

    return base::error::Success();
}

/**
 * @brief 扩展已有序列的 KV Cache（追加 Token 时按需分配新 Block）
 * @param additional_tokens 新增 Token 数
 * @param[out] num_allocated_blocks 本次实际新分配的 Block 数（可为 0）
 */
base::Status KVCacheManager::extend_sequence(int32_t seq_id, int32_t additional_tokens,
                                             int32_t* num_allocated_blocks) {
    if (!initialized_) {
        return base::error::InvalidArgument("KVCacheManager is not initialized");
    }
    if (!block_table_->has_sequence(seq_id)) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }
    // 获取当前状态
    int32_t current_num_blocks = block_table_->get_num_blocks(seq_id);
    if (current_num_blocks < 0) {
        return base::error::InternalError("Failed to get block count for seq " +
                                          std::to_string(seq_id));
    }

    int32_t current_tokens = seq_num_tokens_[seq_id];
    int32_t new_total_tokens = current_tokens + additional_tokens;

    // 计算新的 Block 需求
    int32_t total_blocks_needed = calculate_blocks_needed(new_total_tokens);
    int32_t additional_blocks_needed = total_blocks_needed - current_num_blocks;

    if (additional_blocks_needed > 0) {
        // 需要分配新 Block
        std::vector<int32_t> new_block_ids;
        base::Status status = block_manager_->allocate(additional_blocks_needed, new_block_ids);
        if (!status) {
            return status;
        }

        // 追加到 BlockTable
        status = block_table_->append_blocks(seq_id, new_block_ids);
        if (!status) {
            block_manager_->free(new_block_ids);
            return status;
        }

        VLOG(2) << "Extended sequence " << seq_id << " with " << additional_blocks_needed
                << " new blocks";
    }

    // 更新 Token 计数
    seq_num_tokens_[seq_id] = new_total_tokens;

    if (num_allocated_blocks) {
        *num_allocated_blocks = std::max(0, additional_blocks_needed);
    }

    return base::error::Success();
}

/** @brief 释放序列的所有 KV Cache Block（支持 Prefix Caching 引用计数） */
base::Status KVCacheManager::free_sequence(int32_t seq_id) {
    if (!block_table_->has_sequence(seq_id)) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }
    // 从 BlockTable 移除并获取持有的 Block IDs
    std::vector<int32_t> block_ids;
    base::Status status = block_table_->free_sequence(seq_id, block_ids);
    if (!status) {
        return status;
    }

    if (cached_sequences_.count(seq_id)) {
        // Prefix Caching 序列：使用 release (ref counting)
        // 每个 block 根据自身状态决定是加入驱逐列表还是直接释放
        for (int32_t block_id : block_ids) {
            block_manager_->release(block_id);
        }
        cached_sequences_.erase(seq_id);
    } else {
        // 普通序列：直接释放
        status = block_manager_->free(block_ids);
        if (!status) {
            LOG(ERROR) << "Failed to free blocks for sequence " << seq_id << ": "
                       << status.get_err_msg();
            return status;
        }
    }

    // 移除元数据跟踪
    seq_num_tokens_.erase(seq_id);

    VLOG(1) << "Freed sequence " << seq_id << " (" << block_ids.size() << " blocks)";
    return base::error::Success();
}

tensor::Tensor& KVCacheManager::get_key_cache(int32_t layer_idx) {
    CHECK_LT(layer_idx, num_layers_);
    return key_caches_[layer_idx];
}

tensor::Tensor& KVCacheManager::get_value_cache(int32_t layer_idx) {
    CHECK_LT(layer_idx, num_layers_);
    return value_caches_[layer_idx];
}

/**
 * @brief 获取指定序列集合的 Block Table Tensor（用于 GPU Kernel 输入）
 *
 * 调用 BlockTable::to_gpu_format 生成 [num_seqs, max_blocks_per_seq] 的 Int32 Tensor。
 */
base::Status KVCacheManager::get_block_table_tensor(const std::vector<int32_t>& seq_ids,
                                                    tensor::Tensor& block_table_tensor) {
    if (seq_ids.empty()) {
        return base::error::InvalidArgument("seq_ids cannot be empty");
    }

    int32_t max_blocks = get_max_blocks_per_seq();
    return block_table_->to_gpu_format(seq_ids, max_blocks, block_table_tensor);
}

int32_t KVCacheManager::get_num_blocks(int32_t seq_id) const {
    return block_table_->get_num_blocks(seq_id);
}

int32_t KVCacheManager::get_sequence_capacity(int32_t seq_id) const {
    int32_t num_blocks = block_table_->get_num_blocks(seq_id);
    if (num_blocks < 0) return 0;
    return num_blocks * block_size_;
}

bool KVCacheManager::is_sequence_allocated(int32_t seq_id) const {
    return block_table_->has_sequence(seq_id);
}

int32_t KVCacheManager::get_free_block_num() const {
    return block_manager_->get_free_block_num();
}

int32_t KVCacheManager::get_total_block_num() const {
    return block_manager_->get_total_block_num();
}

float KVCacheManager::get_utilization() const {
    return block_manager_->get_utilization();
}

void KVCacheManager::reset() {
    std::vector<int32_t> seq_ids = block_table_->get_sequence_ids();
    for (int32_t seq_id : seq_ids) {
        free_sequence(seq_id);
    }

    seq_num_tokens_.clear();
    cached_sequences_.clear();
    block_manager_->reset();
    block_table_->reset();

    LOG(INFO) << "KVCacheManager reset complete";
}

int32_t KVCacheManager::get_max_blocks_per_seq() const {
    // 假设最大序列长度，或者根据实际运行时的 max_seq_len 配置
    const int32_t max_seq_len = 4096;
    return (max_seq_len + block_size_ - 1) / block_size_;
}

// ========== Prefix Caching 实现 ==========

/**
 * @brief 计算 Token Block 的链式哈希
 *
 * 使用 boost::hash_combine 风格的混合函数，通过 prev_hash 实现前缀链接：
 *   hash(block_n) = f(hash(block_{n-1}), tokens_n)
 * 相同 token 序列但不同前缀会产生不同的 hash。
 */
uint64_t KVCacheManager::compute_block_hash(uint64_t prev_hash, const int32_t* tokens,
                                            int32_t count) {
    uint64_t hash = prev_hash;
    for (int32_t i = 0; i < count; ++i) {
        // boost::hash_combine 风格混合
        hash ^= static_cast<uint64_t>(static_cast<uint32_t>(tokens[i])) + 0x9e3779b97f4a7c15ULL +
                (hash << 12) + (hash >> 4);
    }
    return hash;
}

/**
 * @brief 带 Prefix Caching 的序列分配
 *
 * 对 Prompt Tokens 按 block_size 分块计算链式哈希：
 *   1. 完整块（Full Block）：通过 hash 尝试复用已缓存的 Block
 *   2. 最后不完整块（Partial Block）：始终新分配（不可缓存）
 *   3. 连续命中的前缀块 → num_cached_tokens = 连续命中数 × block_size
 *
 * 重要：只有从开头连续命中的块才可跳过 Prefill。
 */
base::Status KVCacheManager::allocate_sequence_cached(int32_t seq_id,
                                                      const std::vector<int32_t>& tokens,
                                                      int32_t& num_cached_tokens) {
    if (!initialized_) {
        return base::error::InvalidArgument("KVCacheManager is not initialized");
    }
    if (block_table_->has_sequence(seq_id)) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " already allocated");
    }

    int32_t num_tokens = static_cast<int32_t>(tokens.size());
    int32_t num_full_blocks = num_tokens / block_size_;
    bool has_partial = (num_tokens % block_size_) > 0;
    int32_t total_blocks_needed = num_full_blocks + (has_partial ? 1 : 0);

    // 检查总可用块数（空闲 + 可驱逐）
    if (block_manager_->get_available_block_num() < total_blocks_needed) {
        return base::error::InternalError(
            "Insufficient blocks for prefix-cached allocation. Need: " +
            std::to_string(total_blocks_needed) +
            ", Available: " + std::to_string(block_manager_->get_available_block_num()));
    }

    std::vector<int32_t> block_ids;
    block_ids.reserve(total_blocks_needed);
    num_cached_tokens = 0;
    int32_t consecutive_cached_blocks = 0;
    bool prefix_broken = false;

    // 分配完整块（尝试缓存命中）
    uint64_t prev_hash = 0;
    for (int32_t i = 0; i < num_full_blocks; ++i) {
        const int32_t* block_tokens = tokens.data() + i * block_size_;
        uint64_t block_hash = compute_block_hash(prev_hash, block_tokens, block_size_);

        int32_t block_id;
        bool cache_hit;
        base::Status status = block_manager_->allocate_cached(block_hash, block_id, cache_hit);
        if (!status) {
            // 回滚已分配的块
            for (int32_t bid : block_ids) {
                block_manager_->release(bid);
            }
            return status;
        }
        block_ids.push_back(block_id);

        if (cache_hit && !prefix_broken) {
            consecutive_cached_blocks++;
        } else {
            prefix_broken = true;
        }

        prev_hash = block_hash;
    }

    // 分配最后不完整块（普通分配，不缓存）
    if (has_partial) {
        int32_t block_id;
        base::Status status = block_manager_->allocate(block_id);
        if (!status) {
            for (int32_t bid : block_ids) {
                block_manager_->release(bid);
            }
            return status;
        }
        block_ids.push_back(block_id);
    }

    // 注册到 BlockTable
    base::Status status = block_table_->allocate_sequence(seq_id, block_ids);
    if (!status) {
        for (int32_t bid : block_ids) {
            block_manager_->release(bid);
        }
        return status;
    }

    // 只有从开头连续命中的块才可跳过 Prefill
    num_cached_tokens = consecutive_cached_blocks * block_size_;

    seq_num_tokens_[seq_id] = num_tokens;
    cached_sequences_.insert(seq_id);

    VLOG(1) << "Allocated cached sequence " << seq_id << ": " << num_tokens << " tokens, "
            << total_blocks_needed << " blocks (" << consecutive_cached_blocks << " cached, "
            << num_cached_tokens << " tokens skippable)";

    return base::error::Success();
}

int64_t KVCacheManager::get_prefix_cache_hits() const {
    return block_manager_->get_cache_hits();
}

int64_t KVCacheManager::get_prefix_cache_misses() const {
    return block_manager_->get_cache_misses();
}

}  // namespace engine
