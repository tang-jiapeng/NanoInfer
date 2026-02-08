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

base::Status KVCacheManager::init(std::shared_ptr<base::DeviceAllocator> allocator) {
    if (initialized_) {
        return base::error::InvalidArgument("KVCacheManager is already initialized");
    }
    if (!allocator) {
        return base::error::InvalidArgument("Allocator cannot be null");
    }

    // 计算总显存占用
    // Shape: [num_blocks, num_kv_heads, block_size, head_size]
    size_t elements_per_block =
        static_cast<size_t>(num_kv_heads_) * block_size_ * head_size_;
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
            return base::error::InternalError(
                "Failed to allocate Value Cache for layer " + std::to_string(i));
        }
        value_caches_.push_back(std::move(v_cache));
    }

    initialized_ = true;
    LOG(INFO) << "KV cache initialization complete";
    return base::error::Success();
}

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
        base::Status status =
            block_manager_->allocate(additional_blocks_needed, new_block_ids);
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
    // 归还物理 Block
    status = block_manager_->free(block_ids);
    if (!status) {
        LOG(ERROR) << "Failed to free blocks for sequence " << seq_id << ": "
                   << status.get_err_msg();
        return status;
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
    block_manager_->reset();
    block_table_->reset();

    LOG(INFO) << "KVCacheManager reset complete";
}

int32_t KVCacheManager::get_max_blocks_per_seq() const {
    // 假设最大序列长度，或者根据实际运行时的 max_seq_len 配置
    const int32_t max_seq_len = 4096;
    return (max_seq_len + block_size_ - 1) / block_size_;
}

}  // namespace engine
