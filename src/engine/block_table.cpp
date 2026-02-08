#include "nanoinfer/engine/block_table.h"
#include <glog/logging.h>

namespace engine {

BlockTable::BlockTable(bool thread_safe) : thread_safe_(thread_safe) {
    VLOG(1) << "BlockTable initialized (thread-safe: " << (thread_safe ? "yes" : "no")
            << ")";
}

base::Status BlockTable::allocate_sequence(int32_t seq_id,
                                           const std::vector<int32_t>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    if (seq_to_blocks_.find(seq_id) != seq_to_blocks_.end()) {
        return base::error::InvalidArgument("Sequence ID " + std::to_string(seq_id) +
                                            " already exists");
    }

    seq_to_blocks_[seq_id] = block_ids;

    VLOG(2) << "Allocated sequence " << seq_id << " with " << block_ids.size()
            << " blocks";

    if (VLOG_IS_ON(2)) {
        std::string ids_str = "[";
        for (size_t i = 0; i < block_ids.size(); ++i) {
            if (i > 0) ids_str += ", ";
            ids_str += std::to_string(block_ids[i]);
        }
        ids_str += "]";
        VLOG(2) << ids_str;
    }

    return base::error::Success();
}

base::Status BlockTable::append_block(int32_t seq_id, int32_t block_id) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }

    it->second.push_back(block_id);

    VLOG(2) << "Appended block " << block_id << " to sequence " << seq_id << " (now "
            << it->second.size() << " blocks)";

    return base::error::Success();
}

base::Status BlockTable::append_blocks(int32_t seq_id,
                                       const std::vector<int32_t>& block_ids) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }

    it->second.insert(it->second.end(), block_ids.begin(), block_ids.end());

    VLOG(2) << "Appended " << block_ids.size() << " blocks to sequence " << seq_id
            << " (now " << it->second.size() << " blocks)";

    return base::error::Success();
}

base::Status BlockTable::get_blocks(int32_t seq_id,
                                    std::vector<int32_t>& block_ids) const {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }

    block_ids = it->second;
    return base::error::Success();
}

int32_t BlockTable::get_num_blocks(int32_t seq_id) const {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        LOG(WARNING) << "Sequence " << seq_id << " not found when querying num_blocks";
        return -1;
    }

    return static_cast<int32_t>(it->second.size());
}

base::Status BlockTable::free_sequence(int32_t seq_id,
                                       std::vector<int32_t>& freed_blocks) {
    LockGuard lock(mutex_, thread_safe_);

    auto it = seq_to_blocks_.find(seq_id);
    if (it == seq_to_blocks_.end()) {
        return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                            " does not exist");
    }
    // 移动 Block ID 列表以返回给调用者,以便释放物理显存
    freed_blocks = std::move(it->second);
    // 从 Map 中移除
    seq_to_blocks_.erase(it);

    VLOG(2) << "Freed sequence " << seq_id << " (" << freed_blocks.size()
            << " blocks returned)";

    return base::error::Success();
}

bool BlockTable::has_sequence(int32_t seq_id) const {
    LockGuard lock(mutex_, thread_safe_);
    return seq_to_blocks_.find(seq_id) != seq_to_blocks_.end();
}

std::vector<int32_t> BlockTable::get_sequence_ids() const {
    LockGuard lock(mutex_, thread_safe_);

    std::vector<int32_t> seq_ids;
    seq_ids.reserve(seq_to_blocks_.size());

    for (const auto& [seq_id, _] : seq_to_blocks_) {
        seq_ids.push_back(seq_id);
    }

    return seq_ids;
}

int32_t BlockTable::get_num_sequences() const {
    LockGuard lock(mutex_, thread_safe_);
    return static_cast<int32_t>(seq_to_blocks_.size());
}

base::Status BlockTable::to_gpu_format(const std::vector<int32_t>& seq_ids,
                                       int32_t max_blocks_per_seq,
                                       tensor::Tensor& tensor) const {
    LockGuard lock(mutex_, thread_safe_);

    if (seq_ids.empty()) {
        return base::error::InvalidArgument("seq_ids cannot be empty");
    }
    if (max_blocks_per_seq <= 0) {
        return base::error::InvalidArgument("max_blocks_per_seq must be positive");
    }

    int32_t num_seqs = static_cast<int32_t>(seq_ids.size());

    // 创建 Tensor
    // 使用 CPU Allocator
    // 维度: [num_seqs, max_blocks_per_seq]
    // 数据类型: Int32
    // need_alloc = true: 立即分配内存
    std::shared_ptr<base::DeviceAllocator> alloc =
        base::CPUDeviceAllocatorFactory::get_instance();
    tensor = tensor::Tensor(base::DataType::kDataTypeInt32, num_seqs, max_blocks_per_seq,
                            true, alloc);

    if (tensor.ptr<int32_t>() == nullptr) {
        return base::error::InternalError(
            "Failed to allocate memory for block table tensor");
    }
    // 获取数据指针并填充
    int32_t* data = tensor.ptr<int32_t>();

    // 初始化为 -1 (Padding Value)
    std::fill(data, data + num_seqs * max_blocks_per_seq, -1);

    // 填充每个序列的 Block ID
    for (int32_t i = 0; i < num_seqs; ++i) {
        int32_t seq_id = seq_ids[i];

        auto it = seq_to_blocks_.find(seq_id);
        if (it == seq_to_blocks_.end()) {
            return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                                " not found in block table");
        }

        const auto& blocks = it->second;
        int32_t num_blocks = static_cast<int32_t>(blocks.size());

        if (num_blocks > max_blocks_per_seq) {
            return base::error::InvalidArgument("Sequence " + std::to_string(seq_id) +
                                                " has " + std::to_string(num_blocks) +
                                                " blocks, exceeds max " +
                                                std::to_string(max_blocks_per_seq));
        }

        // 拷贝数据到平铺数组
        // data 是行优先存储 (Row-Major)，第 i 行起始位置为 data + i * max_blocks_per_seq
        int32_t* row_ptr = data + i * max_blocks_per_seq;
        for (int32_t j = 0; j < num_blocks; ++j) {
            row_ptr[j] = blocks[j];
        }
    }
    VLOG(2) << "Converted block table to GPU format: " << num_seqs << " seqs, "
            << max_blocks_per_seq << " max blocks per seq";

    // 注意：此函数生成的 tensor 目前仍在 CPU 上 (device_type = kDeviceCPU)
    // 调用者拿到这个 Tensor 后，需要调用 tensor.to_cuda(stream) 将其搬运到 GPU 供
    // Kernel使用

    return base::error::Success();
}

void BlockTable::reset() {
    LockGuard lock(mutex_, thread_safe_);

    int32_t num_seqs = static_cast<int32_t>(seq_to_blocks_.size());
    seq_to_blocks_.clear();

    LOG(INFO) << "BlockTable reset: " << num_seqs << " sequences cleared";
}

}  // namespace engine