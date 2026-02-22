/**
 * @file scheduler.cpp
 * @brief 请求调度器实现（FCFS 策略 + Continuous Batching）
 *
 * Scheduler 管理请求的等待队列和运行列表：
 *   - add_request()：加入 waiting_queue_
 *   - schedule_next_batch()：
 *     Phase 1: 所有 Running 请求必须参与（Decode 步进）
 *     Phase 2: 从 Waiting 队列取新请求填充 Batch（受 max_batch_size / max_sequences 约束）
 *   - update_after_step()：移除已完成的请求（erase-remove 惯用法）
 */
#include "nanoinfer/engine/scheduler.h"

namespace engine {

Scheduler::Scheduler(int32_t max_batch_size, int32_t max_sequences, int32_t chunk_size,
                     SchedulingPolicy policy)
    : max_batch_size_(max_batch_size),
      max_sequences_(max_sequences),
      chunk_size_(chunk_size),
      policy_(policy),
      next_seq_id_(0) {
    LOG(INFO) << "Scheduler initialized: max_batch_size=" << max_batch_size
              << ", max_running_sequences=" << max_sequences
              << ", prefill_chunk_size=" << chunk_size;
}

/** @brief 将新推理请求加入等待队列，返回分配的 request_id */
int64_t Scheduler::add_request(const std::string& prompt, const std::vector<int32_t>& prompt_tokens,
                               int32_t max_new_tokens,
                               const sampler::SamplingParams& sampling_params) {
    int64_t request_id = next_seq_id_++;
    auto request = std::make_shared<InferenceRequest>(request_id, prompt, prompt_tokens,
                                                      max_new_tokens, sampling_params);

    // 加入等待队列
    waiting_queue_.push_back(request);
    request_map_[request_id] = request;

    VLOG(2) << "Added request " << request_id << " to waiting queue (len=" << prompt_tokens.size()
            << ")";
    return request_id;
}

/**
 * @brief 调度下一个 Batch（两阶段策略）
 *
 * Phase 1: 所有 Running 请求必须参与（保持 KV Cache 连续性）。
 * Phase 2: 从 Waiting 队列按 FCFS 填充剩余 Batch 槽位。
 */
ScheduledBatch Scheduler::schedule_next_batch() {
    ScheduledBatch batch;

    // Phase 1: 必须调度所有正在运行 (Running) 的请求
    // 这些请求已经持有 KV Cache 资源，必须在每一轮都参与计算 (Decode
    // 步进)，否则会导致生成中断。
    for (const auto& req : running_requests_) {
        batch.requests.push_back(req);
    }

    // Phase 2: 如果有空余资源，调度等待 (Waiting) 队列中的请求
    // 限制条件:
    // 1. Batch Size 上限: 防止单次 forward 计算量过大
    // 2. 并发序列数上限 (max_sequences): 防止同时也管理太多序列导致开销过大
    //    注意：物理显存限制由 BlockManager 在外部控制 (Allocation 失败会阻止调度)，
    //    这里只是调度策略上的软限制。
    int32_t remaining_batch_slots = max_batch_size_ - batch.size();
    int32_t current_running_count = static_cast<int32_t>(running_requests_.size());
    int32_t available_seq_slots = max_sequences_ - current_running_count;

    while (remaining_batch_slots > 0 && available_seq_slots > 0 && !waiting_queue_.empty()) {
        // FCFS 策略: 取队首
        auto req = waiting_queue_.front();
        waiting_queue_.pop_front();

        // 状态变迁: Waiting -> Running
        req->start_running();

        // 加入运行列表和当前 Batch
        running_requests_.push_back(req);
        batch.requests.push_back(req);

        // 更新计数器
        remaining_batch_slots--;
        available_seq_slots--;

        VLOG(2) << "Scheduled new request " << req->request_id() << " for execution";
    }

    return batch;
}

/** @brief 移除已完成的请求（erase-remove 惯用法 + finish 状态标记） */
void Scheduler::update_after_step(const std::vector<int64_t>& finished_request_ids) {
    if (finished_request_ids.empty()) {
        return;
    }

    // 从 running_requests_ 中移除已完成的请求
    // 使用 erase-remove 惯用语，但在 remove 谓词中包含副作用 (调用 finish)
    auto it = std::remove_if(running_requests_.begin(), running_requests_.end(),
                             [&](const InferenceRequestPtr& req) {
                                 // 检查该请求是否在完成列表中
                                 for (int64_t fid : finished_request_ids) {
                                     if (req->request_id() == fid) {
                                         // 标记请求完成
                                         req->finish();
                                         VLOG(2) << "Request " << fid << " finished execution";
                                         return true;  // 标记为移除
                                     }
                                 }
                                 return false;
                             });

    // 删除元素
    if (it != running_requests_.end()) {
        running_requests_.erase(it, running_requests_.end());
    }
}

InferenceRequestPtr Scheduler::get_request(int64_t request_id) const {
    auto it = request_map_.find(request_id);
    if (it != request_map_.end()) {
        return it->second;
    }
    return nullptr;
}

bool Scheduler::has_work() const {
    // 只要有正在跑的，或者排队等着的，就算有工作
    return !running_requests_.empty() || !waiting_queue_.empty();
}

Scheduler::Stats Scheduler::get_stats() const {
    Stats stats;
    stats.num_running = num_running();
    stats.num_waiting = num_waiting();

    stats.num_finished = 0;
    for (const auto& kv : request_map_) {
        if (kv.second->is_finished()) {
            stats.num_finished++;
        }
    }

    stats.total_requests = static_cast<int32_t>(request_map_.size());
    return stats;
}

void Scheduler::clear_finished_requests() {
    auto it = request_map_.begin();
    while (it != request_map_.end()) {
        if (it->second->is_finished()) {
            it = request_map_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace engine