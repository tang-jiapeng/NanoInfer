/**
 * @file scheduler.h
 * @brief 请求调度器：Continuous Batching + Chunked Prefill
 */
#ifndef NANO_INFER_SCHEDULER_H
#define NANO_INFER_SCHEDULER_H

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/inference_request.h"

namespace engine {

/// @brief 调度策略
enum class SchedulingPolicy {
    kFCFS,
    kPriority,
};

/// @brief 单轮 Step 的调度结果
struct ScheduledBatch {
    std::vector<InferenceRequestPtr> requests;

    int32_t size() const {
        return static_cast<int32_t>(requests.size());
    }

    bool empty() const {
        return requests.empty();
    }

    void clear() {
        requests.clear();
    }
};

/**
 * @brief 调度器 (Scheduler)
 *
 * 管理 Waiting Queue + Running List，每轮 Step 决定执行哪些请求。
 * Phase 1: 保留所有 Running 请求（保证 Decode 连续性）；
 * Phase 2: 从 Waiting 队列补充至 max_batch_size。
 */
class Scheduler {
   public:
    Scheduler(int32_t max_batch_size, int32_t max_sequences, int32_t chunk_size = 512,
              SchedulingPolicy policy = SchedulingPolicy::kFCFS);

    ~Scheduler() = default;

    /// @brief 提交新请求，返回 Request ID
    int64_t add_request(const std::string& prompt, const std::vector<int32_t>& prompt_tokens,
                        int32_t max_new_tokens,
                        const sampler::SamplingParams& sampling_params = sampler::SamplingParams());

    /// @brief 调度下一个 Batch
    ScheduledBatch schedule_next_batch();

    /// @brief Step 完成后更新状态（移除已结束请求）
    void update_after_step(const std::vector<int64_t>& finished_request_ids);

    InferenceRequestPtr get_request(int64_t request_id) const;

    bool has_work() const;

    int32_t num_active_sequences() const {
        return static_cast<int32_t>(running_requests_.size() + waiting_queue_.size());
    }

    int32_t num_running() const {
        return static_cast<int32_t>(running_requests_.size());
    }

    int32_t num_waiting() const {
        return static_cast<int32_t>(waiting_queue_.size());
    }

    int32_t chunk_size() const {
        return chunk_size_;
    }

    void set_chunk_size(int32_t chunk_size) {
        chunk_size_ = chunk_size;
    }

    /// @brief 统计信息
    struct Stats {
        int32_t num_running;
        int32_t num_waiting;
        int32_t num_finished;
        int32_t total_requests;
    };

    Stats get_stats() const;

    void clear_finished_requests();

   private:
    int32_t max_batch_size_;
    int32_t max_sequences_;
    int32_t chunk_size_;
    SchedulingPolicy policy_;
    int64_t next_seq_id_;

    std::deque<InferenceRequestPtr> waiting_queue_;
    std::vector<InferenceRequestPtr> running_requests_;
    std::unordered_map<int64_t, InferenceRequestPtr> request_map_;
};

}  // namespace engine

#endif