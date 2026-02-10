#ifndef NANO_INFER_SCHEDULER_H
#define NANO_INFER_SCHEDULER_H

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/inference_request.h"

namespace engine {

/**
 * @brief 调度策略
 */
enum class SchedulingPolicy {
    kFCFS,      // 先来先服务 (First-Come-First-Serve)
    kPriority,  // 基于优先级 (预留，暂未实现)
};

/**
 * @brief 调度输出 (本轮 Step 需要执行的请求集合)
 */
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
 * 核心职责：
 * 1. 管理所有进入系统的推理请求 (Waiting Queue + Running List)
 * 2. 决定每一轮 Step 运行哪些请求 (Scheduling Strategy)
 * 3. 实现 Continuous Batching 和 Chunked Prefill 策略
 *
 * 调度逻辑 (参考 vLLM):
 * Phase 1: 必须继续运行当前处于 RUNNING 状态的请求 (保证 Decode 连续性)
 * Phase 2: 如果 Batch Size 还有空余，从 WAITING 队列中取出请求加入 Batch
 */
class Scheduler {
   public:
    /**
     * @brief 构造函数
     *
     * @param max_batch_size 最大并发 Batch Size (限制单次 forward 的序列数)
     * @param max_sequences 系统允许的最大并发序列数 (Running + Waiting 的软限制)
     * @param chunk_size Prefill 阶段的分块大小 (默认 256)
     * @param policy 调度策略
     */
    Scheduler(int32_t max_batch_size, int32_t max_sequences, int32_t chunk_size = 256,
              SchedulingPolicy policy = SchedulingPolicy::kFCFS);

    ~Scheduler() = default;

    /**
     * @brief 添加一个新的推理请求
     *
     * @param prompt 输入 Prompt
     * @param prompt_tokens Prompt Token ID 列表
     * @param max_new_tokens 最大生成长度
     * @return int64_t 生成的 Request ID
     */
    int64_t add_request(const std::string& prompt,
                        const std::vector<int32_t>& prompt_tokens,
                        int32_t max_new_tokens);

    /**
     * @brief 调度下一个 Batch
     *
     * 根据当前状态和配置，生成下一个 Step 需要执行的请求列表。
     *
     * @return ScheduledBatch 包含本轮要执行的请求
     */
    ScheduledBatch schedule_next_batch();

    /**
     * @brief 在 Batch 执行完一步后更新状态
     *
     * @param finished_request_ids 本轮执行完成后结束的请求 ID 列表
     */
    void update_after_step(const std::vector<int64_t>& finished_request_ids);

    /**
     * @brief 获取指定 ID 的请求
     */
    InferenceRequestPtr get_request(int64_t request_id) const;

    /**
     * @brief 检查是否还有任务需要处理 (Waiting 或 Running 不为空)
     */
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

    /**
     * @brief 统计信息结构体
     */
    struct Stats {
        int32_t num_running;
        int32_t num_waiting;
        int32_t num_finished;
        int32_t total_requests;
    };

    Stats get_stats() const;

    /**
     * @brief 清理所有已完成的请求
     */
    void clear_finished_requests();

   private:
    int32_t max_batch_size_;
    int32_t max_sequences_;
    int32_t chunk_size_;
    SchedulingPolicy policy_;
    int64_t next_seq_id_;

    // 请求队列
    // 等待队列: 尚未开始执行的请求
    std::deque<InferenceRequestPtr> waiting_queue_;
    // 运行列表: 正在执行中的请求
    std::vector<InferenceRequestPtr> running_requests_;

    // 全局请求映射表 (用于 ID 查找)
    std::unordered_map<int64_t, InferenceRequestPtr> request_map_;
};

}  // namespace engine

#endif