#ifndef NANO_INFER_INFERENCE_REQUEST_H
#define NANO_INFER_INFERENCE_REQUEST_H

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"

namespace engine {

/**
 * @brief 请求状态机
 *
 * 状态流转:
 * kWaiting -> kRunning -> kFinished
 * kRunning -> kPreempted -> kWaiting -> kRunning
 */
enum class RequestState {
    kWaiting,    // 在队列中等待调度
    kRunning,    // 正在运行 (Prefill 或 Decode)
    kFinished,   // 生成完成
    kPreempted,  // 被抢占 (暂时挂起以释放资源)
};

/**
 * @brief 推理请求对象 (Inference Request)
 *
 * 记录单个请求的所有上下文信息，支持 Token 级调度 (Continuous Batching) 和分块预填充
 * (Chunked Prefill)。
 * * 核心追踪状态:
 * - num_computed_tokens: 已完成计算的 token 数量 (用于 KV Cache 索引和 Resume)
 * - generated_tokens: 已生成的 token 列表
 */
class InferenceRequest {
   public:
    InferenceRequest(int64_t request_id, std::string prompt,
                     std::vector<int32_t> prompt_tokens, int32_t max_new_tokens);

    ~InferenceRequest() = default;

    int64_t request_id() const {
        return request_id_;
    }
    const std::string& prompt() const {
        return prompt_;
    }
    const std::vector<int32_t>& prompt_tokens() const {
        return prompt_tokens_;
    }
    const std::vector<int32_t>& generated_tokens() const {
        return generated_tokens_;
    }
    int32_t max_new_tokens() const {
        return max_new_tokens_;
    }
    RequestState state() const {
        return state_;
    }

    /**
     * @brief 获取当前已计算的 token 总数
     * @note 这是 KV Cache 物理位置索引和 Resume 调度的关键
     */
    int32_t num_computed_tokens() const {
        return num_computed_tokens_;
    }

    int32_t prompt_len() const {
        return static_cast<int32_t>(prompt_tokens_.size());
    }
    int32_t generated_len() const {
        return static_cast<int32_t>(generated_tokens_.size());
    }

    /**
     * @brief 获取总 Token 数 (Prompt + Generated)
     */
    int32_t total_len() const {
        return prompt_len() + generated_len();
    }

    bool is_finished() const {
        return state_ == RequestState::kFinished;
    }

    /**
     * @brief 判断当前是否处于 Prefill (预填充) 阶段
     * @return true 如果已计算 token 数 < Prompt 长度
     */
    bool is_prefill() const {
        return num_computed_tokens_ < prompt_len();
    }

    /**
     * @brief 判断当前是否处于 Decode (解码) 阶段
     */
    bool is_decode() const {
        return num_computed_tokens_ >= prompt_len() && !is_finished();
    }

    /**
     * @brief 获取 Prefill 阶段剩余未计算的 Token 数
     */
    int32_t prefill_remaining() const {
        return std::max(0, prompt_len() - num_computed_tokens_);
    }

    /**
     * @brief 设置状态
     */
    void set_state(RequestState state);

    /**
     * @brief 标记请求开始运行
     * 如果是首次运行，会记录 start_time
     */
    void start_running();

    /**
     * @brief 标记请求完成
     */
    void finish();

    /**
     * @brief 抢占请求 (暂停)
     */
    void preempt();

    /**
     * @brief 恢复请求 (从抢占状态)
     */
    void resume();

    /**
     * @brief 添加一个新生成的 Token
     * * @param token 新生成的 Token ID
     * @param eos_token_id 结束符 ID (用于判断是否生成结束)
     * @return true 继续生成
     * @return false 触发停止条件 (EOS 或 达到 max_new_tokens)，请求已结束
     */
    bool add_token(int32_t token, int32_t eos_token_id);

    /**
     * @brief 增加已计算 Token 计数
     * 通常在 Prefill 阶段完成一个 Chunk 计算后调用
     */
    void add_computed_tokens(int32_t count);

    /**
     * @brief 获取下一个要处理的 Token (用于 Decode 或 单步调试)
     */
    int32_t next_token() const;

    /**
     * @brief 获取下一个 Chunk 的 Token 列表 (核心调度接口)
     * * 用于支持 Chunked Prefill：
     * - Prefill 阶段: 返回 prompt 中接下来的 chunk_size 个 token
     * - Decode 阶段: 返回上一个生成的 token (size=1)
     * * @param chunk_size 最大 Chunk 大小
     */
    std::vector<int32_t> get_next_chunk_tokens(int32_t chunk_size = 256) const;

    /**
     * @brief 获取下一个 Chunk 对应的 Position ID 列表
     * 用于 RoPE 计算
     */
    std::vector<int32_t> get_next_chunk_positions(int32_t chunk_size = 256) const;

    /**
     * @brief 获取从到达开始的延迟 (秒)
     */
    double latency_seconds() const;

    /**
     * @brief 获取实际执行时间 (秒)
     */
    double execution_time_seconds() const;

    /**
     * @brief 获取到达时间点
     */
    std::chrono::high_resolution_clock::time_point arrival_time() const {
        return arrival_time_;
    }

   private:
    int64_t request_id_;
    std::string prompt_;
    std::vector<int32_t> prompt_tokens_;
    int32_t max_new_tokens_;

    RequestState state_;

    // 核心状态: 记录当前已处理到第几个 token。
    // 在 Prefill 阶段，它指向 Prompt 中的进度。
    // 在 Decode 阶段，它等于 prompt_len + generated_len
    int32_t num_computed_tokens_;

    std::vector<int32_t> generated_tokens_;

    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point finish_time_;
    std::chrono::high_resolution_clock::time_point arrival_time_;
};

using InferenceRequestPtr = std::shared_ptr<InferenceRequest>;

}  // namespace engine

#endif