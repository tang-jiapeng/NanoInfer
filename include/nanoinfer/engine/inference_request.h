/**
 * @file inference_request.h
 * @brief 推理请求：Token 级状态跟踪，支持 Continuous Batching / Chunked Prefill
 */
#ifndef NANO_INFER_INFERENCE_REQUEST_H
#define NANO_INFER_INFERENCE_REQUEST_H

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/sampler/sampling_params.h"

namespace engine {

/// @brief 请求状态：kWaiting → kRunning → kFinished，可被 Preempt
enum class RequestState {
    kWaiting,
    kRunning,
    kFinished,
    kPreempted,
};

/**
 * @brief 推理请求 (Inference Request)
 *
 * 追踪单请求的完整上下文：num_computed_tokens 控制 KV Cache 偏移与 Resume，
 * generated_tokens 记录已生成序列。
 */
class InferenceRequest {
   public:
    InferenceRequest(int64_t request_id, std::string prompt, std::vector<int32_t> prompt_tokens,
                     int32_t max_new_tokens,
                     sampler::SamplingParams sampling_params = sampler::SamplingParams());

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

    /// @brief 已计算 Token 总数（KV Cache 偏移 / Resume 的关键索引）
    int32_t num_computed_tokens() const {
        return num_computed_tokens_;
    }

    int32_t prompt_len() const {
        return static_cast<int32_t>(prompt_tokens_.size());
    }
    int32_t generated_len() const {
        return static_cast<int32_t>(generated_tokens_.size());
    }

    /// @brief Prompt + Generated 总 Token 数
    int32_t total_len() const {
        return prompt_len() + generated_len();
    }

    bool is_finished() const {
        return state_ == RequestState::kFinished;
    }

    /// @brief 是否处于 Prefill 阶段（computed < prompt_len）
    bool is_prefill() const {
        return num_computed_tokens_ < prompt_len();
    }

    /// @brief 是否处于 Decode 阶段
    bool is_decode() const {
        return num_computed_tokens_ >= prompt_len() && !is_finished();
    }

    /// @brief Prefill 阶段剩余未计算 Token 数
    int32_t prefill_remaining() const {
        return std::max(0, prompt_len() - num_computed_tokens_);
    }

    void set_state(RequestState state);

    /// @brief 标记开始运行（首次调用记录 start_time）
    void start_running();

    void finish();

    void preempt();

    void resume();

    /**
     * @brief 追加生成 Token
     * @param token 新 Token ID
     * @param eos_token_id EOS Token ID
     * @return false 表示触发停止条件（EOS 或达到 max_new_tokens）
     */
    bool add_token(int32_t token, int32_t eos_token_id, int32_t eos_token_id2 = -1);

    /// @brief 累加已计算 Token 计数（Chunked Prefill 每完成一 Chunk 调用）
    void add_computed_tokens(int32_t count);

    /// @brief 设置已计算 Token 数（用于 Prefix Caching 跳过已缓存前缀）
    void set_num_computed_tokens(int32_t count) {
        num_computed_tokens_ = count;
    }

    /// @brief 返回下一个待处理 Token
    int32_t next_token() const;

    /**
     * @brief 获取下一 Chunk 的 Token 列表
     *
     * Prefill：返回 prompt 中接下来 chunk_size 个 token；
     * Decode：返回上一个生成 token（size=1）。
     */
    std::vector<int32_t> get_next_chunk_tokens(int32_t chunk_size = 512) const;

    /// @brief 获取下一 Chunk 的 Position ID 列表（用于 RoPE）
    std::vector<int32_t> get_next_chunk_positions(int32_t chunk_size = 512) const;

    /// @brief 获取采样参数
    const sampler::SamplingParams& sampling_params() const {
        return sampling_params_;
    }

    /// @brief 设置采样参数
    void set_sampling_params(const sampler::SamplingParams& params) {
        sampling_params_ = params;
    }

    double latency_seconds() const;

    double execution_time_seconds() const;

    std::chrono::high_resolution_clock::time_point arrival_time() const {
        return arrival_time_;
    }

   private:
    int64_t request_id_;
    std::string prompt_;
    std::vector<int32_t> prompt_tokens_;
    int32_t max_new_tokens_;

    RequestState state_;
    int32_t num_computed_tokens_;

    std::vector<int32_t> generated_tokens_;
    sampler::SamplingParams sampling_params_;

    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point finish_time_;
    std::chrono::high_resolution_clock::time_point arrival_time_;
};

using InferenceRequestPtr = std::shared_ptr<InferenceRequest>;

}  // namespace engine

#endif