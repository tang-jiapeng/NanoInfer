/**
 * @file inference_request.cpp
 * @brief 推理请求生命周期管理
 *
 * InferenceRequest 跟踪一条推理请求从提交到完成的全过程：
 *   - 状态机：Waiting → Running → Finished（可被 Preempt 再 Resume）
 *   - Prefill 支持：get_next_chunk_tokens/positions 实现 Chunked Prefill
 *   - Decode 支持：add_token 逐 Token 生成，检查 EOS / 最大长度停止条件
 *   - 延迟统计：latency_seconds / execution_time_seconds
 */
#include "nanoinfer/engine/inference_request.h"

namespace engine {
InferenceRequest::InferenceRequest(int64_t request_id, std::string prompt,
                                   std::vector<int32_t> prompt_tokens, int32_t max_new_tokens,
                                   sampler::SamplingParams sampling_params)
    : request_id_(request_id),
      prompt_(std::move(prompt)),
      prompt_tokens_(std::move(prompt_tokens)),
      max_new_tokens_(max_new_tokens),
      state_(RequestState::kWaiting),
      num_computed_tokens_(0),
      sampling_params_(std::move(sampling_params)),
      arrival_time_(std::chrono::high_resolution_clock::now()) {
}

void InferenceRequest::set_state(RequestState state) {
    state_ = state;
}

/** @brief 开始执行，状态 Waiting → Running，首次调用时记录 start_time */
void InferenceRequest::start_running() {
    state_ = RequestState::kRunning;
    // 如果是第一次运行，记录开始时间
    if (start_time_.time_since_epoch().count() == 0) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
}

void InferenceRequest::finish() {
    state_ = RequestState::kFinished;
    finish_time_ = std::chrono::high_resolution_clock::now();
}

void InferenceRequest::preempt() {
    state_ = RequestState::kPreempted;
}

void InferenceRequest::resume() {
    state_ = RequestState::kWaiting;
}

/**
 * @brief 添加生成的 Token 并检查停止条件
 *
 * 支持两个停止 token（LLaMA3 有 <|end_of_text|>=128001 和 <|eot_id|>=128009）。
 * eos_token_id2 默认为 -1（即不启用第二停止符）。
 *
 * @return true = 继续生成，false = 已完成（遇到任一 EOS 或达到最大长度）
 */
bool InferenceRequest::add_token(int32_t token, int32_t eos_token_id, int32_t eos_token_id2) {
    generated_tokens_.push_back(token);

    // 检查停止条件:
    // 生成了结束符 (EOS / EOT)
    // 达到了最大生成长度限制
    bool is_eos = (token == eos_token_id) || (eos_token_id2 != -1 && token == eos_token_id2);
    if (is_eos || generated_len() >= max_new_tokens_) {
        finish();
        return false;
    }
    return true;
}

void InferenceRequest::add_computed_tokens(int32_t count) {
    num_computed_tokens_ += count;
}

/**
 * @brief 获取下一步要处理的 Token
 *
 * Prefill 阶段：返回 Prompt 中下一个未计算的 Token。
 * Decode 阶段：返回上一步生成的 Token（自回归）。
 */
int32_t InferenceRequest::next_token() const {
    if (is_prefill()) {
        // Prefill 阶段：下一个要计算的是 Prompt 中的 token
        return prompt_tokens_[num_computed_tokens_];
    } else {
        // Decode 阶段：下一个要计算的是上一步生成的 token
        if (!generated_tokens_.empty()) {
            return generated_tokens_.back();
        } else {
            // 刚从 Prefill 转入 Decode 的第一步 (First Decode Step)
            // 此时 generated_tokens 为空，输入应为 Prompt 的最后一个 token
            return prompt_tokens_.back();
        }
    }
}

/**
 * @brief 获取下一个 Chunk 的 Token 序列（支持 Chunked Prefill）
 *
 * Prefill: 返回 min(chunk_size, 剩余未计算) 个 Token。
 * Decode: 返回 1 个 Token（自回归）。
 */
std::vector<int32_t> InferenceRequest::get_next_chunk_tokens(int32_t chunk_size) const {
    std::vector<int32_t> tokens;

    if (is_prefill()) {
        // Prefill 阶段：支持分块处理
        // 取 min(chunk_size, 剩余未计算部分)
        int32_t remaining = prefill_remaining();
        int32_t current_chunk = std::min(chunk_size, remaining);

        int32_t start = num_computed_tokens_;
        int32_t end = start + current_chunk;

        tokens.reserve(current_chunk);
        for (int32_t i = start; i < end; ++i) {
            tokens.push_back(prompt_tokens_[i]);
        }
    } else {
        // Decode 阶段：每次只处理 1 个 token (自回归)
        tokens.push_back(next_token());
    }

    return tokens;
}

/**
 * @brief 获取下一个 Chunk 的绝对位置索引（用于 RoPE）
 *
 * 位置 = num_computed_tokens + 偏移量，保证多次 Chunk 的位置连续。
 */
std::vector<int32_t> InferenceRequest::get_next_chunk_positions(int32_t chunk_size) const {
    std::vector<int32_t> positions;

    int32_t count = 0;
    if (is_prefill()) {
        count = std::min(chunk_size, prefill_remaining());
    } else {
        count = 1;
    }

    positions.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        // 绝对位置编码 = 当前已计算总数 + 偏移量
        // 例如 Prompt 长 10，已算 5，此次取 chunk=2
        // 则 positions 为 [5, 6]
        positions.push_back(num_computed_tokens_ + i);
    }

    return positions;
}

double InferenceRequest::latency_seconds() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - arrival_time_).count();
}

double InferenceRequest::execution_time_seconds() const {
    if (start_time_.time_since_epoch().count() == 0) return 0.0;
    auto end = (finish_time_.time_since_epoch().count() > 0)
                   ? finish_time_
                   : std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start_time_).count();
}

}  // namespace engine