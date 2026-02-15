#include "nanoinfer/engine/inference_request.h"

namespace engine {
InferenceRequest::InferenceRequest(int64_t request_id, std::string prompt,
                                   std::vector<int32_t> prompt_tokens, int32_t max_new_tokens)
    : request_id_(request_id),
      prompt_(std::move(prompt)),
      prompt_tokens_(std::move(prompt_tokens)),
      max_new_tokens_(max_new_tokens),
      state_(RequestState::kWaiting),
      num_computed_tokens_(0),
      arrival_time_(std::chrono::high_resolution_clock::now()) {
}

void InferenceRequest::set_state(RequestState state) {
    state_ = state;
}

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

bool InferenceRequest::add_token(int32_t token, int32_t eos_token_id) {
    generated_tokens_.push_back(token);

    // 检查停止条件:
    // 生成了结束符 (EOS)
    // 达到了最大生成长度限制
    if (token == eos_token_id || generated_len() >= max_new_tokens_) {
        finish();
        return false;
    }
    return true;
}

void InferenceRequest::add_computed_tokens(int32_t count) {
    num_computed_tokens_ += count;
}

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