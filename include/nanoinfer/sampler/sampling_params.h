/**
 * @file sampling_params.h
 * @brief 采样参数结构体（vLLM 风格 per-request 采样配置）
 *
 * 参照 vLLM SamplingParams 设计，支持 Temperature / Top-K / Top-P / Repetition Penalty
 * 等常见采样策略的参数化配置。每个推理请求可以携带独立的采样参数。
 *
 * 采样 Pipeline（按顺序执行）：
 *   1. Repetition Penalty — 对已生成 token 的 logits 施加惩罚
 *   2. Temperature Scaling — logits /= temperature
 *   3. Top-K Filtering    — 仅保留概率最高的 K 个 token
 *   4. Top-P Filtering    — 仅保留累积概率 ≤ p 的最小 token 集合
 *   5. Softmax            — 将 logits 转为概率分布
 *   6. Multinomial Sample  — 按概率分布随机抽样（或 Argmax）
 */
#ifndef NANO_INFER_SAMPLING_PARAMS_H
#define NANO_INFER_SAMPLING_PARAMS_H

#include <cstdint>
#include <vector>

namespace sampler {

/// @brief 单请求采样参数（vLLM 风格）
struct SamplingParams {
    /// @brief Temperature 缩放因子
    /// - 1.0f: 不缩放（默认）
    /// - < 1.0f: 更尖锐（偏 Greedy）
    /// - > 1.0f: 更平坦（更多样化）
    /// - 0.0f: 退化为 Argmax（贪心采样）
    float temperature = 1.0f;

    /// @brief Top-K 过滤
    /// - -1: 不使用 Top-K（默认）
    /// - > 0: 仅保留概率最高的 K 个 token
    int32_t top_k = -1;

    /// @brief Top-P (Nucleus) 过滤
    /// - 1.0f: 不过滤（默认）
    /// - < 1.0f: 仅保留累积概率 ≤ p 的最小 token 集合
    float top_p = 1.0f;

    /// @brief 重复惩罚因子
    /// - 1.0f: 无惩罚（默认）
    /// - > 1.0f: 对已生成 token 的 logit 除以 penalty（正 logit）或乘以 penalty（负 logit）
    float repetition_penalty = 1.0f;

    /// @brief 随机种子
    /// - -1: 使用随机种子（默认）
    /// - >= 0: 固定种子，保证可复现
    int64_t seed = -1;

    /// @brief 是否使用贪心采样（temperature=0 时自动设为 true）
    bool use_greedy() const {
        return temperature <= 0.0f;
    }

    /// @brief 是否需要 Top-K 过滤
    bool use_top_k() const {
        return top_k > 0;
    }

    /// @brief 是否需要 Top-P 过滤
    bool use_top_p() const {
        return top_p < 1.0f && top_p > 0.0f;
    }

    /// @brief 是否需要重复惩罚
    bool use_repetition_penalty() const {
        return repetition_penalty != 1.0f;
    }

    /// @brief 默认贪心采样参数（temperature=0，确定性 Argmax）
    static SamplingParams greedy() {
        SamplingParams p;
        p.temperature = 0.0f;
        return p;
    }
};

}  // namespace sampler

#endif  // NANO_INFER_SAMPLING_PARAMS_H
