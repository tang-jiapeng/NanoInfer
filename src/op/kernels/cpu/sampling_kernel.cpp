/**
 * @file sampling_kernel.cpp
 * @brief CPU 多样化采样算子（Temperature / Top-K / Top-P / Repetition Penalty / Multinomial）
 *
 * CPU 实现的采样 Pipeline，功能与 CUDA 版一致：
 *   1. Repetition Penalty — 对已生成 token 的 logits 施加惩罚
 *   2. Temperature Scaling — logits /= temperature
 *   3. Top-K/Top-P Sampling — 过滤 + Softmax + Multinomial
 */
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU Repetition Penalty 算子
 *
 * @param logits         [batch_size, vocab_size] 原地修改
 * @param penalty_tokens [batch_size, max_penalty_len] 需惩罚的 token IDs（-1 填充）
 * @param penalties      [batch_size] 每个请求的 penalty 值
 * @param stream         CPU 下忽略
 */
void repetition_penalty_kernel_cpu(const tensor::Tensor& logits,
                                   const tensor::Tensor& penalty_tokens,
                                   const tensor::Tensor& penalties, [[maybe_unused]] void* stream) {
    CHECK(logits.device_type() == base::DeviceType::kDeviceCPU);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));
    int32_t max_penalty_len = static_cast<int32_t>(penalty_tokens.get_dim(1));

    float* logits_ptr = const_cast<float*>(logits.ptr<float>());
    const int32_t* tokens_ptr = penalty_tokens.ptr<int32_t>();
    const float* penalties_ptr = penalties.ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        float penalty = penalties_ptr[b];
        if (penalty == 1.0f) continue;

        float* row = logits_ptr + b * vocab_size;
        const int32_t* row_tokens = tokens_ptr + b * max_penalty_len;

        for (int i = 0; i < max_penalty_len; ++i) {
            int32_t token_id = row_tokens[i];
            if (token_id < 0 || token_id >= vocab_size) continue;

            float logit = row[token_id];
            row[token_id] = (logit > 0.0f) ? (logit / penalty) : (logit * penalty);
        }
    }
}

/**
 * @brief CPU Temperature Scaling 算子
 *
 * @param logits       [batch_size, vocab_size] 原地修改
 * @param temperatures [batch_size] 每个请求的 temperature
 * @param stream       CPU 下忽略
 */
void temperature_kernel_cpu(const tensor::Tensor& logits, const tensor::Tensor& temperatures,
                            [[maybe_unused]] void* stream) {
    CHECK(logits.device_type() == base::DeviceType::kDeviceCPU);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));

    float* logits_ptr = const_cast<float*>(logits.ptr<float>());
    const float* temps_ptr = temperatures.ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        float temp = temps_ptr[b];
        if (temp <= 0.0f || temp == 1.0f) continue;

        float inv_temp = 1.0f / temp;
        float* row = logits_ptr + b * vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            row[i] *= inv_temp;
        }
    }
}

/**
 * @brief CPU Top-K / Top-P / Multinomial 采样算子
 *
 * 算法：
 *   1. Top-K: partial_sort 得到 Top-K 个最大 logit
 *   2. Softmax (numerically stable)
 *   3. Top-P: 按概率降序累加，截断到累积概率 <= top_p
 *   4. 重新归一化 + Multinomial 采样
 *
 * @param logits     [batch_size, vocab_size]
 * @param output_ids [batch_size]
 * @param top_ks     [batch_size] Top-K 值（-1 不用）
 * @param top_ps     [batch_size] Top-P 值（1.0 不用）
 * @param seeds      [batch_size] 随机种子
 * @param stream     CPU 下忽略
 */
void top_k_top_p_sampling_kernel_cpu(const tensor::Tensor& logits, const tensor::Tensor& output_ids,
                                     const tensor::Tensor& top_ks, const tensor::Tensor& top_ps,
                                     const tensor::Tensor& seeds, [[maybe_unused]] void* stream) {
    CHECK(logits.device_type() == base::DeviceType::kDeviceCPU);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));

    const float* logits_ptr = logits.ptr<float>();
    int32_t* output_ptr = const_cast<int32_t*>(output_ids.ptr<int32_t>());
    const int32_t* top_ks_ptr = top_ks.ptr<int32_t>();
    const float* top_ps_ptr = top_ps.ptr<float>();
    const int64_t* seeds_ptr = seeds.ptr<int64_t>();

    // 工作缓冲区（复用以减少分配开销）
    std::vector<std::pair<float, int32_t>> candidates(vocab_size);

    for (int b = 0; b < batch_size; ++b) {
        const float* row = logits_ptr + b * vocab_size;
        int32_t k = top_ks_ptr[b];
        float p = top_ps_ptr[b];
        int64_t seed = seeds_ptr[b];

        // 构建 (logit, token_id) 对
        for (int i = 0; i < vocab_size; ++i) {
            candidates[i] = {row[i], i};
        }

        // 确定有效候选数量
        int32_t num_candidates = vocab_size;

        // Top-K 过滤：partial_sort 只排前 K 个
        if (k > 0 && k < vocab_size) {
            num_candidates = k;
            std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            // 不用 Top-K 时也要排序（Top-P 需要降序）
            std::sort(candidates.begin(), candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
        }

        // Softmax（numerically stable）
        float max_logit = candidates[0].first;
        std::vector<float> probs(num_candidates);
        float sum = 0.0f;
        for (int i = 0; i < num_candidates; ++i) {
            probs[i] = std::exp(candidates[i].first - max_logit);
            sum += probs[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < num_candidates; ++i) {
            probs[i] *= inv_sum;
        }

        // Top-P 过滤
        if (p < 1.0f && p > 0.0f) {
            float cumsum = 0.0f;
            int cutoff = num_candidates;
            for (int i = 0; i < num_candidates; ++i) {
                cumsum += probs[i];
                if (cumsum > p) {
                    cutoff = i + 1;
                    break;
                }
            }
            num_candidates = cutoff;

            // 重新归一化
            sum = 0.0f;
            for (int i = 0; i < num_candidates; ++i) {
                sum += probs[i];
            }
            inv_sum = 1.0f / sum;
            for (int i = 0; i < num_candidates; ++i) {
                probs[i] *= inv_sum;
            }
        }

        // Multinomial 采样
        std::mt19937 gen(static_cast<unsigned int>(seed + b));
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float u = dis(gen);

        float cumsum = 0.0f;
        int32_t selected = candidates[num_candidates - 1].second;  // fallback
        for (int i = 0; i < num_candidates; ++i) {
            cumsum += probs[i];
            if (cumsum >= u) {
                selected = candidates[i].second;
                break;
            }
        }

        output_ptr[b] = selected;
    }
}

REGISTER_KERNEL(repetition_penalty, kDeviceCPU, repetition_penalty_kernel_cpu)
REGISTER_KERNEL(temperature, kDeviceCPU, temperature_kernel_cpu)
REGISTER_KERNEL(top_k_top_p_sampling, kDeviceCPU, top_k_top_p_sampling_kernel_cpu)

}  // namespace kernel
