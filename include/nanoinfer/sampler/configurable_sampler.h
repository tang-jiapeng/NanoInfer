/**
 * @file configurable_sampler.h
 * @brief 可配置采样器（支持 Temperature / Top-K / Top-P / Repetition Penalty）
 *
 * 参照 vLLM 的采样 Pipeline 设计，按以下顺序处理：
 *   1. Repetition Penalty  — 惩罚已生成 token
 *   2. Temperature Scaling — 控制输出随机性
 *   3. Top-K Filtering     — 保留 K 个最高概率 token
 *   4. Top-P Filtering     — 按累积概率截断
 *   5. Softmax + Multinomial / Argmax — 最终采样
 *
 * 支持 per-request 独立采样参数（通过 SamplingParams），
 * 向后兼容 Sampler 基类接口（无参版本使用默认 Greedy）。
 */
#ifndef NANO_INFER_CONFIGURABLE_SAMPLER_H
#define NANO_INFER_CONFIGURABLE_SAMPLER_H

#include <memory>
#include <random>
#include <vector>
#include "sampler.h"
#include "sampling_params.h"

namespace sampler {

/**
 * @brief 可配置采样器
 *
 * 核心特性：
 *   - 兼容 Sampler 基类接口（sample_batched 无参版 = Greedy Argmax）
 *   - 扩展 sample_batched 支持 per-request SamplingParams
 *   - 支持 per-request 的 generated_tokens（用于 Repetition Penalty）
 *   - 内部通过 KernelRegistry 分发到 CPU/CUDA kernel
 */
class ConfigurableSampler : public Sampler {
   public:
    explicit ConfigurableSampler(base::DeviceType device_type);

    /**
     * @brief [已弃用] 单条 CPU 采样（退化为 Argmax）
     */
    size_t sample(const float* logits, size_t size, void* stream = nullptr) override;

    /**
     * @brief 批量采样（无参版，等价于 Argmax）
     *
     * 向后兼容 Sampler 基类接口，直接走 Argmax kernel。
     */
    void sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                        void* stream = nullptr) override;

    /**
     * @brief 批量采样（per-request 采样参数）
     *
     * 执行完整的采样 Pipeline：
     *   RepPenalty → Temperature → Top-K/Top-P/Multinomial
     *
     * @param logits      [batch_size, vocab_size] 原始 logits（会被原地修改）
     * @param output_ids  [batch_size] 输出 token IDs
     * @param params      per-request 采样参数列表（size == batch_size）
     * @param generated_tokens_list  per-request 已生成 token 列表（用于 RepPenalty，可空）
     * @param stream      CUDA 流
     */
    void sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                        const std::vector<SamplingParams>& params,
                        const std::vector<std::vector<int32_t>>& generated_tokens_list = {},
                        void* stream = nullptr);

   private:
    /// @brief 内部：对全 greedy 的 batch 走 argmax 快速路径
    void argmax_fallback(const tensor::Tensor& logits, tensor::Tensor& output_ids, void* stream);

    /// @brief 内部：执行 Repetition Penalty kernel
    void apply_repetition_penalty(const tensor::Tensor& logits, int32_t batch_size,
                                  const std::vector<SamplingParams>& params,
                                  const std::vector<std::vector<int32_t>>& generated_tokens_list,
                                  void* stream);

    /// @brief 内部：执行 Temperature Scaling kernel
    void apply_temperature(const tensor::Tensor& logits, int32_t batch_size,
                           const std::vector<SamplingParams>& params, void* stream);

    /// @brief 内部：执行 Top-K/Top-P/Multinomial kernel
    void apply_top_k_top_p_sampling(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                    int32_t batch_size, const std::vector<SamplingParams>& params,
                                    void* stream);

    /// @brief 随机数生成器（用于给无固定 seed 的请求生成种子）
    std::mt19937_64 rng_;
};

}  // namespace sampler

#endif  // NANO_INFER_CONFIGURABLE_SAMPLER_H
