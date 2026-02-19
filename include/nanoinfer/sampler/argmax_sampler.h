/**
 * @file argmax_sampler.h
 * @brief Argmax (贪婪) 采样器
 */
#ifndef NANO_INFER_ARGMAX_SAMPLER_H
#define NANO_INFER_ARGMAX_SAMPLER_H

#include "sampler.h"

namespace sampler {

/**
 * @brief Argmax 采样器
 *
 * 确定性采样：始终选择概率最大的 Token。
 * 无需 Softmax，直接比较 Logits 数值
 */
class ArgmaxSampler : public Sampler {
   public:
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {
    }

    size_t sample(const float* logits, size_t size, void* stream) override;

    /**
     * @brief 批量 Argmax 采样
     * @param logits  [batch_size, vocab_size] float32
     * @param output_ids [batch_size] int32
     */
    void sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                        void* stream = nullptr) override;
};
}  // namespace sampler

#endif  // NANO_INFER_ARGMAX_SAMPLER_H
