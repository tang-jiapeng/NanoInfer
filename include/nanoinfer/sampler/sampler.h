/**
 * @file sampler.h
 * @brief 采样策略抽象基类
 */
#ifndef NANO_INFER_SAMPLER_H
#define NANO_INFER_SAMPLER_H

#include <cstddef>
#include <cstdint>
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

namespace sampler {

/// @brief 采样策略抽象基类 (Argmax / Top-K / Top-P 等)
class Sampler {
   public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {
    }

    /**
     * @brief 单条采样：根据 Logits 选择下一个 Token
     * @param logits Logits 指针 (未经 Softmax)
     * @param size 词表大小
     * @param stream CUDA 流 (可选)
     * @return 被选中的 Token 索引
     */
    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

    /**
     * @brief 批量采样，用于 Continuous Batching Engine
     * @param logits  [batch_size, vocab_size]
     * @param output_ids [batch_size] 输出 Token ID
     */
    virtual void sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                void* stream = nullptr) {
        LOG(ERROR) << "sample_batched not implemented for this sampler";
    }

   protected:
    base::DeviceType device_type_;
};
}  // namespace sampler

#endif  // NANO_INFER_SAMPLER_H
