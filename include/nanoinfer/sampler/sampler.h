#ifndef NANO_INFER_SAMPLER_H
#define NANO_INFER_SAMPLER_H

#include <cstddef>
#include <cstdint>
#include "nanoinfer/base/base.h"

namespace sampler {

/**
 * @brief 采样策略抽象基类
 *
 * 所有的采样方法 (Argmax, Top-K, Top-P 等) 都应继承此类
 */
class Sampler {
   public:
    /**
     * @brief 构造函数
     * @param device_type 采样发生所在的设备 (通常 Logits 在 GPU 上，采样也在 GPU
     * 上进行以减少 D2H 拷贝)
     */
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {
    }

    /**
     * @brief 执行采样 (核心接口)
     *
     * 根据给定的 Logits 概率分布，选择下一个 Token 的 ID。
     *
     * @param logits 模型最后一层的输出指针 (通常未经过 Softmax，但在 Argmax 中不影响结果)
     * @param size 词表大小 (Vocabulary Size)，即 logits 数组的长度
     * @param stream CUDA 流句柄。如果 device_type_ 为 CUDA，则需要在此流上执行 kernel
     * @return size_t 被选中的 Token Index
     */
    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

   protected:
    base::DeviceType device_type_;
};
}  // namespace sampler

#endif  // NANO_INFER_SAMPLER_H
