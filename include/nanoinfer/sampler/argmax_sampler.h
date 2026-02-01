#ifndef NANO_INFER_ARGMAX_SAMPLER_H
#define NANO_INFER_ARGMAX_SAMPLER_H

#include "sampler.h"

namespace sampler {

/**
 * @brief Argmax (贪婪) 采样器
 *
 * 最简单的采样策略：始终选择概率最大的 Token。
 * 对应数学公式：index = argmax(logits)
 *
 * 特点：
 * 1. 确定性 (Deterministic)：相同的输入永远产生相同的输出。
 * 2. 适用于数学题、代码生成等需要严谨逻辑的场景。
 * 3. 不需要 Softmax 归一化，直接比较 Logits 数值大小即可。
 */
class ArgmaxSampler : public Sampler {
   public:
    /**
     * @brief 构造函数
     * @param device_type 运行设备
     */
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {
    }

    /**
     * @brief 执行 Argmax 采样
     *
     * 1. CPU 模式：遍历数组寻找最大值索引。
     * 2. CUDA 模式：调用 Argmax Kernel (通常涉及 Block Reduce)
     *
     * @param logits Logits 指针
     * @param size 词表大小
     * @param stream CUDA 流
     * @return size_t 最大 Logit 对应的索引
     */
    size_t sample(const float* logits, size_t size, void* stream) override;
};
}  // namespace sampler

#endif  // NANO_INFER_ARGMAX_SAMPLER_H
