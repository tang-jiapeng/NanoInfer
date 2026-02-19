/**
 * @file argmax_sampler.cpp
 * @brief Argmax 贪心采样器实现
 *
 * ArgmaxSampler 基于 Greedy Search 策略，从 Logits 中选取最大值对应的 Token ID：
 *   - sample()（已弃用）：仅支持 CPU 的单条采样，直接调用 std::max_element
 *   - sample_batched()：支持 CPU/CUDA 的批量采样，通过 KernelRegistry 分发 "argmax" 算子
 */
#include "nanoinfer/sampler/argmax_sampler.h"
#include "../op/kernels/kernel_registry.h"
#include "../op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"

namespace sampler {

/** @brief [已弃用] 单条 CPU 采样，返回 logits 中最大值的索引 */
// 弃用
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        size_t next = std::distance(logits, std::max_element(logits, logits + size));
        return next;
    } else {
        // GPU 版本后续由 Kernel 实现
        LOG(ERROR) << "ArgmaxSampler::sample for GPU not implemented yet in this file.";
        return 0;
    }
}

/**
 * @brief 批量 Argmax 采样
 *
 * @param logits  输入 Logits [batch_size, vocab_size]
 * @param output_ids 输出 Token IDs [batch_size]（需预分配，容量 ≥ batch_size）
 * @param stream  CUDA 流（CPU 模式下忽略）
 *
 * 通过 KernelRegistry 获取 "argmax" 算子，自动区分 CPU/CUDA 后端。
 */
void ArgmaxSampler::sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                   void* stream) {
    // 维度检查
    CHECK_EQ(logits.dims_size(), 2) << "Logits tensor must be 2D [batch_size, vocab_size]";
    CHECK_EQ(output_ids.dims_size(), 1) << "Output tensor must be 1D [max_batch_size]";

    int32_t batch_size = logits.get_dim(0);

    // 容量检查
    CHECK_GE(output_ids.get_dim(0), batch_size)
        << "Output tensor size (" << output_ids.get_dim(0) << ") is smaller than batch size ("
        << batch_size << ")";

    auto argmax_kernel =
        kernel::KernelRegistry::instance().get<kernel::ArgmaxKernelFn>("argmax", device_type_);

    if (!argmax_kernel) {
        LOG(FATAL) << "Argmax kernel not found for device: " << static_cast<int>(device_type_);
        return;
    }

    // 3. 执行算子
    // 如果是 CPU，kernel 内部会忽略 stream 并同步执行
    // 如果是 CUDA，kernel 内部会异步提交到 stream
    argmax_kernel(logits, output_ids, stream);
}

}  // namespace sampler
