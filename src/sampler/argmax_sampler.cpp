#include "nanoinfer/sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/kernels_interface.h"

namespace sampler {
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

void ArgmaxSampler::sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                   void* stream) {
    // 1. 维度检查
    CHECK_EQ(logits.dims_size(), 2) << "Logits tensor must be 2D [batch_size, vocab_size]";
    CHECK_EQ(output_ids.dims_size(), 1) << "Output tensor must be 1D [max_batch_size]";

    int32_t batch_size = logits.get_dim(0);
    // int32_t vocab_size = logits.get_dim(1); // Kernel 内部会获取

    // 容量检查
    CHECK_GE(output_ids.get_dim(0), batch_size)
        << "Output tensor size (" << output_ids.get_dim(0) << ") is smaller than batch size ("
        << batch_size << ")";

    // 2. 获取算子 (CPU 或 CUDA 由 Factory 决定)
    kernel::ArgmaxKernel argmax_kernel = kernel::get_argmax_kernel(device_type_);

    if (!argmax_kernel) {
        LOG(FATAL) << "Argmax kernel is not implemented or registered for device type: "
                   << static_cast<int>(device_type_);
        return;
    }

    // 3. 执行算子
    // 如果是 CPU，kernel 内部会忽略 stream 并同步执行
    // 如果是 CUDA，kernel 内部会异步提交到 stream
    argmax_kernel(logits, output_ids, stream);
}

}  // namespace sampler
