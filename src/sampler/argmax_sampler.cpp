#include "nanoinfer/sampler/argmax_sampler.h"
#include <algorithm>

namespace sampler {
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
    // 维度检查
    CHECK_EQ(logits.dims_size(), 2) << "Logits tensor must be 2D [batch_size, vocab_size]";
    CHECK_EQ(output_ids.dims_size(), 1) << "Output tensor must be 1D [max_batch_size]";

    int32_t batch_size = logits.get_dim(0);
    int32_t vocab_size = logits.get_dim(1);

    // 允许 output_ids 的容量 (max_batch_size) 大于当前 batch_size
    CHECK_GE(output_ids.get_dim(0), batch_size)
        << "Output tensor size (" << output_ids.get_dim(0) << ") is smaller than batch size ("
        << batch_size << ")";

    if (device_type_ == base::DeviceType::kDeviceCPU) {
        const float* logits_ptr = logits.ptr<float>();
        int32_t* output_ptr = output_ids.ptr<int32_t>();

        // 只遍历当前有效的 batch_size，不会越界
        for (int i = 0; i < batch_size; ++i) {
            const float* row_start = logits_ptr + i * vocab_size;
            const float* row_end = row_start + vocab_size;

            auto max_iter = std::max_element(row_start, row_end);
            int32_t max_idx = static_cast<int32_t>(std::distance(row_start, max_iter));

            output_ptr[i] = max_idx;
        }
    } else {
        LOG(ERROR) << "ArgmaxSampler::sample_batched for GPU not implemented yet.";
    }
}

}  // namespace sampler
