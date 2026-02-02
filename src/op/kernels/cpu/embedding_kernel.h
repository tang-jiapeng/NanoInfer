#ifndef KUIPER_INFER_EMB_KERNEL_H
#define KUIPER_INFER_EMB_KERNEL_H
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, int32_t vocab_size,
                       void* stream = nullptr);
}  // namespace kernel
#endif  // KUIPER_INFER_EMB_KERNEL_H
