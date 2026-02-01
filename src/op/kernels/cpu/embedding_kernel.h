#ifndef NANO_INFER_EMBEDDING_KERNEL_H
#define NANO_INFER_EMBEDDING_KERNEL_H

#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size,
                          void* stream = nullptr);
}

#endif  // NANO_INFER_EMBEDDING_KERNEL_H
