#ifndef LLAMA_INFER_ADD_KERNEL_H
#define LLAMA_INFER_ADD_KERNEL_H
#include "tensor/tensor.h"

namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}

#endif  // LLAMA_INFER_ADD_KERNEL_H
