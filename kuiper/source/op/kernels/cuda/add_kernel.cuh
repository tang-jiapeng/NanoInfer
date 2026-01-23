#ifndef LLAMA_INFER_ADD_KERNEL_CUH
#define LLAMA_INFER_ADD_KERNEL_CUH

#include "tensor/tensor.h"

namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}

#endif  // LLAMA_INFER_ADD_KERNEL_CUH
