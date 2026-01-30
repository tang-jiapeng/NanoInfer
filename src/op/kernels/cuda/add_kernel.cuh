#ifndef NANO_INFER_ADD_KERNEL_CUH
#define NANO_INFER_ADD_KERNEL_CUH

#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}

#endif  // NANO_INFER_ADD_KERNEL_CUH
