#ifndef NANO_INFER_SWIGLU_KERNEL_H
#define NANO_INFER_SWIGLU_KERNEL_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);
}

#endif  // NANO_INFER_SWIGLU_KERNEL_H
