#ifndef NANO_INFER_SOFTMAX_KERNEL_H
#define NANO_INFER_SOFTMAX_KERNEL_H

#include "nanoinfer/tensor/tensor.h"
#include <armadillo>

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& input, void* stream = nullptr);
}  // namespace kernel
#endif  // NANO_INFER_SOFTMAX_KERNEL_H
