#ifndef NANO_INFER_ARGMAX_KERNEL_H
#define NANO_INFER_ARGMAX_KERNEL_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void argmax_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& output,
                       void* stream = nullptr);
}  // namespace kernel
#endif