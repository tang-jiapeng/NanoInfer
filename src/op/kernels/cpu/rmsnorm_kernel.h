#ifndef NANO_INFER_RMSNORM_KERNEL_H
#define NANO_INFER_RMSNORM_KERNEL_H
#include "nanoinfer/tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif  // NANO_INFER_RMSNORM_KERNEL_H
