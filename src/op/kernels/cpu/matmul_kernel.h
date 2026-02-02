#ifndef NANO_INFER_MATMUL_KERNEL_H
#define NANO_INFER_MATMUL_KERNEL_H
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"
#include <armadillo>

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const CudaConfig* config = nullptr);
}  // namespace kernel
#endif  // NANO_INFER_MATMUL_KERNEL_H
