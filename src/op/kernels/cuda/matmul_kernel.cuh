#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include "nanoinfer/tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);
}

#endif