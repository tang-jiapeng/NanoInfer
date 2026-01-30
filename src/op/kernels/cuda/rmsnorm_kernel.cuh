#ifndef NANO_INFER_RMSNORM_KERNEL_CUH
#define NANO_INFER_RMSNORM_KERNEL_CUH
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream = nullptr);

}

#endif  // NANO_INFER_RMSNORM_KERNEL_CUH
