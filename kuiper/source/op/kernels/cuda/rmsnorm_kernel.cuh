#ifndef LLAMA_INFER_RMSNORM_KERNEL_CUH
#define LLAMA_INFER_RMSNORM_KERNEL_CUH
#include "tensor/tensor.h"

namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream = nullptr);

}

#endif  // LLAMA_INFER_RMSNORM_KERNEL_CUH
