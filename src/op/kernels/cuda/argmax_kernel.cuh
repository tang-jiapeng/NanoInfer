#ifndef ARGMAX_KERNEL_CUH
#define ARGMAX_KERNEL_CUH
#include "nanoinfer/tensor/tensor.h"
namespace kernel {
void argmax_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& output,
                      void* stream);
}
#endif  // ARGMAX_KERNEL_CUH