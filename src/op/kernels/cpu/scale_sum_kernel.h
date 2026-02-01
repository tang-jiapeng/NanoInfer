#ifndef NANO_INFER_SCALE_SUM_KERNEL_H
#define NANO_INFER_SCALE_SUM_KERNEL_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                          const tensor::Tensor& output, int pos, int size, int stride,
                          void* stream);
}

#endif  // NANO_INFER_SCALE_SUM_KERNEL_H
