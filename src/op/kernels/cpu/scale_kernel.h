#ifndef NANO_INFER_SCALE_KERNEL_H
#define NANO_INFER_SCALE_KERNEL_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {

void scale_kernel_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);

}
#endif  // SCALE_KERNEL_H
