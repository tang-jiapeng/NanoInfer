#ifndef NANO_INFER_SOFTMAX_H
#define NANO_INFER_SOFTMAX_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {

void softmax_kernel_cpu(const tensor::Tensor& input, void* stream = nullptr);

}


#endif //NANO_INFER_SOFTMAX_H
