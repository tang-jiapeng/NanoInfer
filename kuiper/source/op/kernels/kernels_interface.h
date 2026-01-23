#ifndef LLAMA_INFER_KERNELS_INTERFACE_H
#define LLAMA_INFER_KERNELS_INTERFACE_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"

namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);


AddKernel get_add_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

}

#endif  // LLAMA_INFER_KERNELS_INTERFACE_H
