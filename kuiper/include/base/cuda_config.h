#ifndef LLAMA_INFER_CUDA_CONFIG_H
#define LLAMA_INFER_CUDA_CONFIG_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
    cudaStream_t stream = nullptr;
    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};
}  // namespace kernel

#endif  // LLAMA_INFER_CUDA_CONFIG_H
