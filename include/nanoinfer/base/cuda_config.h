/**
 * @file cuda_config.h
 * @brief CUDA 执行上下文 (Stream + cuBLAS Handle) 的 RAII 封装
 */
#ifndef NANO_INFER_CUDA_CONFIG_H
#define NANO_INFER_CUDA_CONFIG_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace kernel {

/**
 * @brief CUDA 执行配置，持有 Stream 和 cuBLAS Handle
 *
 * 析构时自动销毁资源。禁止浅拷贝，建议通过 std::shared_ptr 传递
 */
struct CudaConfig {
    cublasHandle_t cublas_handle;   ///< cuBLAS Handle
    cudaStream_t stream = nullptr;  ///< CUDA Stream

    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
        if (cublas_handle) {
            cublasDestroy(cublas_handle);
        }
    }
};
}  // namespace kernel

#endif  // NANO_INFER_CUDA_CONFIG_H
