#ifndef NANO_INFER_CUDA_CONFIG_H
#define NANO_INFER_CUDA_CONFIG_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace kernel {

/**
 * @brief CUDA 执行配置容器
 *
 * 用于传递和管理 CUDA Kernel 执行所需的上下文资源，核心是 CUDA Stream
 * 采用 RAII (Resource Acquisition Is Initialization) 机制管理 Stream 的生命周期
 *
 * @note
 * 该结构体的析构函数会调用 cudaStreamDestroy 销毁流
 *
 * 警告：请避免直接拷贝该对象 (Shallow Copy)，否则两个对象的析构函数会尝试释放同一个
 * Stream， 导致 Double Free 崩溃
 *
 * 最佳实践是使用 std::shared_ptr<CudaConfig> 进行管理和传递
 */
struct CudaConfig {
    cublasHandle_t cublas_handle;   ///< cuBLAS 句柄，供需要使用 cuBLAS 的 Kernel 使用
    cudaStream_t stream = nullptr;  ///< CUDA 流句柄，用于实现异步并行计算

    /**
     * @brief 析构函数
     *
     * 自动销毁持有的 CUDA 流资源，防止显存/句柄泄漏。
     * 只有当 stream 指针非空时才执行销毁。
     */
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
