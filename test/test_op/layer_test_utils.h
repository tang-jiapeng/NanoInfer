#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

// ---------------------------------------------------------------------------
// 创建 CUDA 执行配置 (Stream + cuBLAS handle)
// ---------------------------------------------------------------------------
inline std::shared_ptr<kernel::CudaConfig> make_cuda_config() {
    auto cfg = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cfg->stream);
    cublasCreate(&cfg->cublas_handle);
    cublasSetStream(cfg->cublas_handle, cfg->stream);
    return cfg;
}

// ---------------------------------------------------------------------------
// 在 CPU tensor 上填充均匀值
// ---------------------------------------------------------------------------
inline void fill_cpu(tensor::Tensor& t, float val) {
    float* p = t.ptr<float>();
    for (int32_t i = 0; i < t.size(); ++i) p[i] = val;
}

// ---------------------------------------------------------------------------
// 创建并填充 CPU float tensor (1-D)
// ---------------------------------------------------------------------------
inline tensor::Tensor make_cpu_tensor(int32_t n, float val = 0.f) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(base::DataType::kDataTypeFp32, n, true, alloc);
    fill_cpu(t, val);
    return t;
}

// ---------------------------------------------------------------------------
// 创建并填充 CPU float tensor (2-D)
// ---------------------------------------------------------------------------
inline tensor::Tensor make_cpu_tensor_2d(int32_t r, int32_t c, float val = 0.f) {
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(base::DataType::kDataTypeFp32, r, c, true, alloc);
    fill_cpu(t, val);
    return t;
}

// ---------------------------------------------------------------------------
// 创建 CUDA float tensor (2-D), 并用 cudaMemset 置零
// ---------------------------------------------------------------------------
inline tensor::Tensor make_cuda_tensor_2d(int32_t r, int32_t c) {
    auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
    return tensor::Tensor(base::DataType::kDataTypeFp32, r, c, true, alloc);
}

// ---------------------------------------------------------------------------
// 创建 CUDA float tensor (1-D)
// ---------------------------------------------------------------------------
inline tensor::Tensor make_cuda_tensor(int32_t n) {
    auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
    return tensor::Tensor(base::DataType::kDataTypeFp32, n, true, alloc);
}

// ---------------------------------------------------------------------------
// D2H: 把 CUDA tensor 数据读回 CPU vector
// ---------------------------------------------------------------------------
inline std::vector<float> d2h(const tensor::Tensor& t) {
    std::vector<float> buf(t.size());
    cudaMemcpy(buf.data(), t.ptr<float>(), t.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return buf;
}
