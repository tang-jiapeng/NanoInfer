#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "../src/op/kernels/cpu/matmul_kernel.h"
#include "../src/op/kernels/cuda/matmul_kernel.cuh"
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

using namespace kernel;

class MatMulKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // Init cuBLAS
        cuda_config_ = std::make_unique<CudaConfig>();
        cublasStatus_t status = cublasCreate(&cuda_config_->cublas_handle);
        ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS) << "Failed to create cuBLAS handle";
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
    std::unique_ptr<CudaConfig> cuda_config_;
};

TEST_F(MatMulKernelTest, CompareCpuWithGpu) {
    // 模拟 Decode 阶段参数
    // Batch=4, InputDim(K)=128, OutputDim(N)=32
    int32_t batch = 4;
    int32_t K = 128;
    int32_t N = 32;

    int32_t in_size = batch * K;
    int32_t wei_size = N * K;
    int32_t out_size = batch * N;

    // 1. Data Initialization
    std::vector<float> h_in(in_size);
    std::vector<float> h_wei(wei_size);

    // 初始化数据：Input=1.0, Weight=0.5 -> Output应该全是 1.0 * 0.5 * 128 = 64.0
    for (auto& v : h_in) v = 1.0f;
    for (auto& v : h_wei) v = 0.5f;

    // 2. CPU Tensors
    tensor::Tensor cpu_in(base::DataType::kDataTypeFp32, batch, K, true, cpu_alloc_);
    tensor::Tensor cpu_wei(base::DataType::kDataTypeFp32, N, K, true, cpu_alloc_);
    tensor::Tensor cpu_out(base::DataType::kDataTypeFp32, batch, N, true, cpu_alloc_);

    std::memcpy(cpu_in.ptr<void>(), h_in.data(), in_size * sizeof(float));
    std::memcpy(cpu_wei.ptr<void>(), h_wei.data(), wei_size * sizeof(float));

    // 3. GPU Tensors
    tensor::Tensor gpu_in(base::DataType::kDataTypeFp32, batch, K, true, gpu_alloc_);
    tensor::Tensor gpu_wei(base::DataType::kDataTypeFp32, N, K, true, gpu_alloc_);
    tensor::Tensor gpu_out(base::DataType::kDataTypeFp32, batch, N, true, gpu_alloc_);

    cudaMemcpy(gpu_in.ptr<void>(), h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wei.ptr<void>(), h_wei.data(), wei_size * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Run CPU
    kernel::matmul_kernel_cpu(cpu_in, cpu_wei, cpu_out, 1.0f, nullptr);

    // 5. Run GPU
    kernel::matmul_kernel_cu(gpu_in, gpu_wei, gpu_out, 1.0f, cuda_config_.get());
    cudaDeviceSynchronize();

    // 6. Check Results
    std::vector<float> h_out_gpu(out_size);
    cudaMemcpy(h_out_gpu.data(), gpu_out.ptr<void>(), out_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 获取 CPU 计算结果
    const float* cpu_res = cpu_out.ptr<float>();

    for (int i = 0; i < out_size; ++i) {
        // CPU vs Expected (64.0)
        EXPECT_NEAR(cpu_res[i], 64.0f, 1e-3);
        // GPU vs Expected (64.0)
        EXPECT_NEAR(h_out_gpu[i], 64.0f, 1e-3);
        // CPU vs GPU
        EXPECT_NEAR(cpu_res[i], h_out_gpu[i], 1e-3);
    }
}