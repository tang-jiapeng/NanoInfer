#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

class RMSNormComparisonTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_allocator_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }

    // 辅助函数：生成随机 float 数据
    void Randomize(std::vector<float>& data, float min = -1.0f, float max = 1.0f) {
        std::mt19937 gen(2024);
        std::uniform_real_distribution<float> dis(min, max);
        for (auto& val : data) {
            val = dis(gen);
        }
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;
};

TEST_F(RMSNormComparisonTest, CompareCpuWithGpu) {
    // 1. 设置测试规模 (模拟真实的 Llama 场景)
    int32_t batch_size = 16;  // Batch Size
    int32_t seq_len = 128;    // Sequence Length
    int32_t dim = 4096;       // Hidden Dimension
    int32_t total_tokens = batch_size * seq_len;

    // 2. 准备 Host 数据
    std::vector<float> h_input(total_tokens * dim);
    std::vector<float> h_weight(dim);
    std::vector<float> h_out_cpu(total_tokens * dim);
    std::vector<float> h_out_gpu(total_tokens * dim);

    // 填充随机数据
    Randomize(h_input, -2.0f, 2.0f);
    Randomize(h_weight, 0.5f, 1.5f);  // 权重通常是非负且接近 1

    // 3. 准备 Tensor
    // CPU Tensors
    tensor::Tensor t_in_cpu(base::DataType::kDataTypeFp32, total_tokens, dim, true, cpu_allocator_);
    tensor::Tensor t_wei_cpu(base::DataType::kDataTypeFp32, dim, true, cpu_allocator_);
    tensor::Tensor t_out_cpu(base::DataType::kDataTypeFp32, total_tokens, dim, true,
                             cpu_allocator_);

    // GPU Tensors
    tensor::Tensor t_in_gpu(base::DataType::kDataTypeFp32, total_tokens, dim, true,
                            cuda_allocator_);
    tensor::Tensor t_wei_gpu(base::DataType::kDataTypeFp32, dim, true, cuda_allocator_);
    tensor::Tensor t_out_gpu(base::DataType::kDataTypeFp32, total_tokens, dim, true,
                             cuda_allocator_);

    // 4. 拷贝数据 Host -> CPU Tensor & GPU Tensor
    std::memcpy(t_in_cpu.ptr<void>(), h_input.data(), h_input.size() * sizeof(float));
    std::memcpy(t_wei_cpu.ptr<void>(), h_weight.data(), h_weight.size() * sizeof(float));

    cudaMemcpy(t_in_gpu.ptr<void>(), h_input.data(), h_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(t_wei_gpu.ptr<void>(), h_weight.data(), h_weight.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // 5. 运行 CPU Kernel
    auto rmsnorm_cpu = kernel::KernelRegistry::instance().get<kernel::RMSNormKernelFn>(
        "rmsnorm", base::DeviceType::kDeviceCPU);
    rmsnorm_cpu(t_in_cpu, t_wei_cpu, t_out_cpu, 1e-5f, nullptr);

    // 6. 运行 GPU Kernel
    auto rmsnorm_cu = kernel::KernelRegistry::instance().get<kernel::RMSNormKernelFn>(
        "rmsnorm", base::DeviceType::kDeviceCUDA);
    rmsnorm_cu(t_in_gpu, t_wei_gpu, t_out_gpu, 1e-5f, nullptr);
    cudaDeviceSynchronize();  // 确保 GPU 跑完

    // 7. 拷贝 GPU 结果回 Host 用于对比
    cudaMemcpy(h_out_gpu.data(), t_out_gpu.ptr<void>(), h_out_gpu.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 8. 逐位对比
    // 获取 CPU 计算结果指针
    const float* cpu_res = t_out_cpu.ptr<float>();
    const float* gpu_res = h_out_gpu.data();

    double max_diff = 0.0;
    for (int i = 0; i < total_tokens * dim; ++i) {
        float diff = std::abs(cpu_res[i] - gpu_res[i]);
        max_diff = std::max(max_diff, (double)diff);

        // 允许较小的误差 (GPU rsqrtf 和 CPU sqrt 精度略有不同)
        // 1e-4 对于 fp32 累加来说是合理的阈值
        ASSERT_NEAR(cpu_res[i], gpu_res[i], 1e-4)
            << "Mismatch at index " << i << " CPU: " << cpu_res[i] << " GPU: " << gpu_res[i];
    }

    LOG(INFO) << "RMSNorm CPU vs GPU Check Passed! Max Diff: " << max_diff;
}