#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "../src/op/kernels/cpu/swiglu_kernel.h"
#include "../src/op/kernels/cuda/swiglu_kernel.cuh"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

class SwiGLUKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_allocator_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }

    void Randomize(std::vector<float>& data) {
        std::mt19937 gen(2024);
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        for (auto& val : data) val = dis(gen);
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;
};

TEST_F(SwiGLUKernelTest, CompareCpuWithGpu) {
    // 模拟 Llama-2-7b 的 FFN Hidden Dim (11008) * Batch Size (2)
    int32_t size = 11008 * 2;

    // 1. Host Data
    std::vector<float> h_in1(size);  // Gate
    std::vector<float> h_in2(size);  // Up
    std::vector<float> h_out_gpu(size);

    Randomize(h_in1);
    Randomize(h_in2);

    // 2. Tensors
    tensor::Tensor cpu_in1(base::DataType::kDataTypeFp32, size, true, cpu_allocator_);
    tensor::Tensor cpu_in2(base::DataType::kDataTypeFp32, size, true, cpu_allocator_);
    tensor::Tensor cpu_out(base::DataType::kDataTypeFp32, size, true, cpu_allocator_);

    tensor::Tensor gpu_in1(base::DataType::kDataTypeFp32, size, true, cuda_allocator_);
    tensor::Tensor gpu_in2(base::DataType::kDataTypeFp32, size, true, cuda_allocator_);
    tensor::Tensor gpu_out(base::DataType::kDataTypeFp32, size, true, cuda_allocator_);

    // 3. Copy Data
    std::memcpy(cpu_in1.ptr<void>(), h_in1.data(), size * sizeof(float));
    std::memcpy(cpu_in2.ptr<void>(), h_in2.data(), size * sizeof(float));

    cudaMemcpy(gpu_in1.ptr<void>(), h_in1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_in2.ptr<void>(), h_in2.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Run CPU
    kernel::swiglu_kernel_cpu(cpu_in1, cpu_in2, cpu_out, nullptr);

    // 5. Run GPU
    kernel::swiglu_kernel_cu(gpu_in1, gpu_in2, gpu_out, nullptr);
    cudaDeviceSynchronize();

    // 6. Compare
    cudaMemcpy(h_out_gpu.data(), gpu_out.ptr<void>(), size * sizeof(float), cudaMemcpyDeviceToHost);

    const float* cpu_res = cpu_in1.ptr<float>();
    const float* cpu_real_res = cpu_out.ptr<float>();

    double max_diff = 0.0;
    for (int i = 0; i < size; ++i) {
        float c = cpu_real_res[i];
        float g = h_out_gpu[i];
        float diff = std::abs(c - g);
        max_diff = std::max(max_diff, (double)diff);

        // Swish involves exp, might have some precision divergence
        EXPECT_NEAR(c, g, 1e-4) << "Mismatch at index " << i;
    }

    // 验证 Input1 没有被修改 (针对 CPU 算子之前的 Bug)
    // 重新比对 cpu_in1 和 h_in1
    for (int i = 0; i < size; ++i) {
        EXPECT_EQ(cpu_in1.ptr<float>()[i], h_in1[i]) << "Input1 was modified by CPU kernel!";
    }

    LOG(INFO) << "SwiGLU CPU vs GPU Passed. Max Diff: " << max_diff;
}