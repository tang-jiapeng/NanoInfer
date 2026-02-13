#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../src/op/kernels/cpu/rope_kernel.h"
#include "../src/op/kernels/cuda/rope_kernel.cuh"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

class RoPEKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
};

TEST_F(RoPEKernelTest, CompareCpuWithGpu) {
    // 参数设置
    int32_t total_tokens = 2;  // Batch=2
    int32_t dim = 4;           // Q Dim
    int32_t kv_dim = 2;        // K Dim (MQA 场景，只有 Q 一半)
    int32_t head_size = 2;     // Head Size
    int32_t max_seq_len = 5;

    // 1. 准备数据
    // Q: Token0=[1,1,1,1], Token1=[2,2,2,2]
    std::vector<float> h_q = {1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f};
    // K: Token0=[1,1],     Token1=[2,2]
    std::vector<float> h_k = {1.f, 1.f, 2.f, 2.f};

    // Pos: Token0在位置0, Token1在位置1
    std::vector<int32_t> h_pos = {0, 1};

    // Sin/Cos Cache: 简单伪造
    // Pos 0: Sin=0, Cos=1 (不旋转)
    // Pos 1: Sin=1, Cos=0 (旋转 90度)
    // Cache Size = max_seq_len * head_size = 5 * 2 = 10
    std::vector<float> h_sin(10, 0.0f);
    std::vector<float> h_cos(10, 1.0f);

    // 手动修改 Pos 1 对应的值
    // idx = pos * head_size + dim
    h_sin[1 * 2 + 0] = 1.0f;
    h_sin[1 * 2 + 1] = 1.0f;
    h_cos[1 * 2 + 0] = 0.0f;
    h_cos[1 * 2 + 1] = 0.0f;

    // 2. CPU Tensors
    tensor::Tensor cpu_q(base::DataType::kDataTypeFp32, total_tokens, dim, true, cpu_alloc_);
    tensor::Tensor cpu_k(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, cpu_alloc_);
    tensor::Tensor cpu_pos(base::DataType::kDataTypeInt32, total_tokens, true, cpu_alloc_);
    tensor::Tensor cpu_sin(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, cpu_alloc_);
    tensor::Tensor cpu_cos(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, cpu_alloc_);

    // 拷贝到 CPU
    std::memcpy(cpu_q.ptr<void>(), h_q.data(), h_q.size() * sizeof(float));
    std::memcpy(cpu_k.ptr<void>(), h_k.data(), h_k.size() * sizeof(float));
    std::memcpy(cpu_pos.ptr<void>(), h_pos.data(), h_pos.size() * sizeof(int32_t));
    std::memcpy(cpu_sin.ptr<void>(), h_sin.data(), h_sin.size() * sizeof(float));
    std::memcpy(cpu_cos.ptr<void>(), h_cos.data(), h_cos.size() * sizeof(float));

    // 3. GPU Tensors
    tensor::Tensor gpu_q(base::DataType::kDataTypeFp32, total_tokens, dim, true, gpu_alloc_);
    tensor::Tensor gpu_k(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, gpu_alloc_);
    tensor::Tensor gpu_pos(base::DataType::kDataTypeInt32, total_tokens, true, gpu_alloc_);
    tensor::Tensor gpu_sin(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, gpu_alloc_);
    tensor::Tensor gpu_cos(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, gpu_alloc_);

    // 拷贝到 GPU
    cudaMemcpy(gpu_q.ptr<void>(), h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_k.ptr<void>(), h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pos.ptr<void>(), h_pos.data(), h_pos.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_sin.ptr<void>(), h_sin.data(), h_sin.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cos.ptr<void>(), h_cos.data(), h_cos.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // 4. Run CPU
    kernel::rope_kernel_cpu(dim, kv_dim, head_size, cpu_q, cpu_k, cpu_pos, cpu_sin, cpu_cos,
                            nullptr);

    // 5. Run GPU
    kernel::rope_kernel_cu(dim, kv_dim, head_size, gpu_q, gpu_k, gpu_pos, gpu_sin, gpu_cos,
                           nullptr);
    cudaDeviceSynchronize();

    // 6. 验证结果
    std::vector<float> res_q_gpu(h_q.size());
    std::vector<float> res_k_gpu(h_k.size());

    cudaMemcpy(res_q_gpu.data(), gpu_q.ptr<void>(), h_q.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(res_k_gpu.data(), gpu_k.ptr<void>(), h_k.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    const float* res_q_cpu = cpu_q.ptr<float>();
    const float* res_k_cpu = cpu_k.ptr<float>();

    // 比对 Q
    for (size_t i = 0; i < h_q.size(); ++i) {
        EXPECT_NEAR(res_q_cpu[i], res_q_gpu[i], 1e-4) << "Q mismatch at " << i;
    }
    // 比对 K
    for (size_t i = 0; i < h_k.size(); ++i) {
        EXPECT_NEAR(res_k_cpu[i], res_k_gpu[i], 1e-4) << "K mismatch at " << i;
    }

    // 额外逻辑验证: Token 1 应该被旋转 (原始 2, 旋转后 -2, 2)
    // Q Token 1 Start Index = 4
    EXPECT_NEAR(res_q_cpu[4], -2.0f, 1e-4);
    EXPECT_NEAR(res_q_cpu[5], 2.0f, 1e-4);
}