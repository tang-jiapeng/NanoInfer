#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "../src/op/kernels/cpu/embedding_kernel.h"
#include "../src/op/kernels/cuda/embedding_kernel.cuh"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

class EmbeddingKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_allocator_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;
};
TEST_F(EmbeddingKernelTest, BasicLookup) {
    int32_t vocab_size = 10;
    int32_t dim = 8;
    int32_t token_num = 3;

    // 1. 准备 Weight [10, 8]
    // 为了验证方便，第 i 行的值全设为 i
    std::vector<float> h_weight(vocab_size * dim);
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < dim; ++j) {
            h_weight[i * dim + j] = static_cast<float>(i);
        }
    }

    // 2. 准备 Input Tokens [1, 5, 9]
    std::vector<int32_t> h_input = {1, 5, 9};

    // 3. GPU Tensors
    tensor::Tensor d_weight(base::DataType::kDataTypeFp32, vocab_size, dim, true, cuda_allocator_);
    tensor::Tensor d_input(base::DataType::kDataTypeInt32, token_num, true, cuda_allocator_);
    tensor::Tensor d_output(base::DataType::kDataTypeFp32, token_num, dim, true, cuda_allocator_);

    // 4. Copy to Device
    cudaMemcpy(d_weight.ptr<void>(), h_weight.data(), h_weight.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input.ptr<void>(), h_input.data(), h_input.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // 5. Run Kernel
    kernel::embedding_kernel_cu(d_input, d_weight, d_output, vocab_size, nullptr);

    // 6. Check Result
    std::vector<float> h_output(token_num * dim);
    cudaMemcpy(h_output.data(), d_output.ptr<void>(), h_output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Token 1 -> Output Row 0 should be all 1.0
    for (int j = 0; j < dim; ++j) EXPECT_FLOAT_EQ(h_output[0 * dim + j], 1.0f);
    // Token 5 -> Output Row 1 should be all 5.0
    for (int j = 0; j < dim; ++j) EXPECT_FLOAT_EQ(h_output[1 * dim + j], 5.0f);
    // Token 9 -> Output Row 2 should be all 9.0
    for (int j = 0; j < dim; ++j) EXPECT_FLOAT_EQ(h_output[2 * dim + j], 9.0f);
}

TEST_F(EmbeddingKernelTest, CompareCpuWithGpu) {
    int32_t vocab_size = 100;
    int32_t dim = 64;
    int32_t total_tokens = 20;  // 模拟 Batch Flatten 后的总 Token 数

    // 1. 准备 Host 数据 (随机初始化)
    std::vector<float> h_weight(vocab_size * dim);
    std::vector<int32_t> h_input(total_tokens);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> token_dis(0, vocab_size - 1);

    for (auto& w : h_weight) w = dis(gen);
    for (auto& t : h_input) t = token_dis(gen);

    // 2. 准备 CPU Tensors 并填充数据
    tensor::Tensor t_in_cpu(base::DataType::kDataTypeInt32, total_tokens, true, cpu_allocator_);
    tensor::Tensor t_wei_cpu(base::DataType::kDataTypeFp32, vocab_size, dim, true, cpu_allocator_);
    tensor::Tensor t_out_cpu(base::DataType::kDataTypeFp32, total_tokens, dim, true,
                             cpu_allocator_);

    std::memcpy(t_in_cpu.ptr<void>(), h_input.data(), h_input.size() * sizeof(int32_t));
    std::memcpy(t_wei_cpu.ptr<void>(), h_weight.data(), h_weight.size() * sizeof(float));

    // 3. 准备 GPU Tensors 并拷贝数据
    tensor::Tensor t_in_gpu(base::DataType::kDataTypeInt32, total_tokens, true, cuda_allocator_);
    tensor::Tensor t_wei_gpu(base::DataType::kDataTypeFp32, vocab_size, dim, true, cuda_allocator_);
    tensor::Tensor t_out_gpu(base::DataType::kDataTypeFp32, total_tokens, dim, true,
                             cuda_allocator_);

    cudaMemcpy(t_in_gpu.ptr<void>(), h_input.data(), h_input.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(t_wei_gpu.ptr<void>(), h_weight.data(), h_weight.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // 4. Run CPU Kernel
    kernel::embedding_kernel_cpu(t_in_cpu, t_wei_cpu, t_out_cpu, vocab_size, nullptr);

    // 5. Run GPU Kernel
    kernel::embedding_kernel_cu(t_in_gpu, t_wei_gpu, t_out_gpu, vocab_size, nullptr);
    cudaDeviceSynchronize();

    // 6. 对比结果
    std::vector<float> h_out_cpu(total_tokens * dim);
    std::vector<float> h_out_gpu(total_tokens * dim);

    std::memcpy(h_out_cpu.data(), t_out_cpu.ptr<void>(), h_out_cpu.size() * sizeof(float));
    cudaMemcpy(h_out_gpu.data(), t_out_gpu.ptr<void>(), h_out_gpu.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < h_out_cpu.size(); ++i) {
        // Embedding 只是拷贝，应该是完全相等的，不需要 tolerance
        EXPECT_EQ(h_out_cpu[i], h_out_gpu[i]) << "Mismatch at index " << i << " (Token "
                                              << h_input[i / dim] << ", Dim " << i % dim << ")";
    }
}