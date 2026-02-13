#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"
#include "../src/op/kernels/cpu/argmax_kernel.h"
#include "../src/op/kernels/cuda/argmax_kernel.cuh"


class ArgmaxTest : public ::testing::Test {
   protected:
    void SetUp() override {
        allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

TEST_F(ArgmaxTest, BatchedArgmax) {

    int32_t batch_size = 4;
    int32_t vocab_size = 10000;  // 模拟较真实的 Vocab Size

    // 1. 准备数据 (Host)
    std::vector<float> h_input(batch_size * vocab_size);
    std::vector<int32_t> expected_indices(batch_size);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 填充随机数
    for (auto& v : h_input) v = dis(gen);

    // 手动设置最大值，确保我们知道答案
    // Row 0: Max at index 100
    h_input[0 * vocab_size + 100] = 10.0f;
    expected_indices[0] = 100;
    // Row 1: Max at index 5000
    h_input[1 * vocab_size + 5000] = 10.0f;
    expected_indices[1] = 5000;
    // Row 2: Max at index 9999
    h_input[2 * vocab_size + 9999] = 10.0f;
    expected_indices[2] = 9999;
    // Row 3: Max at index 0
    h_input[3 * vocab_size + 0] = 10.0f;
    expected_indices[3] = 0;

    // 2. GPU Tensors
    tensor::Tensor t_in(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, allocator_);
    tensor::Tensor t_out(base::DataType::kDataTypeInt32, batch_size, true,
                         allocator_);  // Output is [batch]

    cudaMemcpy(t_in.ptr<void>(), h_input.data(), h_input.size() * 4, cudaMemcpyHostToDevice);

    // 3. Run Kernel
    kernel::argmax_kernel_cu(t_in, t_out, nullptr);
    cudaDeviceSynchronize();

    // 4. Verify
    std::vector<int32_t> h_out(batch_size);
    cudaMemcpy(h_out.data(), t_out.ptr<void>(), batch_size * 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(h_out[i], expected_indices[i]) << "Mismatch at batch " << i;
    }
}