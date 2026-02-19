#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"


class AddKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 获取 CUDA 分配器
        allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
        cpu_allocator_ = base::CPUDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> allocator_;
    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
};

TEST_F(AddKernelTest, BasicAddFP32) {
    int32_t size = 1024;  // 测试向量长度

    // 1. 准备 CPU 数据
    std::vector<float> h_in1(size, 1.0f);  // 全是 1.0
    std::vector<float> h_in2(size, 2.0f);  // 全是 2.0
    std::vector<float> h_out(size, 0.0f);

    // 2. 准备 GPU Tensor
    tensor::Tensor d_in1(base::DataType::kDataTypeFp32, size, true, allocator_);
    tensor::Tensor d_in2(base::DataType::kDataTypeFp32, size, true, allocator_);
    tensor::Tensor d_out(base::DataType::kDataTypeFp32, size, true, allocator_);

    // 3. 数据拷贝 Host -> Device
    cudaMemcpy(d_in1.ptr<void>(), h_in1.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2.ptr<void>(), h_in2.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // 4. 调用 Kernel (使用默认流 nullptr)
    auto add_cu = kernel::KernelRegistry::instance().get<kernel::AddKernelFn>(
        "add", base::DeviceType::kDeviceCUDA);
    add_cu(d_in1, d_in2, d_out, nullptr);

    // 5. 数据拷贝 Device -> Host
    cudaMemcpy(h_out.data(), d_out.ptr<void>(), size * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. 验证结果
    for (int i = 0; i < size; ++i) {
        // 1.0 + 2.0 应该等于 3.0
        EXPECT_FLOAT_EQ(h_out[i], 3.0f) << "Error at index " << i;
    }
}

TEST_F(AddKernelTest, LargeBatchAdd) {
    // 模拟 Batched 场景: [Batch=4, Hidden=4096] -> Flatten size = 16384
    int32_t size = 4 * 4096;

    tensor::Tensor d_in1(base::DataType::kDataTypeFp32, size, true, allocator_);
    tensor::Tensor d_in2(base::DataType::kDataTypeFp32, size, true, allocator_);
    tensor::Tensor d_out(base::DataType::kDataTypeFp32, size, true, allocator_);

    // 初始化 GPU 数据 (利用 cudaMemset 简单测试)
    // 这里的 float 设置比较 trick，但在 bit 层面 0x00 是 0.0f
    cudaMemset(d_in1.ptr<void>(), 0, size * sizeof(float));
    cudaMemset(d_in2.ptr<void>(), 0, size * sizeof(float));

    // 我们手动拷几个值进去验证
    float val1 = 10.5f;
    float val2 = 20.5f;
    cudaMemcpy(d_in1.ptr<float>() + 100, &val1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2.ptr<float>() + 100, &val2, sizeof(float), cudaMemcpyHostToDevice);

    auto add_cu = kernel::KernelRegistry::instance().get<kernel::AddKernelFn>(
        "add", base::DeviceType::kDeviceCUDA);
    add_cu(d_in1, d_in2, d_out, nullptr);

    float res;
    cudaMemcpy(&res, d_out.ptr<float>() + 100, sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(res, 31.0f);
}