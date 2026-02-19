#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>
#include "layer_test_utils.h"
#include "nanoinfer/op/add.h"

// ===========================================================================
// VecAddLayerTest — 测试 VecAddLayer 的 check() 错误路径 + forward() 正确性
//   check() 错误路径:
//     - 输入为空
//     - 输入尺寸不一致
//     - 输入/输出设备与层设备不匹配
//   forward() 正确性:
//     - CPU: 1.0 + 2.0 = 3.0
//     - CUDA: 同上
//     - 便捷重载 forward(in1, in2, out)
// ===========================================================================
class VecAddLayerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
};

// ---------------------------------------------------------------------------
// check() 错误路径: 输入为空
TEST_F(VecAddLayerTest, CheckFailsOnEmptyInput) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    // 不设置任何 inputs —— 默认槽位是空 Tensor
    tensor::Tensor out = make_cpu_tensor(16, 0.f);
    layer.set_output(0, out);
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 两个输入尺寸不一致
TEST_F(VecAddLayerTest, CheckFailsOnSizeMismatch) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    layer.set_input(0, make_cpu_tensor(16, 1.f));
    layer.set_input(1, make_cpu_tensor(32, 2.f));  // 尺寸不匹配
    layer.set_output(0, make_cpu_tensor(16, 0.f));
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输入设备与层设备不一致
TEST_F(VecAddLayerTest, CheckFailsOnDeviceMismatch) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    // 输入在 CUDA, 但层是 CPU
    tensor::Tensor d_in(base::DataType::kDataTypeFp32, 16, true, gpu_alloc_);
    layer.set_input(0, d_in);
    layer.set_input(1, make_cpu_tensor(16, 1.f));
    layer.set_output(0, make_cpu_tensor(16, 0.f));
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输出尺寸与输入不一致
TEST_F(VecAddLayerTest, CheckFailsOnOutputSizeMismatch) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    layer.set_input(0, make_cpu_tensor(16, 1.f));
    layer.set_input(1, make_cpu_tensor(16, 2.f));
    layer.set_output(0, make_cpu_tensor(8, 0.f));  // 输出尺寸不对
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// forward() CPU: 1.0 + 2.0 == 3.0
TEST_F(VecAddLayerTest, ForwardCPU) {
    const int32_t N = 1024;
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);

    tensor::Tensor in1 = make_cpu_tensor(N, 1.f);
    tensor::Tensor in2 = make_cpu_tensor(N, 2.f);
    tensor::Tensor out = make_cpu_tensor(N, 0.f);

    layer.set_input(0, in1);
    layer.set_input(1, in2);
    layer.set_output(0, out);

    ASSERT_TRUE(layer.forward());

    float* p = layer.get_output(0).ptr<float>();
    for (int32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(p[i], 3.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CUDA: 1.0 + 2.0 == 3.0
TEST_F(VecAddLayerTest, ForwardCUDA) {
    const int32_t N = 1024;
    op::VecAddLayer layer(base::DeviceType::kDeviceCUDA);
    layer.set_cuda_config(make_cuda_config());

    // 准备 CPU 数据并上传
    std::vector<float> h1(N, 1.f), h2(N, 2.f);
    tensor::Tensor d_in1 = make_cuda_tensor(N);
    tensor::Tensor d_in2 = make_cuda_tensor(N);
    tensor::Tensor d_out = make_cuda_tensor(N);
    cudaMemcpy(d_in1.ptr<void>(), h1.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2.ptr<void>(), h2.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    layer.set_input(0, d_in1);
    layer.set_input(1, d_in2);
    layer.set_output(0, d_out);

    ASSERT_TRUE(layer.forward());
    cudaDeviceSynchronize();

    auto result = d2h(layer.get_output(0));
    for (int32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(result[i], 3.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// 便捷重载 forward(in1, in2, out): 自动 set_input 并执行
TEST_F(VecAddLayerTest, ConvenienceForwardCPU) {
    const int32_t N = 256;
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);

    tensor::Tensor in1 = make_cpu_tensor(N, 3.f);
    tensor::Tensor in2 = make_cpu_tensor(N, 7.f);
    tensor::Tensor out = make_cpu_tensor(N, 0.f);

    ASSERT_TRUE(layer.forward(in1, in2, out));

    float* p = out.ptr<float>();
    for (int32_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(p[i], 10.f) << "index " << i;
    }
}
