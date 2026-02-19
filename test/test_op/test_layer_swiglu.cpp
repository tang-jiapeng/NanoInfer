#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "layer_test_utils.h"
#include "nanoinfer/op/swiglu.h"

// ===========================================================================
// SwiGLULayerTest — 测试 SwiGLULayer:
//   check() 错误路径:
//     - 输入为空
//     - 输入尺寸不一致
//     - 输出尺寸不一致
//   forward() 数值正确性:
//     SwiGLU(gate, value) = SiLU(gate) * value
//     SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//     测试 gate=1.0, value=1.0:
//       SiLU(1.0) = 1.0 / (1+e^{-1}) ≈ 0.7310586
//       output = 0.7310586 * 1.0 = 0.7310586
// ===========================================================================
class SwiGLULayerTest : public ::testing::Test {
   protected:
    static constexpr int32_t kHiddenDim = 256;

    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
};

// ---------------------------------------------------------------------------
// 元信息
TEST_F(SwiGLULayerTest, Metadata) {
    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, kHiddenDim);
    EXPECT_EQ(layer.layer_type(), op::LayerType::kLayerSwiGLU);
    EXPECT_EQ(layer.input_size(), 2u);
    EXPECT_EQ(layer.output_size(), 1u);
}

// ---------------------------------------------------------------------------
// check() 错误路径: gate 和 value 尺寸不一致
TEST_F(SwiGLULayerTest, CheckFailsOnInputSizeMismatch) {
    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, kHiddenDim);
    layer.set_input(0, make_cpu_tensor(kHiddenDim, 1.f));
    layer.set_input(1, make_cpu_tensor(kHiddenDim / 2, 1.f));  // 不一致
    layer.set_output(0, make_cpu_tensor(kHiddenDim, 0.f));
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输出尺寸不匹配
TEST_F(SwiGLULayerTest, CheckFailsOnOutputSizeMismatch) {
    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, kHiddenDim);
    layer.set_input(0, make_cpu_tensor(kHiddenDim, 1.f));
    layer.set_input(1, make_cpu_tensor(kHiddenDim, 1.f));
    layer.set_output(0, make_cpu_tensor(kHiddenDim / 2, 0.f));  // 错误
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// forward() CPU: gate=1.0, value=1.0 → SiLU(1.0)*1.0 ≈ 0.7310586
TEST_F(SwiGLULayerTest, ForwardCPU) {
    const float kGate = 1.0f;
    const float kValue = 1.0f;
    const float kExpected = kGate / (1.f + std::exp(-kGate)) * kValue;

    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, kHiddenDim);

    tensor::Tensor gate = make_cpu_tensor(kHiddenDim, kGate);
    tensor::Tensor value = make_cpu_tensor(kHiddenDim, kValue);
    tensor::Tensor output = make_cpu_tensor(kHiddenDim, 0.f);

    layer.set_input(0, gate);
    layer.set_input(1, value);
    layer.set_output(0, output);

    ASSERT_TRUE(layer.forward());

    float* p = output.ptr<float>();
    for (int32_t i = 0; i < kHiddenDim; ++i) {
        EXPECT_NEAR(p[i], kExpected, 1e-5f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CPU: gate=0.0, value=5.0 → SiLU(0)*5 = 0.5*5 = 2.5
TEST_F(SwiGLULayerTest, ForwardCPUGateZero) {
    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, kHiddenDim);

    tensor::Tensor gate = make_cpu_tensor(kHiddenDim, 0.f);
    tensor::Tensor value = make_cpu_tensor(kHiddenDim, 5.f);
    tensor::Tensor output = make_cpu_tensor(kHiddenDim, 0.f);

    layer.set_input(0, gate);
    layer.set_input(1, value);
    layer.set_output(0, output);
    ASSERT_TRUE(layer.forward());

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    float* p = output.ptr<float>();
    for (int32_t i = 0; i < kHiddenDim; ++i) {
        EXPECT_NEAR(p[i], 0.f, 1e-5f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CUDA: gate=1.0, value=1.0 → ≈ 0.7310586
TEST_F(SwiGLULayerTest, ForwardCUDA) {
    const float kGate = 1.0f;
    const float kValue = 1.0f;
    const float kExpected = kGate / (1.f + std::exp(-kGate)) * kValue;

    op::SwiGLULayer layer(base::DeviceType::kDeviceCUDA, kHiddenDim);
    layer.set_cuda_config(make_cuda_config());

    std::vector<float> h_gate(kHiddenDim, kGate), h_value(kHiddenDim, kValue);
    tensor::Tensor d_gate = make_cuda_tensor(kHiddenDim);
    tensor::Tensor d_value = make_cuda_tensor(kHiddenDim);
    tensor::Tensor d_output = make_cuda_tensor(kHiddenDim);

    cudaMemcpy(d_gate.ptr<void>(), h_gate.data(), kHiddenDim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_value.ptr<void>(), h_value.data(), kHiddenDim * sizeof(float),
               cudaMemcpyHostToDevice);

    layer.set_input(0, d_gate);
    layer.set_input(1, d_value);
    layer.set_output(0, d_output);

    ASSERT_TRUE(layer.forward());
    cudaDeviceSynchronize();

    auto result = d2h(d_output);
    for (int32_t i = 0; i < kHiddenDim; ++i) {
        EXPECT_NEAR(result[i], kExpected, 1e-5f) << "index " << i;
    }
}
