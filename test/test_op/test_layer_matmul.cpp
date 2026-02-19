#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>
#include "layer_test_utils.h"
#include "nanoinfer/op/matmul.h"

// ===========================================================================
// MatmulLayerTest — 测试 MatmulLayer:
//   check() 错误路径:
//     - 输入为空
//     - 输入最后一维 (K dim) 不匹配
//     - 输出最后一维 (N dim) 不匹配
//   set_weight 从裸指针加载
//   forward() CPU / CUDA 数值正确性
//
// 维度约定 (见 matmul.h + check()):
//   MatmulLayer(device, dim0, dim1) — Weight shape: [dim0, dim1]
//   Input  shape: [batch, dim1]   (最后一维 = K = dim1)
//   Output shape: [batch, dim0]   (最后一维 = N = dim0)
//   计算:  Y[b, n] = sum_k(X[b, k] * W[n, k])  (W 转置)
//
// 简单数值验证:
//   dim0=4 (out), dim1=8 (in), batch=2
//   X = all 1.0  → Y[b,n] = sum_k(1*1) = dim1 = 8.0 (当 W 全 1 时)
// ===========================================================================
class MatmulLayerTest : public ::testing::Test {
   protected:
    static constexpr int32_t kDim0 = 4;  // 输出维度
    static constexpr int32_t kDim1 = 8;  // 输入维度 (K)
    static constexpr int32_t kBatch = 2;

    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
        // Weight: [dim0, dim1] = [4, 8], 全 1
        weight_data_.assign(kDim0 * kDim1, 1.0f);
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
    std::vector<float> weight_data_;
};

// ---------------------------------------------------------------------------
// check() 错误路径: 输入为空
TEST_F(MatmulLayerTest, CheckFailsOnEmptyInput) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);
    layer.set_output(0, make_cpu_tensor_2d(kBatch, kDim0, 0.f));
    // 不设 input
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输入 K 维不匹配
TEST_F(MatmulLayerTest, CheckFailsOnInputKDimMismatch) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);
    layer.set_input(0, make_cpu_tensor_2d(kBatch, kDim1 + 1, 1.f));  // K 不匹配
    layer.set_output(0, make_cpu_tensor_2d(kBatch, kDim0, 0.f));
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输出 N 维不匹配
TEST_F(MatmulLayerTest, CheckFailsOnOutputNDimMismatch) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);
    layer.set_input(0, make_cpu_tensor_2d(kBatch, kDim1, 1.f));
    layer.set_output(0, make_cpu_tensor_2d(kBatch, kDim0 + 1, 0.f));  // N 不匹配
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// set_weight 从裸指针加载
TEST_F(MatmulLayerTest, SetWeightFromRawPtr) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    const tensor::Tensor& w = layer.get_weight(0);
    EXPECT_FALSE(w.is_empty());
    EXPECT_EQ(w.size(), kDim0 * kDim1);
    EXPECT_EQ(w.get_dim(0), kDim0);
    EXPECT_EQ(w.get_dim(1), kDim1);
}

// ---------------------------------------------------------------------------
// forward() CPU: X=all1, W=all1 → Y[b,n] = kDim1 = 8.0
TEST_F(MatmulLayerTest, ForwardCPU) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    tensor::Tensor input = make_cpu_tensor_2d(kBatch, kDim1, 1.0f);
    tensor::Tensor output = make_cpu_tensor_2d(kBatch, kDim0, 0.0f);
    layer.set_input(0, input);
    layer.set_output(0, output);

    ASSERT_TRUE(layer.forward());

    float expected = static_cast<float>(kDim1);  // 每个输出元素 = sum over K of (1*1)
    float* p = output.ptr<float>();
    for (int32_t i = 0; i < kBatch * kDim0; ++i) {
        EXPECT_NEAR(p[i], expected, 1e-3f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CUDA: 同上数值
TEST_F(MatmulLayerTest, ForwardCUDA) {
    op::MatmulLayer layer(base::DeviceType::kDeviceCPU, kDim0, kDim1);
    layer.set_weight(0, {kDim0, kDim1}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    layer.set_device_type(base::DeviceType::kDeviceCUDA);
    auto cfg = make_cuda_config();
    layer.to_cuda();
    layer.set_cuda_config(cfg);

    std::vector<float> h_input(kBatch * kDim1, 1.0f);
    tensor::Tensor d_input = make_cuda_tensor_2d(kBatch, kDim1);
    tensor::Tensor d_output = make_cuda_tensor_2d(kBatch, kDim0);
    cudaMemcpy(d_input.ptr<void>(), h_input.data(), h_input.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    layer.set_input(0, d_input);
    layer.set_output(0, d_output);
    ASSERT_TRUE(layer.forward());
    cudaDeviceSynchronize();

    float expected = static_cast<float>(kDim1);
    auto result = d2h(d_output);
    for (int32_t i = 0; i < kBatch * kDim0; ++i) {
        EXPECT_NEAR(result[i], expected, 1e-2f) << "index " << i;
    }
}
