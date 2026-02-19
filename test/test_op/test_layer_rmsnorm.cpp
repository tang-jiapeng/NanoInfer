#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "layer_test_utils.h"
#include "nanoinfer/op/rmsnorm.h"

// ===========================================================================
// RmsNormLayerTest — 测试 RmsNormLayer:
//   - check() 维度验证 (dim 不匹配 → 报错)
//   - set_weight() 从裸指针加载权重
//   - to_cuda() 权重迁移到 CUDA
//   - forward() CPU / CUDA 数值正确性
//
// 数值验证:
//   如果 input=1.0 (全 1), weight=1.0 (全 1), 则:
//   RMS = sqrt(mean(1^2) + eps) ≈ sqrt(1 + 1e-5) ≈ 1.0
//   output = input / RMS * weight ≈ 1.0
// ===========================================================================
class RmsNormLayerTest : public ::testing::Test {
   protected:
    static constexpr int32_t kDim = 64;

    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // 预先创建全 1 的权重数据
        weight_data_.assign(kDim, 1.0f);
    }

    // 创建已加载权重的 CPU RmsNormLayer
    op::RmsNormLayer make_cpu_layer() {
        op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
        layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);
        return layer;
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
    std::vector<float> weight_data_;
};

// ---------------------------------------------------------------------------
// check() 错误路径: dim 不匹配
TEST_F(RmsNormLayerTest, CheckFailsOnDimMismatch) {
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
    layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    // 输入最后一维是 kDim*2, 与构造时传入的 kDim 不同
    tensor::Tensor input = make_cpu_tensor_2d(4, kDim * 2, 1.f);
    tensor::Tensor output = make_cpu_tensor_2d(4, kDim * 2, 0.f);
    layer.set_input(0, input);
    layer.set_output(0, output);

    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输出尺寸不匹配
TEST_F(RmsNormLayerTest, CheckFailsOnOutputSizeMismatch) {
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
    layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    tensor::Tensor input = make_cpu_tensor_2d(4, kDim, 1.f);
    tensor::Tensor output = make_cpu_tensor_2d(2, kDim, 0.f);  // batch 尺寸不对
    layer.set_input(0, input);
    layer.set_output(0, output);

    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// set_weight 从裸指针加载 → get_weight 可正确读回
TEST_F(RmsNormLayerTest, SetWeightFromRawPtr) {
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
    layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    const tensor::Tensor& w = layer.get_weight(0);
    EXPECT_FALSE(w.is_empty());
    EXPECT_EQ(w.size(), kDim);
    EXPECT_EQ(w.device_type(), base::DeviceType::kDeviceCPU);
    // 值相等
    for (int32_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(w.ptr<float>()[i], 1.0f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CPU: all-one input, all-one weight → output ≈ 1.0
TEST_F(RmsNormLayerTest, ForwardCPU) {
    auto layer = make_cpu_layer();
    const int32_t seq = 8;

    tensor::Tensor input = make_cpu_tensor_2d(seq, kDim, 1.0f);
    tensor::Tensor output = make_cpu_tensor_2d(seq, kDim, 0.0f);
    layer.set_input(0, input);
    layer.set_output(0, output);

    ASSERT_TRUE(layer.forward());

    float* p = output.ptr<float>();
    for (int32_t i = 0; i < seq * kDim; ++i) {
        EXPECT_NEAR(p[i], 1.0f, 1e-4f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// forward() CPU: 验证 RMSNorm 归一化效果 (非均匀输入)
// 输入 = [1,2,3,...,dim], weight = 全 1
// 手工验证: RMS = sqrt(mean(i^2) + eps), output_i = i / RMS
TEST_F(RmsNormLayerTest, ForwardCPUNumericalCheck) {
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
    layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    // 构造输入 [1, kDim]
    tensor::Tensor input = make_cpu_tensor_2d(1, kDim, 0.f);
    tensor::Tensor output = make_cpu_tensor_2d(1, kDim, 0.f);
    float* in_ptr = input.ptr<float>();
    for (int32_t i = 0; i < kDim; ++i) in_ptr[i] = float(i + 1);
    layer.set_input(0, input);
    layer.set_output(0, output);
    ASSERT_TRUE(layer.forward());

    // 手工计算期望值
    double sum_sq = 0.0;
    for (int32_t i = 0; i < kDim; ++i) sum_sq += (i + 1.0) * (i + 1.0);
    double rms = std::sqrt(sum_sq / kDim + 1e-5);

    float* out_ptr = output.ptr<float>();
    for (int32_t i = 0; i < kDim; ++i) {
        float expected = float((i + 1.0) / rms);
        EXPECT_NEAR(out_ptr[i], expected, 1e-4f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// to_cuda() 后权重在 CUDA 侧, forward() CUDA 也正确
TEST_F(RmsNormLayerTest, ToCudaAndForwardCUDA) {
    op::RmsNormLayer layer(base::DeviceType::kDeviceCPU, kDim);
    layer.set_weight(0, {kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    // 迁移整层到 CUDA
    layer.set_device_type(base::DeviceType::kDeviceCUDA);
    layer.to_cuda();
    layer.set_cuda_config(make_cuda_config());

    // 验证权重已在 CUDA
    EXPECT_EQ(layer.get_weight(0).device_type(), base::DeviceType::kDeviceCUDA);

    const int32_t seq = 4;
    tensor::Tensor d_input = make_cuda_tensor_2d(seq, kDim);
    tensor::Tensor d_output = make_cuda_tensor_2d(seq, kDim);

    // 填充输入 = 1.0
    std::vector<float> h_in(seq * kDim, 1.0f);
    cudaMemcpy(d_input.ptr<void>(), h_in.data(), h_in.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    layer.set_input(0, d_input);
    layer.set_output(0, d_output);
    ASSERT_TRUE(layer.forward());
    cudaDeviceSynchronize();

    auto result = d2h(d_output);
    for (int32_t i = 0; i < seq * kDim; ++i) {
        EXPECT_NEAR(result[i], 1.0f, 1e-4f) << "index " << i;
    }
}
