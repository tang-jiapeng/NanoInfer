#include <gtest/gtest.h>
#include "layer_test_utils.h"
#include "nanoinfer/op/add.h"
#include "nanoinfer/op/layer.h"
#include "nanoinfer/op/swiglu.h"

// ===========================================================================
// LayerBaseTest — 测试 BaseLayer / Layer / LayerParam 通用基础行为:
//   - 元信息 (device_type, layer_type, layer_name, data_type)
//   - 输入输出槽位管理 (reset/set/get/size)
//   - 权重槽位管理 (LayerParam)
//   - init() 返回 Success
// 使用 VecAddLayer / SwiGLULayer 作为轻量实例
// ===========================================================================
class LayerBaseTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
};

// ---------------------------------------------------------------------------
// 元信息
TEST_F(LayerBaseTest, MetadataCPU) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    EXPECT_EQ(layer.device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(layer.layer_type(), op::LayerType::kLayerAdd);
    EXPECT_EQ(layer.data_type(), base::DataType::kDataTypeFp32);
    EXPECT_EQ(layer.get_layer_name(), "Add");
}

TEST_F(LayerBaseTest, SetLayerName) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    layer.set_layer_name("MyAdd");
    EXPECT_EQ(layer.get_layer_name(), "MyAdd");
}

TEST_F(LayerBaseTest, SetDeviceType) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    layer.set_device_type(base::DeviceType::kDeviceCUDA);
    EXPECT_EQ(layer.device_type(), base::DeviceType::kDeviceCUDA);
}

// ---------------------------------------------------------------------------
// 输入输出槽位数量
TEST_F(LayerBaseTest, AddLayerSlotCount) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    EXPECT_EQ(layer.input_size(), 2u);
    EXPECT_EQ(layer.output_size(), 1u);
}

TEST_F(LayerBaseTest, SwiGLULayerSlotCount) {
    op::SwiGLULayer layer(base::DeviceType::kDeviceCPU, 32);
    EXPECT_EQ(layer.input_size(), 2u);
    EXPECT_EQ(layer.output_size(), 1u);
}

// ---------------------------------------------------------------------------
// set_input / get_input —— 设置后可正确读回
TEST_F(LayerBaseTest, SetGetInput) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    tensor::Tensor t = make_cpu_tensor_2d(4, 8, 1.f);
    layer.set_input(0, t);

    const tensor::Tensor& got = layer.get_input(0);
    EXPECT_EQ(got.size(), 4 * 8);
    EXPECT_FALSE(got.is_empty());
    EXPECT_FLOAT_EQ(*got.ptr<float>(), 1.f);
}

// ---------------------------------------------------------------------------
// set_output / get_output
TEST_F(LayerBaseTest, SetGetOutput) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    tensor::Tensor t = make_cpu_tensor(16, 0.f);
    layer.set_output(0, t);

    tensor::Tensor& got = layer.get_output(0);
    EXPECT_EQ(got.size(), 16);
}

// ---------------------------------------------------------------------------
// init() 应该成功
TEST_F(LayerBaseTest, InitReturnsSuccess) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCPU);
    EXPECT_TRUE(layer.init());
}

// ---------------------------------------------------------------------------
// cuda_config 默认为空, 设置后可读回
TEST_F(LayerBaseTest, CudaConfigSetGet) {
    op::VecAddLayer layer(base::DeviceType::kDeviceCUDA);
    EXPECT_EQ(layer.cuda_config(), nullptr);

    auto cfg = make_cuda_config();
    layer.set_cuda_config(cfg);
    EXPECT_NE(layer.cuda_config(), nullptr);
    EXPECT_EQ(layer.cuda_config().get(), cfg.get());
}
