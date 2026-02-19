#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <nanoinfer/tensor/tensor.h>
#include <cstring>
#include "../utils.cuh"
#include "nanoinfer/base/buffer.h"


// ===========================================================================
// TensorTest — 测试 tensor::Tensor 的创建、拷贝、Clone、Assign 等功能
// ===========================================================================
class TensorTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
};

// ---------------------------------------------------------------------------
// GPU -> CPU 拷贝
TEST_F(TensorTest, ToCPU) {
    tensor::Tensor t1_cu(base::DataType::kDataTypeFp32, 32, 32, true, gpu_alloc_);
    ASSERT_FALSE(t1_cu.is_empty());
    set_value_cu(t1_cu.ptr<float>(), 32 * 32);

    t1_cu.to_cpu();
    ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
    float* cpu_ptr = t1_cu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_FLOAT_EQ(cpu_ptr[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU -> CUDA 拷贝
TEST_F(TensorTest, ToCUDA) {
    tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 32, 32, true, cpu_alloc_);
    ASSERT_FALSE(t1_cpu.is_empty());
    float* p1 = t1_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) p1[i] = 1.f;

    t1_cpu.to_cuda();
    ASSERT_EQ(t1_cpu.device_type(), base::DeviceType::kDeviceCUDA);

    std::vector<float> result(32 * 32, 0.f);
    cudaMemcpy(result.data(), t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_FLOAT_EQ(result[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// Clone GPU Tensor
TEST_F(TensorTest, CloneCUDA) {
    tensor::Tensor t1_cu(base::DataType::kDataTypeFp32, 32, 32, true, gpu_alloc_);
    ASSERT_FALSE(t1_cu.is_empty());
    set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

    tensor::Tensor t2_cu = t1_cu.clone();
    ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
    ASSERT_EQ(t2_cu.size(), 32 * 32);
    ASSERT_NE(t2_cu.ptr<float>(), t1_cu.ptr<float>());  // 不应共享内存

    std::vector<float> buf(32 * 32, 0.f);
    cudaMemcpy(buf.data(), t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_FLOAT_EQ(buf[i], 1.f) << "index " << i;
    }

    // Clone 后 to_cpu
    t2_cu.to_cpu();
    float* p = t2_cu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_FLOAT_EQ(p[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// Clone CPU Tensor
TEST_F(TensorTest, CloneCPU) {
    tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 32, 32, true, cpu_alloc_);
    ASSERT_FALSE(t1_cpu.is_empty());
    for (int i = 0; i < 32 * 32; ++i) t1_cpu.index<float>(i) = 1.f;

    tensor::Tensor t2_cpu = t1_cpu.clone();
    ASSERT_NE(t2_cpu.ptr<float>(), t1_cpu.ptr<float>());  // 应深拷贝
    float* p2 = t2_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_FLOAT_EQ(p2[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// 2D 形状的 Tensor 初始化
TEST_F(TensorTest, Init2D) {
    int32_t rows = 32, cols = 151;
    tensor::Tensor t1(base::DataType::kDataTypeFp32, rows, cols, true, cpu_alloc_);
    ASSERT_FALSE(t1.is_empty());
    ASSERT_EQ(t1.size(), rows * cols);
}

// ---------------------------------------------------------------------------
// is_empty 标志测试
TEST_F(TensorTest, IsEmptyWhenNotAllocated) {
    int32_t size = 32 * 151;
    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, cpu_alloc_);
    ASSERT_TRUE(t1.is_empty());  // alloc_memory=false 应为空
}

// ---------------------------------------------------------------------------
// 外部指针模式 (不拥有内存所有权)
TEST_F(TensorTest, ExternalPointer) {
    float* ptr = new float[32];
    ptr[0] = 31.0f;
    tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
    ASSERT_FALSE(t1.is_empty());
    ASSERT_EQ(t1.ptr<float>(), ptr);  // 应直接使用外部指针
    ASSERT_FLOAT_EQ(*t1.ptr<float>(), 31.0f);
    delete[] ptr;
}

// ---------------------------------------------------------------------------
// Assign 外部 Buffer
TEST_F(TensorTest, AssignBuffer) {
    using namespace base;
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, cpu_alloc_);
    ASSERT_FALSE(t1_cpu.is_empty());

    int32_t size = 32 * 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; ++i) ptr[i] = float(i);

    auto buffer = std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::kDeviceCPU);

    ASSERT_TRUE(t1_cpu.assign(buffer));
    ASSERT_FALSE(t1_cpu.is_empty());
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
    delete[] ptr;
}

// ---------------------------------------------------------------------------
// 业务场景: Int32 Tensor
TEST_F(TensorTest, DataTypeInt32) {
    const int32_t n = 64;
    tensor::Tensor t(base::DataType::kDataTypeInt32, n, true, cpu_alloc_);
    ASSERT_FALSE(t.is_empty());
    ASSERT_EQ(t.data_type(), base::DataType::kDataTypeInt32);
    int32_t* p = t.ptr<int32_t>();
    for (int i = 0; i < n; ++i) p[i] = i;
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(t.index<int32_t>(i), i) << "index " << i;
    }
}
