#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../utils.cuh"
#include "nanoinfer/base/buffer.h"

// ===========================================================================
// BufferTest — 测试 base::Buffer 的分配、外部指针、异构 memcpy
// ===========================================================================
class BufferTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
};

// ---------------------------------------------------------------------------
// 基本 CPU 分配
TEST_F(BufferTest, AllocateCPU) {
    base::Buffer buffer(32, cpu_alloc_);
    ASSERT_NE(buffer.ptr(), nullptr);
    EXPECT_EQ(buffer.device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(buffer.byte_size(), 32u);
}

// ---------------------------------------------------------------------------
// 外部指针 (不拥有内存)
TEST_F(BufferTest, ExternalPointer) {
    float* ptr = new float[32];
    base::Buffer buffer(32, nullptr, ptr, true);
    EXPECT_TRUE(buffer.is_external());
    EXPECT_EQ(buffer.ptr(), ptr);
    delete[] ptr;
}

// ---------------------------------------------------------------------------
// CPU -> CUDA memcpy
TEST_F(BufferTest, MemcpyCPU2CUDA) {
    const int32_t size = 32;
    std::vector<float> src(size);
    for (int i = 0; i < size; ++i) src[i] = float(i);

    base::Buffer cpu_buf(size * sizeof(float), nullptr, src.data(), true);
    cpu_buf.set_device_type(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(cpu_buf.is_external());

    base::Buffer cu_buf(size * sizeof(float), gpu_alloc_);
    cu_buf.copy_from(cpu_buf);
    EXPECT_EQ(cu_buf.device_type(), base::DeviceType::kDeviceCUDA);

    std::vector<float> result(size, 0.f);
    cudaMemcpy(result.data(), cu_buf.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], float(i)) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// CUDA -> CUDA memcpy
TEST_F(BufferTest, MemcpyCUDA2CUDA) {
    const int32_t size = 32;
    base::Buffer cu_buf1(size * sizeof(float), gpu_alloc_);
    base::Buffer cu_buf2(size * sizeof(float), gpu_alloc_);

    set_value_cu((float*)cu_buf2.ptr(), size);
    cu_buf1.copy_from(cu_buf2);  // D2D

    std::vector<float> result(size, 0.f);
    cudaMemcpy(result.data(), cu_buf1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// CUDA -> CPU memcpy
TEST_F(BufferTest, MemcpyCUDA2CPU) {
    const int32_t size = 32;
    base::Buffer cu_buf(size * sizeof(float), gpu_alloc_);
    base::Buffer cpu_buf(size * sizeof(float), cpu_alloc_);

    set_value_cu((float*)cu_buf.ptr(), size);
    cpu_buf.copy_from(cu_buf);  // D2H

    float* p = (float*)cpu_buf.ptr();
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(p[i], 1.f) << "index " << i;
    }
}

// ---------------------------------------------------------------------------
// CPU -> CPU memcpy
TEST_F(BufferTest, MemcpyCPU2CPU) {
    const int32_t size = 32;
    std::vector<float> src(size);
    for (int i = 0; i < size; ++i) src[i] = float(i) * 2.f;

    base::Buffer src_buf(size * sizeof(float), nullptr, src.data(), true);
    src_buf.set_device_type(base::DeviceType::kDeviceCPU);

    base::Buffer dst_buf(size * sizeof(float), cpu_alloc_);
    dst_buf.copy_from(src_buf);

    float* p = (float*)dst_buf.ptr();
    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(p[i], float(i) * 2.f) << "index " << i;
    }
}
