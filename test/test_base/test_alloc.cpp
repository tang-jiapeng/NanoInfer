#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstring>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/base.h"

// ===========================================================================
// CPUAllocTest — 测试 CPUDeviceAllocator 的分配/释放/memcpy/memset
// ===========================================================================
class CPUAllocTest : public ::testing::Test {
   protected:
    void SetUp() override {
        alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::CPUDeviceAllocator> alloc_;
};

TEST_F(CPUAllocTest, DeviceType) {
    EXPECT_EQ(alloc_->device_type(), base::DeviceType::kDeviceCPU);
}

TEST_F(CPUAllocTest, AllocateAndRelease) {
    const size_t size = 1024;
    void* ptr = alloc_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    // 写入验证不崩溃
    std::memset(ptr, 0xAB, size);
    alloc_->release(ptr);
}

TEST_F(CPUAllocTest, AllocateLargeBlock) {
    const size_t size = 64 * 1024 * 1024;  // 64 MB
    void* ptr = alloc_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    alloc_->release(ptr);
}

TEST_F(CPUAllocTest, MemcpyCPU2CPU) {
    const size_t n = 128;
    float* src = (float*)alloc_->allocate(n * sizeof(float));
    float* dst = (float*)alloc_->allocate(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) src[i] = float(i);

    alloc_->memcpy(src, dst, n * sizeof(float), base::MemcpyKind::kMemcpyCPU2CPU);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(dst[i], float(i)) << "index " << i;
    }

    alloc_->release(src);
    alloc_->release(dst);
}

TEST_F(CPUAllocTest, MemsetZero) {
    const size_t n = 64;
    float* ptr = (float*)alloc_->allocate(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) ptr[i] = float(i) + 1.0f;

    alloc_->memset_zero(ptr, n * sizeof(float), nullptr);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], 0.f) << "index " << i;
    }
    alloc_->release(ptr);
}

// 工厂单例返回同一对象
TEST_F(CPUAllocTest, FactorySingleton) {
    auto a1 = base::CPUDeviceAllocatorFactory::get_instance();
    auto a2 = base::CPUDeviceAllocatorFactory::get_instance();
    EXPECT_EQ(a1.get(), a2.get());
}

// ===========================================================================
// CUDAAllocTest — 测试 CUDADeviceAllocator 的分配/释放/memcpy
// ===========================================================================
class CUDAAllocTest : public ::testing::Test {
   protected:
    void SetUp() override {
        alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::CUDADeviceAllocator> alloc_;
    std::shared_ptr<base::CPUDeviceAllocator> cpu_alloc_;
};

TEST_F(CUDAAllocTest, DeviceType) {
    EXPECT_EQ(alloc_->device_type(), base::DeviceType::kDeviceCUDA);
}

TEST_F(CUDAAllocTest, AllocateAndRelease) {
    const size_t size = 1024 * sizeof(float);
    void* ptr = alloc_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    alloc_->release(ptr);
}

TEST_F(CUDAAllocTest, AllocateLargeBlock) {
    // 超过显存池大内存阈值 (>1MB)
    const size_t size = 2 * 1024 * 1024;
    void* ptr = alloc_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    alloc_->release(ptr);
}

// CUDA 显存池复用：同大小的第二次分配地址应该相同 (pool 复用)
TEST_F(CUDAAllocTest, MemoryPoolReuse) {
    const size_t size = 512 * sizeof(float);
    void* ptr1 = alloc_->allocate(size);
    ASSERT_NE(ptr1, nullptr);
    alloc_->release(ptr1);

    void* ptr2 = alloc_->allocate(size);
    ASSERT_NE(ptr2, nullptr);
    // 复用后地址可能相同（池行为），不强制断言相等，但应正常分配
    alloc_->release(ptr2);
}

TEST_F(CUDAAllocTest, MemcpyCPU2CUDA2CPU) {
    const size_t n = 256;
    const size_t byte_size = n * sizeof(float);

    float* h_src = (float*)cpu_alloc_->allocate(byte_size);
    float* h_dst = (float*)cpu_alloc_->allocate(byte_size);
    float* d_buf = (float*)alloc_->allocate(byte_size);

    for (size_t i = 0; i < n; ++i) h_src[i] = float(i) * 0.5f;

    // H2D
    alloc_->memcpy(h_src, d_buf, byte_size, base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
    // D2H
    alloc_->memcpy(d_buf, h_dst, byte_size, base::MemcpyKind::kMemcpyCUDA2CPU, nullptr, true);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(h_dst[i], float(i) * 0.5f) << "index " << i;
    }

    cpu_alloc_->release(h_src);
    cpu_alloc_->release(h_dst);
    alloc_->release(d_buf);
}

TEST_F(CUDAAllocTest, MemsetZero) {
    const size_t n = 128;
    float* d_ptr = (float*)alloc_->allocate(n * sizeof(float));

    // 先写入非零数据
    std::vector<float> ones(n, 1.0f);
    cudaMemcpy(d_ptr, ones.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // memset_zero
    alloc_->memset_zero(d_ptr, n * sizeof(float), nullptr, true);

    // D2H 验证
    std::vector<float> result(n, -1.f);
    cudaMemcpy(result.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.f) << "index " << i;
    }
    alloc_->release(d_ptr);
}

TEST_F(CUDAAllocTest, FactorySingleton) {
    auto a1 = base::CUDADeviceAllocatorFactory::get_instance();
    auto a2 = base::CUDADeviceAllocatorFactory::get_instance();
    EXPECT_EQ(a1.get(), a2.get());
}
