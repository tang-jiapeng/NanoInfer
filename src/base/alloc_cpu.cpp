/**
 * @file alloc_cpu.cpp
 * @brief CPU 内存分配器实现
 *
 * CPUDeviceAllocator 使用 posix_memalign 进行对齐分配：
 *   - 大于等于 1KB 的分配使用 32 字节对齐（适配 AVX/NEON 向量化）
 *   - 小于 1KB 的分配使用 16 字节对齐（SSE 级别）
 *
 * 通过 CPUDeviceAllocatorFactory 单例工厂获取全局唯一实例。
 */
#include <glog/logging.h>
#include <cstdlib>
#include "nanoinfer/base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define NANOINFER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

/**
 * @brief CPU 内存分配（posix_memalign 对齐分配）
 *
 * 对齐策略：≥ 1KB 时 32 字节对齐（AVX），< 1KB 时 16 字节对齐（SSE）。
 * 若平台不支持 posix_memalign 则回退为 malloc。
 */
void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) {
        return nullptr;
    }
#ifdef NANOINFER_HAVE_POSIX_MEMALIGN
    void* data = nullptr;
    const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
    int status = posix_memalign(
        (void**)&data, ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)), byte_size);
    if (status != 0) {
        return nullptr;
    }
    return data;
#else
    void* data = malloc(byte_size);
    return data;
#endif
}

/** @brief 释放 CPU 内存（直接调用 free） */
void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr) {
        free(ptr);
    }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base