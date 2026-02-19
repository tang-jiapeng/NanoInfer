/**
 * @file alloc.cpp
 * @brief 设备内存操作的统一调度实现（memcpy / memset）
 *
 * DeviceAllocator 基类提供了跨设备的内存拷贝与清零操作：
 *   - memcpy：根据 MemcpyKind 枚举分发到 CPU memcpy 或 CUDA cudaMemcpy/cudaMemcpyAsync
 *   - memset_zero：根据设备类型调用 std::memset 或 cudaMemset
 *
 * 具体的 allocate / release 接口由子类 CPUDeviceAllocator 和
 * CUDADeviceAllocator 分别在 alloc_cpu.cpp 和 alloc_cuda.cpp 中实现。
 */
#include "nanoinfer/base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {

/**
 * @brief 跨设备内存拷贝统一入口
 *
 * 根据 MemcpyKind 枚举分发到对应后端：
 *   - CPU→CPU : std::memcpy
 *   - CPU↔CUDA: cudaMemcpy / cudaMemcpyAsync（取决于是否提供 stream）
 *   - CUDA→CUDA: cudaMemcpy / cudaMemcpyAsync
 *
 * @param need_sync 若为 true，拷贝完成后调用 cudaDeviceSynchronize 强制同步
 */
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (!byte_size) {
        return;
    }

    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<CUstream_st*>(stream);
    }
    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        if (!stream_) {
            cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
        }
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

/**
 * @brief 内存清零（支持 CPU 与 CUDA）
 *
 * CPU 端调用 std::memset，CUDA 端调用 cudaMemset / cudaMemsetAsync。
 */
void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
    CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }
        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }
}

}  // namespace base