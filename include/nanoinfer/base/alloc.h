/**
 * @file alloc.h
 * @brief 设备内存分配器接口及 CPU / CUDA 实现
 */
#ifndef NANO_INFER_ALLOC_H
#define NANO_INFER_ALLOC_H
#include <map>
#include <memory>
#include <vector>
#include "base.h"

namespace base {

/// @brief 内存拷贝方向
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,    ///< Host → Host
    kMemcpyCPU2CUDA = 1,   ///< Host → Device (H2D)
    kMemcpyCUDA2CPU = 2,   ///< Device → Host (D2H)
    kMemcpyCUDA2CUDA = 3,  ///< Device → Device (D2D)
};

/**
 * @brief 设备内存分配器基类
 *
 * 定义统一的 allocate / release / memcpy / memset_zero 接口
 */
class DeviceAllocator {
   public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {};

    virtual DeviceType device_type() const {
        return device_type_;
    }

    virtual void release(void* ptr) const = 0;

    virtual void* allocate(size_t size) const = 0;

    /**
     * @brief 跨设备内存拷贝
     * @param stream 非空时执行 cudaMemcpyAsync
     * @param need_sync 拷贝后是否立即同步设备
     */
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                        bool need_sync = false) const;

    /**
     * @brief 内存置零
     * @param stream 非空时执行 cudaMemsetAsync
     */
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

   private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

/// @brief CPU 内存分配器，使用 malloc/free (对齐分配)
class CPUDeviceAllocator : public DeviceAllocator {
   public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t size) const override;

    void release(void* ptr) const override;
};

/// @brief CUDA 显存块元数据，用于显存池管理
struct CudaMemoryBuffer {
    void* data;        ///< 显存指针
    size_t byte_size;  ///< 块大小 (bytes)
    bool busy;         ///< 是否在使用中

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {
    }
};

/**
 * @brief CUDA 显存分配器 (带缓存池)
 *
 * 释放时不立即 cudaFree，而是标记为空闲并缓存；
 * 再次分配时优先复用合适的空闲块，以减少 cudaMalloc 开销
 */
class CUDADeviceAllocator : public DeviceAllocator {
   public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;

   private:
    mutable std::map<int, size_t> no_busy_cnt_;  ///< 各 GPU 空闲内存统计
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;  ///< 大块缓存 (>1MB)
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;  ///< 小块缓存
};

/// @brief CPU 分配器单例工厂
class CPUDeviceAllocatorFactory {
   public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

   private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

/// @brief CUDA 分配器单例工厂
class CUDADeviceAllocatorFactory {
   public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

   private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

}  // namespace base

#endif  // NANO_INFER_ALLOC_H
