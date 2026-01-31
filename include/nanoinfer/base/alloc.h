#ifndef NANO_INFER_ALLOC_H
#define NANO_INFER_ALLOC_H
#include <map>
#include <memory>
#include <vector>
#include "base.h"

namespace base {

/**
 * @brief 内存拷贝类型枚举
 * 定义了源设备和目标设备的组合
 */
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,    ///< Host 到 Host
    kMemcpyCPU2CUDA = 1,   ///< Host 到 Device (H2D)
    kMemcpyCUDA2CPU = 2,   ///< Device 到 Host (D2H)
    kMemcpyCUDA2CUDA = 3,  ///< Device 到 Device (D2D)
};

/**
 * @brief 设备内存分配器基类
 *
 * 定义了统一的内存分配、释放、拷贝和设置接口
 * 所有具体的设备分配器（如 CPU、CUDA）都必须继承此类
 */
class DeviceAllocator {
   public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){};

    virtual DeviceType device_type() const {
        return device_type_;
    }

    /**
     * @brief 释放内存
     * @param ptr 待释放的内存指针
     */
    virtual void release(void* ptr) const = 0;

    /**
     * @brief 分配指定大小的内存
     * @param size 需要分配的字节数
     * @return void* 分配得到的内存首地址。如果分配失败可能返回 nullptr。
     */
    virtual void* allocate(size_t size) const = 0;

    /**
     * @brief 内存拷贝 (支持异构设备间拷贝)
     *
     * @param src_ptr 源地址指针
     * @param dest_ptr 目标地址指针
     * @param byte_size 拷贝字节数
     * @param memcpy_kind 拷贝类型 (如 H2D, D2H)
     * @param stream CUDA 流。如果非空，则执行异步拷贝 (cudaMemcpyAsync)。
     * @param need_sync 是否需要在拷贝后立即同步设备 (cudaDeviceSynchronize)。
     */
    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                        void* stream = nullptr, bool need_sync = false) const;

    /**
     * @brief 内存置零
     *
     * @param ptr 目标内存指针
     * @param byte_size 置零字节数
     * @param stream CUDA 流。如果非空，则执行异步置零 (cudaMemsetAsync)。
     * @param need_sync 是否需要在置零后立即同步设备。
     */
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream,
                             bool need_sync = false);

   private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

/**
 * @brief CPU 内存分配器
 *
 * 使用标准的 malloc/free 或 posix_memalign 进行内存管理。
 * 通常保证内存对齐（如 32 字节或 16 字节对齐）以利用 SIMD 指令
 */
class CPUDeviceAllocator : public DeviceAllocator {
   public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t size) const override;

    void release(void* ptr) const override;
};

/**
 * @brief CUDA 显存块元数据
 * 用于简单的显存池管理
 */
struct CudaMemoryBuffer {
    void* data;        ///< 显存指针
    size_t byte_size;  ///< 显存块大小
    bool busy;         ///< 是否正在被使用

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {
    }
};

/**
 * @brief CUDA 显存分配器 (带简易显存池)
 *
 * 为了减少 cudaMalloc/cudaFree 的高昂开销，实现了一个简单的缓存机制：
 * 释放的显存不会立即归还给 OS，而是标记为空闲 (busy=false) 并存入 map 中
 * 下次分配时，优先从 map 中寻找大小合适的空闲块
 */
class CUDADeviceAllocator : public DeviceAllocator {
   public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;

   private:
    // 使用 mutable 关键字，允许在 const 成员函数 (allocate/release) 中修改这些缓存结构
    // 这是一种逻辑上的 const：分配器的"接口"不变，但内部状态（缓存池）会变
    mutable std::map<int, size_t> no_busy_cnt_;  ///< 统计各 GPU 设备上空闲内存的总量

    // 大内存块缓存 (> 1MB)，按 GPU ID 分组
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;

    // 小内存块缓存，按 GPU ID 分组
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

/**
 * @brief CPU 分配器工厂 (单例模式)
 */
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

/**
 * @brief CUDA 分配器工厂 (单例模式)
 */
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
