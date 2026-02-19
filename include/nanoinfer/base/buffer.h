/**
 * @file buffer.h
 * @brief RAII 内存缓冲区，封装设备感知的连续字节内存
 */
#ifndef NANO_INFER_BUFFER_H
#define NANO_INFER_BUFFER_H
#include "alloc.h"

namespace base {

/**
 * @brief 内存缓冲区 (Buffer)
 *
 * 封装连续内存的 RAII 包装，不可拷贝，支持 CPU / CUDA 设备感知。
 *
 * 三种所有权模式：
 * - Allocate:       ptr=nullptr, use_external=false → 自动分配和释放
 * - View (外部引用): ptr!=nullptr, use_external=true  → 不释放，调用方管理生命周期
 * - TakeOwnership:  ptr!=nullptr, use_external=false → 接管所有权，析构时释放
 */
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
   private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;

   public:
    explicit Buffer() = default;

    /**
     * @brief 构造 Buffer
     * @param byte_size 字节大小
     * @param allocator 分配器（为空则不自动分配）
     * @param ptr 外部指针（非空时不分配新内存）
     * @param use_external 为 true 时析构不释放 ptr
     */
    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    virtual ~Buffer();

    /// @brief 执行延迟内存分配
    bool allocate();

    /// @brief 从另一个 Buffer 拷贝数据，自动处理跨设备拷贝
    void copy_from(const Buffer& buffer) const;

    void copy_from(const Buffer* buffer) const;

    void* ptr();

    const void* ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;
};
}  // namespace base

#endif  // NANO_INFER_BUFFER_H
