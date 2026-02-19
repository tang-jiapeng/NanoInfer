/**
 * @file buffer.cpp
 * @brief Buffer 内存缓冲区实现（RAII 封装 + 跨设备拷贝）
 *
 * Buffer 是 NanoInfer 中最底层的内存抽象，封装了一段连续的设备内存：
 *   - 构造时通过 DeviceAllocator 分配内存（或包装外部指针，is_external 模式）
 *   - 析构时自动释放（非 external 模式）
 *   - copy_from 支持四种设备间拷贝：CPU↔CPU、CPU→CUDA、CUDA→CPU、CUDA↔CUDA
 *   - 支持 shared_from_this 语义用于安全共享
 */
#include "nanoinfer/base/buffer.h"
#include <glog/logging.h>

namespace base {

/**
 * @brief 构造 Buffer
 *
 * 两种模式：
 *   1. 托管分配：ptr == nullptr 且提供 allocator，自动分配 byte_size 字节
 *   2. 外部指针包装：use_external = true，不拥有内存所有权
 */
Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    : byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
}

/** @brief 析构：非 external 模式时通过 allocator 释放内存 */
Buffer::~Buffer() {
    if (!use_external_) {
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

void* Buffer::ptr() {
    return ptr_;
}

const void* Buffer::ptr() const {
    return ptr_;
}

size_t Buffer::byte_size() const {
    return byte_size_;
}

/** @brief 延迟分配：在构造时未分配内存的情况下手动触发 */
bool Buffer::allocate() {
    if (allocator_ && byte_size_ != 0) {
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        if (!ptr_) {
            return false;
        } else {
            return true;
        }
    } else {
        return false;
    }
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
    return allocator_;
}

/**
 * @brief 跨设备内存拷贝（从源 Buffer 拷贝到当前 Buffer）
 *
 * 自动推断两端设备类型，选择对应的 MemcpyKind（CPU↔CPU / CPU↔CUDA / CUDA↔CUDA）。
 * 拷贝字节数取两者的较小值（安全截断）。
 */
void Buffer::copy_from(const Buffer& buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);

    size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
    } else if (buffer_device == DeviceType::kDeviceCUDA &&
               current_device == DeviceType::kDeviceCPU) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
    } else if (buffer_device == DeviceType::kDeviceCPU &&
               current_device == DeviceType::kDeviceCUDA) {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, MemcpyKind::kMemcpyCPU2CUDA);
    } else {
        return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                  MemcpyKind::kMemcpyCUDA2CUDA);
    }
}

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

    size_t dest_size = byte_size_;
    size_t src_size = buffer->byte_size_;
    size_t byte_size = src_size < dest_size ? src_size : dest_size;

    const DeviceType& buffer_device = buffer->device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
    } else if (buffer_device == DeviceType::kDeviceCUDA &&
               current_device == DeviceType::kDeviceCPU) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, MemcpyKind::kMemcpyCUDA2CPU);
    } else if (buffer_device == DeviceType::kDeviceCPU &&
               current_device == DeviceType::kDeviceCUDA) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, MemcpyKind::kMemcpyCPU2CUDA);
    } else {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                  MemcpyKind::kMemcpyCUDA2CUDA);
    }
}

DeviceType Buffer::device_type() const {
    return device_type_;
}

void Buffer::set_device_type(DeviceType device_type) {
    device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
    return shared_from_this();
}

bool Buffer::is_external() const {
    return this->use_external_;
}

}  // namespace base