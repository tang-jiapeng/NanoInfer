#ifndef NANO_INFER_BUFFER_H
#define NANO_INFER_BUFFER_H
#include "alloc.h"

namespace base {

/**
 * @brief 内存缓冲区类 (Buffer)
 *
 * Buffer 是对一段连续内存的封装，它不关心数据的具体维度或类型，只关心字节大小和存储设备
 *
 * 核心特性：
 * RAII 管理：负责内存的分配与释放（除非标记为外部内存）。
 * 不可拷贝：继承自 NoCopyable，防止对原始指针的意外浅拷贝。
 * 设备感知：知道数据位于 CPU 还是 GPU，并持有相应的分配器。
 */
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
   private:
    size_t byte_size_ = 0;       ///< 内存块的字节大小
    void* ptr_ = nullptr;        ///< 原始数据指针
    bool use_external_ = false;  ///< 是否使用外部内存 (即不拥有所有权)
    DeviceType device_type_ = DeviceType::kDeviceUnknown;  ///< 所在的设备类型
    std::shared_ptr<DeviceAllocator> allocator_;           ///< 关联的内存分配器

   public:
    /**
     * @brief 默认构造函数
     * 创建一个空的、无效的 Buffer。
     */
    explicit Buffer() = default;

    /**
     * @brief 构造函数
     *
     * @param byte_size 缓冲区的字节大小
     * @param allocator 内存分配器。如果需要 Buffer 自动分配内存，此参数不能为空
     * @param ptr 预先存在的内存指针 (可选)
     * @param use_external 是否为外部内存 (决定了析构时的行为)
     *
     * @note 所有权说明:
     * Allocate Mode: (ptr=nullptr, use_external=false)
     * - Buffer 会调用 allocator->allocate(byte_size) 分配新内存
     * - 析构时会自动释放
     *
     * View Mode (外部引用): (ptr!=nullptr, use_external=true)
     * - Buffer 直接使用传入的 ptr
     * - 析构时不会释放内存。调用者需保证 ptr 的生命周期长于 Buffer
     *
     * Take Ownership Mode (接管): (ptr!=nullptr, use_external=false)
     * - Buffer 接管传入 ptr 的所有权
     * - 析构时会调用 allocator->release(ptr) 释放内存
     * - 警告：必须确保 ptr 是由该 allocator 分配的，否则行为未定义
     */
    explicit Buffer(size_t byte_size,
                    std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    /**
     * @brief 析构函数
     * 如果 use_external_ 为 false 且 ptr_ 有效，则通过 allocator_ 释放内存
     */
    virtual ~Buffer();

    /**
     * @brief 执行内存分配
     *
     * 通常在默认构造后，或需要延迟分配时调用。
     * 如果 allocator 为空或 byte_size 为 0，则返回 false。
     *
     * @return true 分配成功
     * @return false 分配失败
     */
    bool allocate();

    /**
     * @brief 从另一个 Buffer 拷贝数据
     *
     * 自动处理不同设备间的拷贝 (CPU<->CPU, CPU<->GPU, GPU<->GPU)
     * 实际执行由 allocator_->memcpy 分发
     *
     * @param buffer 源 Buffer
     */
    void copy_from(const Buffer& buffer) const;

    /**
     * @brief 从另一个 Buffer 指针拷贝数据
     * @param buffer 源 Buffer 指针
     */
    void copy_from(const Buffer* buffer) const;

    /**
     * @brief 获取原始数据指针 (可读写)
     * @return void*
     */
    void* ptr();

    /**
     * @brief 获取原始数据指针 (只读)
     * @return const void*
     */
    const void* ptr() const;

    /**
     * @brief 获取缓冲区字节大小
     */
    size_t byte_size() const;

    /**
     * @brief 获取关联的分配器
     */
    std::shared_ptr<DeviceAllocator> allocator() const;

    /**
     * @brief 获取设备类型
     */
    DeviceType device_type() const;

    /**
     * @brief 设置设备类型
     * @note 仅修改标记，不触发数据迁移。通常由 Allocator 自动设置
     */
    void set_device_type(DeviceType device_type);

    /**
     * @brief 获取自身的 shared_ptr
     *
     * 用于在类内部安全地分发自身的共享所有权
     */
    std::shared_ptr<Buffer> get_shared_from_this();

    /**
     * @brief 检查是否为外部内存
     * @return true 表示内存不由 Buffer 管理 (析构时不释放)
     */
    bool is_external() const;
};
}  // namespace base

#endif  // NANO_INFER_BUFFER_H
