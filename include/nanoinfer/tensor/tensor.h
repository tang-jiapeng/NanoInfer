#ifndef NANO_INFER_TENSOR_H
#define NANO_INFER_TENSOR_H

#include <driver_types.h>
#include <future>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/buffer.h"

namespace tensor {

/**
 * @brief 张量类 (Tensor)
 *
 * Tensor 是 NanoInfer 中用于表示多维数组的核心数据结构。
 * 它封装了维度信息 (dims)、数据类型 (data_type) 以及底层数据缓冲区 (Buffer)
 *
 * @note
 * 1. Tensor 的拷贝构造函数和赋值操作符执行的是浅拷贝，即共享底层的 Buffer
 * 如果需要独立的内存副本，请使用 clone() 方法。
 * 2. Tensor 支持 CPU 和 CUDA 两种设备类型的内存管理，可通过 to_cpu() / to_cuda() 进行迁移
 */
class Tensor {
   public:
    /**
     * @brief 默认构造函数，创建一个空的张量
     */
    explicit Tensor() = default;

    /**
     * @brief 构造一个一维张量
     *
     * @param data_type 数据类型 (如 kDataTypeFp32, kDataTypeInt8)
     * @param dim0 第0维的大小
     * @param need_alloc 是否立即分配内存。
     * - true: 必须提供 alloc 参数，会在构造时分配内存。
     * - false: 不分配内存，或者使用外部传入的 ptr。
     * @param alloc 设备分配器 (DeviceAllocator)。如果 need_alloc 为true，此参数不能为空
     * @param ptr 外部传入的数据指针。
     * - 如果非空，Tensor 将管理这块外部内存（不会自动释放，除非 Buffer配置了接管）
     * - 如果 ptr 非空，need_alloc 必须为 false
     */
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /**
     * @brief 构造一个二维张量
     * @param dim1 第1维的大小
     * @see Tensor(base::DataType, int32_t, bool, std::shared_ptr<base::DeviceAllocator>,
     * void*)
     */
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /**
     * @brief 构造一个三维张量
     * @param dim2 第2维的大小
     * @see Tensor(base::DataType, int32_t, bool, std::shared_ptr<base::DeviceAllocator>,
     * void*)
     */
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /**
     * @brief 构造一个四维张量
     * @param dim3 第3维的大小
     * @see Tensor(base::DataType, int32_t, bool, std::shared_ptr<base::DeviceAllocator>,
     * void*)
     */
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    int32_t dim3, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /**
     * @brief 构造一个多维张量 (通用构造函数)
     *
     * @param dims 维度向量 (std::vector<int32_t>)
     * @see Tensor(base::DataType, int32_t, bool, std::shared_ptr<base::DeviceAllocator>,
     * void*)
     */
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims,
                    bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /**
     * @brief 将 Tensor 数据迁移到 CPU
     *
     * 如果当前数据在 CUDA 上，会触发 DeviceToHost 拷贝。
     * 迁移后，Tensor 的 device_type 将变为 kDeviceCPU。
     * 如果当前已经在 CPU 上，则不进行任何操作。
     */
    void to_cpu();

    /**
     * @brief 将 Tensor 数据迁移到 CUDA
     *
     * 如果当前数据在 CPU 上，会触发 HostToDevice 拷贝。
     * 迁移后，Tensor 的 device_type 将变为 kDeviceCUDA。
     *
     * @param stream CUDA 流，用于异步拷贝操作。如果为 nullptr，则使用默认流。
     */
    void to_cuda(cudaStream_t stream = nullptr);

    /**
     * @brief 判断 Tensor 是否为空
     * @return true 如果 size 为 0 或底层 buffer 未初始化
     */
    bool is_empty() const;

    /**
     * @brief 初始化 Buffer (通常用于延迟分配或包装外部指针)
     *
     * @param alloc 设备分配器
     * @param data_type 数据类型
     * @param need_alloc 是否需要重新分配内存
     * @param ptr 外部数据指针 (可选)
     */
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                     base::DataType data_type, bool need_alloc, void* ptr);

    /**
     * @brief 获取指向数据的原始指针 (可变)
     *
     * @tparam T 目标数据类型 (如 float, int8_t)
     * @return T* 数据首地址指针。如果 buffer 为空，返回 nullptr。
     */
    template <typename T>
    T* ptr();

    /**
     * @brief 获取指向数据的原始指针 (只读)
     *
     * @tparam T 目标数据类型
     * @return const T* 只读数据首地址指针。
     */
    template <typename T>
    const T* ptr() const;

    /**
     * @brief 重塑 Tensor 的形状 (Reshape)
     *
     * 修改 Tensor 的维度信息。
     * 如果新维度的元素总数 (element count) 大于当前分配的内存容量，
     * 将会触发重新分配 (Reallocate) 并拷贝旧数据。
     *
     * @param dims 新的维度向量
     */
    void reshape(const std::vector<int32_t>& dims);

    /**
     * @brief 获取底层 Buffer 对象
     * @return std::shared_ptr<base::Buffer>
     */
    std::shared_ptr<base::Buffer> get_buffer() const;

    /**
     * @brief 获取 Tensor 的元素总个数
     * 例如: dims=[2, 3]，size=6
     */
    size_t size() const;

    /**
     * @brief 获取 Tensor 占用的总字节数
     * 公式: size() * sizeof(DataType)
     */
    size_t byte_size() const;

    /**
     * @brief 获取维度的阶数 (Rank)
     * 例如: dims=[2, 3]，dims_size=2
     */
    int32_t dims_size() const;

    /**
     * @brief 获取数据类型
     */
    base::DataType data_type() const;

    /**
     * @brief 获取指定维度的长度
     * @param idx 维度索引 (从 0 开始)
     * @return int32_t 该维度的长度
     */
    int32_t get_dim(int32_t idx) const;

    /**
     * @brief 获取完整的维度向量
     */
    const std::vector<int32_t>& dims() const;

    /**
     * @brief 获取步长向量 (Strides)
     *
     * 步长表示在某一维度上移动一个元素需要跨越的内存距离（以元素个数为单位）。
     * 默认假设内存是连续紧凑的 (Contiguous)。
     */
    std::vector<size_t> strides() const;

    /**
     * @brief 将当前 Tensor 指向一个新的 Buffer
     *
     * @param buffer 新的 Buffer 对象
     * @return true 成功
     * @return false 失败 (例如 buffer 大小不足以容纳当前 Tensor)
     */
    bool assign(std::shared_ptr<base::Buffer> buffer);

    /**
     * @brief 重置 Tensor 的元数据 (类型和维度)，并解绑当前的 Buffer
     *
     * 调用后，Tensor 变为空 (需要重新分配内存)。
     *
     * @param data_type 新的数据类型
     * @param dims 新的维度
     */
    void reset(base::DataType data_type, const std::vector<int32_t>& dims);

    /**
     * @brief 设置设备类型 (仅修改标记，不触发迁移)
     * @note 慎用，通常应通过 to_cpu/to_cuda 修改
     */
    void set_device_type(base::DeviceType device_type) const;

    /**
     * @brief 获取当前设备类型
     */
    base::DeviceType device_type() const;

    /**
     * @brief 为 Tensor 分配内存
     *
     * @param allocator 设备分配器
     * @param need_realloc 如果当前 buffer 足够大，是否强制重新分配。
     * - false: 如果容量足够则复用
     * - true: 强制销毁旧 buffer 并分配新的
     * @return true 分配成功
     */
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                  bool need_realloc = false);

    /**
     * @brief 获取指定偏移量的元素指针
     * @param index 线性偏移量 (Linear Index)
     * @return T* 元素指针
     */
    template <typename T>
    T* ptr(int64_t index);

    /**
     * @brief 获取指定偏移量的元素指针 (只读)
     * @param index 线性偏移量
     * @return const T* 元素指针
     */
    template <typename T>
    const T* ptr(int64_t index) const;

    /**
     * @brief 获取指定偏移量的元素引用 (带边界检查)
     *
     * @param offset 线性偏移量
     * @return T& 元素引用
     */
    template <typename T>
    T& index(int64_t offset);

    /**
     * @brief 获取指定偏移量的元素引用 (只读，带边界检查)
     * @param offset 线性偏移量
     * @return const T& 元素引用
     */
    template <typename T>
    const T& index(int64_t offset) const;

    /**
     * @brief 创建当前 Tensor 的深拷贝副本
     *
     * 创建一个新的 Tensor，分配独立的内存，并将数据从当前 Tensor 拷贝过去
     *
     * @return Tensor 深拷贝后的新对象
     */
    tensor::Tensor clone() const;

   private:
    size_t size_ = 0;                       ///< 元素总数
    std::vector<int32_t> dims_;             ///< 维度信息
    std::shared_ptr<base::Buffer> buffer_;  ///< 内存缓冲区 (共享指针)
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;  ///< 数据类型
};

template <typename T>
T& Tensor::index(int64_t offset) {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, this->size());
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
    CHECK_GE(offset, 0);
    CHECK_LT(offset, this->size());
    const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor

#endif  // NANO_INFER_TENSOR_H
