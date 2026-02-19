/**
 * @file tensor.h
 * @brief 多维张量类，支持 CPU / CUDA 设备感知
 */
#ifndef NANO_INFER_TENSOR_H
#define NANO_INFER_TENSOR_H

#include <driver_types.h>
#include <future>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/buffer.h"

namespace tensor {

/**
 * @brief 多维张量 (Tensor)
 *
 * 封装维度、数据类型与底层 Buffer。拷贝语义为浅拷贝（共享 Buffer），
 * 如需深拷贝请使用 clone()。支持 to_cpu() / to_cuda() 设备迁移
 */
class Tensor {
   public:
    explicit Tensor() = default;

    /// @brief 构造 1-D Tensor
    explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    /// @brief 构造 2-D Tensor
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    /// @brief 构造 3-D Tensor
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                    void* ptr = nullptr);

    /// @brief 构造 4-D Tensor
    explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    int32_t dim3, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    /// @brief 构造任意维度 Tensor
    explicit Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                    std::shared_ptr<base::DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    /// @brief 迁移数据到 CPU (D2H)
    void to_cpu();

    /// @brief 迁移数据到 CUDA (H2D)
    void to_cuda(cudaStream_t stream = nullptr);

    bool is_empty() const;

    /**
     * @brief 初始化底层 Buffer（延迟分配或包装外部指针）
     */
    void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                     bool need_alloc, void* ptr);

    template <typename T>
    T* ptr();

    template <typename T>
    const T* ptr() const;

    /**
     * @brief Reshape：修改维度，元素总数增大时自动重新分配内存
     */
    void reshape(const std::vector<int32_t>& dims);

    std::shared_ptr<base::Buffer> get_buffer() const;

    /// @brief 元素总数
    size_t size() const;

    /// @brief 总字节数 = size() × DataTypeSize
    size_t byte_size() const;

    /// @brief 维度阶数 (Rank)
    int32_t dims_size() const;

    base::DataType data_type() const;

    int32_t get_dim(int32_t idx) const;

    const std::vector<int32_t>& dims() const;

    /// @brief 获取步长向量 (假设内存连续)
    std::vector<size_t> strides() const;

    /// @brief 将 Tensor 指向新的 Buffer
    bool assign(std::shared_ptr<base::Buffer> buffer);

    /// @brief 重置元数据并解绑 Buffer
    void reset(base::DataType data_type, const std::vector<int32_t>& dims);

    void set_device_type(base::DeviceType device_type) const;

    base::DeviceType device_type() const;

    /**
     * @brief 分配内存
     * @param need_realloc 为 true 时强制重新分配
     */
    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

    template <typename T>
    T* ptr(int64_t index);

    template <typename T>
    const T* ptr(int64_t index) const;

    /// @brief 按线性偏移取元素引用 (带边界检查)
    template <typename T>
    T& index(int64_t offset);

    template <typename T>
    const T& index(int64_t offset) const;

    /// @brief 深拷贝，返回独立内存副本
    tensor::Tensor clone() const;

   private:
    size_t size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<base::Buffer> buffer_;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
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
