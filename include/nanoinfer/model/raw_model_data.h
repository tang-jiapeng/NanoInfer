#ifndef NANO_INFER_RAW_MODEL_DATA_H
#define NANO_INFER_RAW_MODEL_DATA_H

#include <cstddef>
#include <cstdint>

namespace model {

/**
 * @brief 原始模型数据基类
 *
 * 负责管理模型文件的底层资源（如文件描述符、内存映射指针）。
 * 通常采用 mmap (Memory Mapping) 技术将巨大的模型文件映射到虚拟内存空间，
 * 实现权重的按需加载 (Lazy Loading) 和零拷贝 (Zero-Copy) 访问。
 */
struct RawModelData {
    /**
     * @brief 析构函数
     * 负责自动释放资源：关闭文件描述符 (close fd) 并解除内存映射 (munmap)。
     */
    ~RawModelData();

    int32_t fd = -1;              ///< 模型文件的文件描述符
    size_t file_size = 0;         ///< 模型文件总字节大小
    void* data = nullptr;         ///< mmap 映射的起始虚拟地址 (指向文件头)
    void* weight_data = nullptr;  ///< 权重数据的起始地址 (跳过文件头后的位置)

    /**
     * @brief 获取指定偏移量的权重指针 
     *
     * @param offset 相对于 weight_data 的字节偏移量
     * @return const void* 指向具体权重的原始指针
     */
    virtual const void* weight(size_t offset) const = 0;
};

/**
 * @brief FP32 格式的原始模型数据
 * 适用于标准精度模型。
 */
struct RawModelDataFp32 : RawModelData {
    /**
     * @brief 获取 FP32 权重指针
     * 实现通常直接返回 static_cast<const char*>(weight_data) + offset
     */
    const void* weight(size_t offset) const override;
};

/**
 * @brief Int8 量化格式的原始模型数据
 * 适用于量化模型，权重数据排列可能更加紧凑。
 */
struct RawModelDataInt8 : RawModelData {
    /**
     * @brief 获取 Int8 权重指针
     */
    const void* weight(size_t offset) const override;
};

}  // namespace model

#endif  // NANO_INFER_RAW_MODEL_DATA_H