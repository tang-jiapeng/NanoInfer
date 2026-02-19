/**
 * @file raw_model_data.h
 * @brief 原始模型数据：mmap 映射 + 权重指针访问
 */
#ifndef NANO_INFER_RAW_MODEL_DATA_H
#define NANO_INFER_RAW_MODEL_DATA_H

#include <cstddef>
#include <cstdint>

namespace model {

/// @brief 原始模型数据基类（mmap 零拷贝访问）
struct RawModelData {
    ~RawModelData();

    int32_t fd = -1;
    size_t file_size = 0;
    void* data = nullptr;
    void* weight_data = nullptr;

    /// @brief 返回 weight_data + offset 处的权重指针
    virtual const void* weight(size_t offset) const = 0;
};

/// @brief FP32 格式模型数据
struct RawModelDataFp32 : RawModelData {
    const void* weight(size_t offset) const override;
};

/// @brief Int8 量化格式模型数据
struct RawModelDataInt8 : RawModelData {
    const void* weight(size_t offset) const override;
};

}  // namespace model

#endif  // NANO_INFER_RAW_MODEL_DATA_H