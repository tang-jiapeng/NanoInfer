/**
 * @file raw_model_data.cpp
 * @brief 原始模型数据资源管理（mmap 释放 + 权重偏移访问）
 *
 * RawModelData 管理通过 mmap 映射的模型文件生命周期：
 *   - 析构时调用 munmap 解除映射并 close 文件描述符
 *   - RawModelDataFp32::weight() 返回 float* + 偏移量
 *   - RawModelDataInt8::weight() 返回 int8_t* + 偏移量
 */
#include "nanoinfer/model/raw_model_data.h"
#include <sys/mman.h>
#include <unistd.h>
namespace model {
RawModelData::~RawModelData() {
    if (data != nullptr && data != MAP_FAILED) {
        munmap(data, file_size);
        data = nullptr;
    }
    if (fd != -1) {
        close(fd);
        fd = -1;
    }
}

const void* RawModelDataFp32::weight(size_t offset) const {
    return static_cast<float*>(weight_data) + offset;
}

const void* RawModelDataInt8::weight(size_t offset) const {
    return static_cast<int8_t*>(weight_data) + offset;
}
}  // namespace model