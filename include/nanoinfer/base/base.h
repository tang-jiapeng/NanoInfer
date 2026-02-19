/**
 * @file base.h
 * @brief 基础类型、错误码与状态定义
 */
#ifndef NANO_INFER_BASE_H
#define NANO_INFER_BASE_H

#include <cstdint>
#include <string>
#include "glog/logging.h"

/// @brief 消除未使用变量的编译警告
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

namespace base {

/// @brief 计算设备类型
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

/// @brief 数据类型枚举，用于 Tensor 元数据描述
enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,   ///< 32-bit float
    kDataTypeInt8 = 2,   ///< 8-bit int (量化)
    kDataTypeInt32 = 3,  ///< 32-bit int
};

/// @brief 模型架构类型
enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,  ///< LLaMA-2
};

/// @brief 获取 DataType 对应的字节大小
inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    } else {
        return 0;
    }
}

/**
 * @brief 不可拷贝基类 (Mixin)
 *
 * 继承此类以禁用拷贝构造和赋值，适用于管理独占资源的类
 */
class NoCopyable {
   protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

/// @brief 状态码
enum StatusCode : uint8_t {
    kSuccess = 0,              ///< 成功
    kFunctionUnImplement = 1,  ///< 功能未实现
    kPathNotValid = 2,         ///< 路径无效
    kModelParseError = 3,      ///< 模型解析错误
    kInternalError = 5,        ///< 内部错误
    kKeyValueHasExist = 6,     ///< 键值已存在
    kInvalidArgument = 7,      ///< 参数无效
};

/// @brief Tokenizer 编码类型
enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,  ///< SentencePiece
    kEncodeBpe = 1,  ///< Byte Pair Encoding
};

/// @brief 操作状态类，承载错误码与错误消息
class Status {
   public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code);

    bool operator==(int code) const;

    bool operator!=(int code) const;

    operator int() const;

    operator bool() const;

    int32_t get_err_code() const;

    const std::string& get_err_msg() const;

    void set_err_msg(const std::string& err_msg);

   private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

namespace error {

/// @brief 状态检查宏：失败时打印错误并终止程序 (LOG(FATAL))
#define STATUS_CHECK(call)                                                                       \
    do {                                                                                         \
        const base::Status& status = call;                                                       \
        if (!status) {                                                                           \
            const size_t buf_size = 512;                                                         \
            char buf[buf_size];                                                                  \
            snprintf(buf, buf_size - 1,                                                          \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
                     __LINE__, int(status), status.get_err_msg().c_str());                       \
            LOG(FATAL) << buf;                                                                   \
        }                                                                                        \
    } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base

#endif  // NANO_INFER_BASE_H