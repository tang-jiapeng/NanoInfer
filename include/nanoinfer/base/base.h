#ifndef NANO_INFER_BASE_H
#define NANO_INFER_BASE_H

#include <cstdint>
#include <string>
#include "glog/logging.h"

/**
 * @brief 消除未使用的变量警告
 */
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)

namespace model {

/**
 * @brief 模型推理过程中的缓冲区类型枚举
 *
 * 定义了 LLM 推理过程中各个阶段产生的中间数据或特定的 Buffer 用途
 * 主要用于显存管理和算子间的数据传递
 */
enum class ModelBufferType {
    kInputTokens = 0,        ///< 输入 Token ID 序列
    kInputEmbeddings = 1,    ///< Token 对应的 Embedding 向量
    kOutputRMSNorm = 2,      ///< RMSNorm 层的输出
    kKeyCache = 3,           ///< KV Cache 中的 Key 缓存
    kValueCache = 4,         ///< KV Cache 中的 Value 缓存
    kQuery = 5,              ///< Attention 中的 Query 向量
    kInputPos = 6,           ///< 输入 Token 的位置索引 (RoPE 使用)
    kScoreStorage = 7,       ///< Attention Score 存储
    kOutputMHA = 8,          ///< Multi-Head Attention 的输出结果
    kAttnOutput = 9,         ///< Attention 层的最终输出
    kW1Output = 10,          ///< FFN 层 W1 (Gate) 的输出
    kW2Output = 11,          ///< FFN 层 W2 (Down) 的输出
    kW3Output = 12,          ///< FFN 层 W3 (Up) 的输出
    kFFNRMSNorm = 13,        ///< FFN 之前的 RMSNorm 输出
    kForwardOutput = 15,     ///< 模型最终的前向传播输出 (GPU)
    kForwardOutputCPU = 16,  ///< 模型最终输出的 CPU 副本

    kSinCache = 17,  ///< RoPE 预计算的 Sin 表
    kCosCache = 18,  ///< RoPE 预计算的 Cos 表
};
}  // namespace model

namespace base {

/**
 * @brief 计算设备类型
 */
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

/**
 * @brief 数据类型
 * 支持常见的浮点和整型格式，用于 Tensor 的元数据描述
 */
enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,   ///< 32位浮点数 (float)
    kDataTypeInt8 = 2,   ///< 8位有符号整数 (int8_t)，用于量化
    kDataTypeInt32 = 3,  ///< 32位有符号整数 (int32_t)
};

/**
 * @brief 支持的模型架构类型
 */
enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,  ///< Llama2 系列架构
};

/**
 * @brief 获取数据类型占用的字节数
 *
 * @param data_type 数据类型枚举
 * @return size_t 字节大小 (例如 kDataTypeFp32 返回 4)
 */
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
 * @brief 禁止拷贝基类 (Mixin)
 *
 * 继承此类的子类将无法被拷贝构造或赋值。
 * 适用于管理独占资源（如内存指针、文件句柄）的类。
 */
class NoCopyable {
   protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

/**
 * @brief 状态码枚举
 */
enum StatusCode : uint8_t {
    kSuccess = 0,              ///< 成功
    kFunctionUnImplement = 1,  ///< 功能未实现
    kPathNotValid = 2,         ///< 路径无效
    kModelParseError = 3,      ///< 模型解析错误
    kInternalError = 5,        ///< 内部错误
    kKeyValueHasExist = 6,     ///< 键值已存在
    kInvalidArgument = 7,      ///< 参数无效
};

/**
 * @brief Tokenizer 编码类型
 */
enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,  ///< SentencePiece
    kEncodeBpe = 1,  ///< Byte Pair Encoding
};

/**
 * @brief 状态类 (Status)
 *
 * 用于函数返回值，表示操作是否成功以及错误信息
 * 类似于 Rust 的 Result 或 Google Abseil 的 Status
 */
class Status {
   public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code);

    bool operator==(int code) const;

    bool operator!=(int code) const;

    operator int() const;

    /**
     * @brief 转换为布尔值
     * @return true 表示成功 (kSuccess)，false 表示失败
     */
    operator bool() const;

    const std::string& get_err_msg() const;

    void set_err_msg(const std::string& err_msg);

   private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

namespace error {

/**
 * @brief 状态检查宏
 *
 * 检查函数调用的返回值 (Status)
 * 如果状态不是 Success，则打印错误日志 (包括文件名、行号、错误码和信息) 并终止程序
 * (LOG(FATAL))
 */
#define STATUS_CHECK(call)                                                             \
    do {                                                                               \
        const base::Status& status = call;                                             \
        if (!status) {                                                                 \
            const size_t buf_size = 512;                                               \
            char buf[buf_size];                                                        \
            snprintf(buf, buf_size - 1,                                                \
                     "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", \
                     __FILE__, __LINE__, int(status), status.get_err_msg().c_str());   \
            LOG(FATAL) << buf;                                                         \
        }                                                                              \
    } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

/**
 * @brief 重载流输出操作符，方便打印 Status 错误信息
 */
std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base

#endif  // NANO_INFER_BASE_H