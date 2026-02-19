/**
 * @file base.cpp
 * @brief Status 状态码类实现及错误工厂函数
 *
 * 提供 NanoInfer 全局统一的错误处理基础设施：
 *   - Status 类：持有 StatusCode + 错误消息，支持 bool 转换与流式输出
 *   - error 命名空间：提供一组便捷工厂函数（Success / InvalidArgument /
 *     InternalError / PathNotValid 等），统一创建 Status 对象
 */
#include "nanoinfer/base/base.h"
#include <string>
namespace base {

/** @brief 构造 Status 对象，携带错误码与描述信息 */
Status::Status(int code, std::string err_message) : code_(code), message_(std::move(err_message)) {
}

Status& Status::operator=(int code) {
    code_ = code;
    return *this;
};

bool Status::operator==(int code) const {
    if (code_ == code) {
        return true;
    } else {
        return false;
    }
};

bool Status::operator!=(int code) const {
    if (code_ != code) {
        return true;
    } else {
        return false;
    }
};

Status::operator int() const {
    return code_;
}

Status::operator bool() const {
    return code_ == kSuccess;
}

int32_t Status::get_err_code() const {
    return code_;
}

const std::string& Status::get_err_msg() const {
    return message_;
}

void Status::set_err_msg(const std::string& err_msg) {
    message_ = err_msg;
}

// -----------------------------------------------------------------------
// error 命名空间：便捷工厂函数，统一创建各类 Status 对象
// -----------------------------------------------------------------------
namespace error {
/** @brief 成功状态 */
Status Success(const std::string& err_msg) {
    return Status{kSuccess, err_msg};
}

Status FunctionNotImplement(const std::string& err_msg) {
    return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
    return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
    return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
    return Status{kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
    return Status{kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
    return Status{kKeyValueHasExist, err_msg};
}
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.get_err_msg();
    return os;
}

}  // namespace base
