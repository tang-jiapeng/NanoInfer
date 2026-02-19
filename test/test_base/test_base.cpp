#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>
#include "nanoinfer/base/base.h"

// ===========================================================================
// DataTypeTest — 测试 DataType 枚举及 DataTypeSize 工具函数
// ===========================================================================
class DataTypeTest : public ::testing::Test {};

TEST_F(DataTypeTest, Fp32Size) {
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeFp32), sizeof(float));
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeFp32), 4u);
}

TEST_F(DataTypeTest, Int8Size) {
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt8), sizeof(int8_t));
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt8), 1u);
}

TEST_F(DataTypeTest, Int32Size) {
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt32), sizeof(int32_t));
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeInt32), 4u);
}

TEST_F(DataTypeTest, UnknownSizeIsZero) {
    EXPECT_EQ(base::DataTypeSize(base::DataType::kDataTypeUnknown), 0u);
}

// ===========================================================================
// StatusTest — 测试 base::Status 的构造、比较、bool 转换和错误工厂
// ===========================================================================
class StatusTest : public ::testing::Test {};

TEST_F(StatusTest, DefaultIsSuccess) {
    base::Status s;
    EXPECT_TRUE(s);
    EXPECT_EQ(s.get_err_code(), base::StatusCode::kSuccess);
    EXPECT_EQ(static_cast<int>(s), base::StatusCode::kSuccess);
}

TEST_F(StatusTest, SuccessFactory) {
    base::Status s = base::error::Success();
    EXPECT_TRUE(s);
    EXPECT_EQ(s.get_err_code(), base::StatusCode::kSuccess);
}

TEST_F(StatusTest, ErrorStatusIsFalse) {
    base::Status s = base::error::InternalError("something went wrong");
    EXPECT_FALSE(s);
    EXPECT_NE(s.get_err_code(), base::StatusCode::kSuccess);
    EXPECT_FALSE(s.get_err_msg().empty());
    EXPECT_NE(s.get_err_msg().find("something"), std::string::npos);
}

TEST_F(StatusTest, InvalidArgumentFactory) {
    base::Status s = base::error::InvalidArgument("bad param");
    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::StatusCode::kInvalidArgument);
}

TEST_F(StatusTest, FunctionNotImplementFactory) {
    base::Status s = base::error::FunctionNotImplement("not impl");
    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::StatusCode::kFunctionUnImplement);
}

TEST_F(StatusTest, EqualityOperator) {
    base::Status s = base::error::Success();
    EXPECT_TRUE(s == base::StatusCode::kSuccess);
    EXPECT_FALSE(s == base::StatusCode::kInternalError);
}

TEST_F(StatusTest, NotEqualityOperator) {
    base::Status s = base::error::InternalError("err");
    EXPECT_TRUE(s != base::StatusCode::kSuccess);
}

TEST_F(StatusTest, SetErrMsg) {
    base::Status s = base::error::InternalError("initial");
    s.set_err_msg("updated message");
    EXPECT_EQ(s.get_err_msg(), "updated message");
}

TEST_F(StatusTest, AssignFromInt) {
    base::Status s;
    s = base::StatusCode::kInternalError;
    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::StatusCode::kInternalError);

    s = base::StatusCode::kSuccess;
    EXPECT_TRUE(s);
}

TEST_F(StatusTest, StreamOutput) {
    base::Status s = base::error::InternalError("stream_test");
    std::ostringstream oss;
    oss << s;
    // 流输出至少包含错误信息
    EXPECT_FALSE(oss.str().empty());
}

// ===========================================================================
// DeviceTypeTest — 验证枚举值与文档注释一致
// ===========================================================================
class DeviceTypeTest : public ::testing::Test {};

TEST_F(DeviceTypeTest, EnumValues) {
    EXPECT_EQ(static_cast<uint8_t>(base::DeviceType::kDeviceUnknown), 0u);
    EXPECT_EQ(static_cast<uint8_t>(base::DeviceType::kDeviceCPU), 1u);
    EXPECT_EQ(static_cast<uint8_t>(base::DeviceType::kDeviceCUDA), 2u);
}

// ===========================================================================
// NoCopyableTest — 验证继承 NoCopyable 的类不可被拷贝 (编译期测试)
// ===========================================================================
class NoCopyableTest : public ::testing::Test {};

class MyResource : public base::NoCopyable {
   public:
    explicit MyResource(int v) : value(v) {
    }
    int value;
};

TEST_F(NoCopyableTest, CanConstructNotCopy) {
    MyResource r(42);
    EXPECT_EQ(r.value, 42);
    // MyResource r2 = r;  // 如果取消注释，编译应报错 (deleted copy ctor)
    // MyResource r3(r);   // 同上
}

TEST_F(NoCopyableTest, CanMoveIfDefined) {
    // NoCopyable 没有禁止移动，子类可以自定义
    MyResource r(99);
    EXPECT_EQ(r.value, 99);
}
