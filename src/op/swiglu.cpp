/**
 * @file swiglu.cpp
 * @brief SwiGLU 激活层实现（SwiGLULayer）
 *
 * 计算：Output = Swish(Gate) * Up
 *   - Gate = Input1（门控投影 W1 的输出）
 *   - Up   = Input2（上投影 W3 的输出）
 *   - Swish(x) = x * sigmoid(x)
 *
 * 常用于 LLaMA FFN 层中的非线性激活。
 */
#include "nanoinfer/op/swiglu.h"
#include "kernels/kernel_registry.h"
#include "kernels/kernel_types.h"

namespace op {

SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
    reset_input_size(2);
    reset_output_size(1);
}

/** @brief 输入校验：两个输入形状一致、输出尺寸匹配 */
base::Status SwiGLULayer::check() const {
    const auto& input1 = get_input(0);
    const auto& input2 = get_input(1);
    const auto& output = get_output(0);

    if (input1.is_empty() || input2.is_empty() || output.is_empty())
        return base::error::InvalidArgument("Tensor empty");

    // 检查两个输入形状是否一致
    if (input1.size() != input2.size()) {
        return base::error::InvalidArgument("SwiGLU inputs size mismatch");
    }

    // 检查 hidden_dim_ (虽然 SwiGLU 计算是逐元素的，但通常最后一维是 hidden_dim)
    if (output.size() != input1.size()) {
        return base::error::InvalidArgument("SwiGLU output size mismatch");
    }

    return base::error::Success();
}

/** @brief 前向计算：校验后分发 "swiglu" 算子 */
base::Status SwiGLULayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }
    auto swiglu_kernel =
        kernel::KernelRegistry::instance().get<kernel::SwigluKernelFn>("swiglu", device_type_);
    if (!swiglu_kernel) {
        return base::error::InternalError("SwiGLU kernel not found for device: " +
                                          std::to_string(static_cast<int>(device_type_)));
    }

    swiglu_kernel(get_input(0), get_input(1), get_output(0),
                  cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}

}  // namespace op