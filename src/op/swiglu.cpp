#include "nanoinfer/op/swiglu.h"
#include "kernels/kernels_interface.h"

namespace op {

SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
    reset_input_size(2);
    reset_output_size(1);
}

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

base::Status SwiGLULayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }
    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }
    kernel::get_swiglu_kernel(device_type_)(input1, input2, output,
                                            cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}

}  // namespace op