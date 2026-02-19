/**
 * @file add.cpp
 * @brief 向量加法层实现（VecAddLayer）
 *
 * 计算：Output = Input1 + Input2（逐元素）
 * 通过 KernelRegistry 分发 "add" 算子到 CPU/CUDA 后端。
 */
#include "nanoinfer/op/add.h"
#include "kernels/kernel_registry.h"
#include "kernels/kernel_types.h"

namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
    reset_input_size(2);
    reset_output_size(1);
}

/** @brief 输入校验：检查两个输入和输出的空/设备/类型/形状一致性 */
base::Status VecAddLayer::check() const {
    const auto& input1 = get_input(0);
    const auto& input2 = get_input(1);
    const auto& output = get_output(0);

    // 判空
    if (input1.is_empty() || input2.is_empty() || output.is_empty()) {
        return base::error::InvalidArgument("Input or Output tensors are empty in AddLayer");
    }

    // 检查设备与类型
    if (input1.device_type() != device_type_ || input2.device_type() != device_type_ ||
        output.device_type() != device_type_) {
        return base::error::InvalidArgument("Device type mismatch in AddLayer");
    }
    if (input1.data_type() != data_type_ || input2.data_type() != data_type_ ||
        output.data_type() != data_type_) {
        return base::error::InvalidArgument("Data type mismatch in AddLayer");
    }

    // 检查形状一致性
    if (input1.size() != input2.size()) {
        LOG(ERROR) << "Input tensor sizes mismatch in AddLayer: " << input1.size() << " vs "
                   << input2.size();
        return base::error::InvalidArgument("Input tensor size mismatch");
    }

    // 检查输出形状
    if (output.size() != input1.size()) {
        LOG(ERROR) << "Output tensor size mismatch in AddLayer";
        return base::error::InvalidArgument("Output tensor size mismatch");
    }

    return base::error::Success();
}

/** @brief 前向计算：校验后分发 "add" 算子到 CPU/CUDA 后端 */
base::Status VecAddLayer::forward() {
    auto status = this->check();
    if (!status) {
        return status;
    }
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }

    auto add_kernel =
        kernel::KernelRegistry::instance().get<kernel::AddKernelFn>("add", device_type_);
    if (!add_kernel) {
        return base::error::InternalError("Add kernel not found for device: " +
                                          std::to_string(static_cast<int>(device_type_)));
    }

    add_kernel(get_input(0), get_input(1), get_output(0),
               cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}
}  // namespace op