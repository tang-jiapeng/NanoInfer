#include "nanoinfer/op/rmsnorm.h"
#include "kernels/kernels_interface.h"

namespace op {
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"), dim_(dim) {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status RmsNormLayer::check() const {
    const auto& input = get_input(0);
    const auto& output = get_output(0);

    // 基础检查
    if (input.is_empty() || output.is_empty()) return base::error::InvalidArgument("Tensor empty");
    if (input.device_type() != device_type_ || output.device_type() != device_type_)
        return base::error::InvalidArgument("Device mismatch");

    // 维度检查
    // 输入至少 2D [Batch*Seq, Hidden]
    if (input.dims_size() < 1) return base::error::InvalidArgument("Dims size error");

    // 检查最后一维是否等于 dim_ (初始化参数)
    int32_t last_dim = input.get_dim(input.dims_size() - 1);
    if (last_dim != dim_) {
        LOG(ERROR) << "RMSNorm input last dim mismatch. Expected " << dim_ << " got " << last_dim;
        return base::error::InvalidArgument("RMSNorm dimension mismatch");
    }

    // 输出检查
    if (output.size() != input.size()) {
        return base::error::InvalidArgument("Output size mismatch");
    }

    return base::error::Success();
}

base::Status RmsNormLayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }
    auto input = this->get_input(0);
    auto weight = this->get_weight(0);
    auto output = this->get_output(0);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }

    kernel::get_rmsnorm_kernel(device_type_)(input, weight, output,
                                             cuda_config_ ? cuda_config_->stream : nullptr);

    return base::error::Success();
}

}  // namespace op
