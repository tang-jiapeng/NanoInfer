#include "nanoinfer/op/matmul.h"
#include "kernels/kernels_interface.h"

namespace op {

MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
    if (has_bias_) {
        bias_.resize(1);
    }
}

base::Status MatmulLayer::check() const {
    const auto& input_tensor = get_input(0);
    
    // 检查是否为空
    if (input_tensor.is_empty()) {
        return base::error::InvalidArgument("The input tensor is empty.");
    }
    // 检查设备和类型
    if (input_tensor.device_type() != device_type_ || input_tensor.data_type() != data_type_) {
        LOG(ERROR) << "The input tensor device or data type error in the matmul layer.";
        return base::error::InvalidArgument("Input tensor device/dtype mismatch");
    }
    
    // 检查维度数 (至少 2D)
    if (input_tensor.dims_size() < 2) {
        LOG(ERROR) << "The input tensor dims size error (must >= 2) in the matmul layer.";
        return base::error::InvalidArgument("Input tensor dims size error");
    }
    
    // 只检查最后一维 (K 维度) 是否匹配 dim1_
    // 也就是 inputs 的列数必须等于权重的输入通道数
    int32_t last_dim = input_tensor.get_dim(input_tensor.dims_size() - 1);
    if (last_dim != dim1_) {
        LOG(ERROR) << "The input tensor last dim error in the matmul layer. Expected " 
                   << dim1_ << " but got " << last_dim;
        return base::error::InvalidArgument("Input tensor K-dimension mismatch");
    }

    base::Status status;
    if (!is_quant_layer_) {
        status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    } else {
        // 量化权重通常也是固定的，可以直接查
        status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8, dim0_, dim1_);
    }
    
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the matmul layer.";
        return status;
    }

    if (is_quant_layer_) {
        // Scales 的大小检查比较宽松，或者是 group_size 相关，这里仅检查类型
        if (scales_.is_empty()) {
            return base::error::InvalidArgument("The scale tensor is empty.");
        }
        if (scales_.device_type() != device_type_ || scales_.data_type() != base::DataType::kDataTypeFp32) {
            return base::error::InvalidArgument("The scale tensor device/dtype error.");
        }
    }

    const auto& output_tensor = get_output(0);
    if (output_tensor.is_empty()) return base::error::InvalidArgument("Output tensor is empty");

    // 输出的 batch 维度 (dim0) 必须等于输入的 batch 维度
    if (output_tensor.get_dim(0) != input_tensor.get_dim(0)) {
         LOG(ERROR) << "Output batch size does not match input batch size.";
         return base::error::InvalidArgument("Output batch size mismatch");
    }
    
    // 输出的最后一维必须等于 dim0_ (N 维度)
    if (output_tensor.get_dim(output_tensor.dims_size() - 1) != dim0_) {
         LOG(ERROR) << "Output tensor last dim mismatch. Expected " << dim0_;
         return base::error::InvalidArgument("Output tensor N-dimension mismatch");
    }

    return base::error::Success();
}

base::Status MatmulLayer::forward() {
    auto status = check();
    if (!status) {
        return status;
    }
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }
    if (is_quant_layer_) {
        kernel::get_matmul_kernel_quant8(device_type_)(
            get_input(0), get_weight(0), get_output(0), group_size_, scales_,
            cuda_config_ ? cuda_config_.get() : nullptr);
    } else {
        kernel::get_matmul_kernel(device_type_)(
            get_input(0), get_weight(0), get_output(0), 1.f,
            cuda_config_ ? cuda_config_.get() : nullptr);
    }

    if (has_bias_) {
        kernel::get_add_kernel(device_type_)(
            get_output(0), get_bias(0), get_output(0),
            cuda_config_ ? cuda_config_->stream : nullptr);
    }

    return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);

    size_t size = dim * sizeof(float);
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
    if (device_type != base::DeviceType::kDeviceUnknown) {
        buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_) {
        tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));
        // LOG(INFO) << "bias:" << bias.index<float>(0);
        bias_.at(idx) = bias;
    } else {
        // is quant layer
        tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));
        bias_.at(idx) = bias;

        const int32_t bias_size = static_cast<int32_t>(bias.size());
        CHECK(bias_size % group_size_ == 0);

        int32_t scale_nums = bias_size / group_size_;
        scales_ =
            tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                           reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
        scales_.set_device_type(device_type);
    }

    return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
    LayerParam::to_cuda();
    if (has_bias_) {
        for (auto& bias : bias_) {
            bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}

}  // namespace op