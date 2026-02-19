#include "nanoinfer/op/embedding.h"
#include "kernels/kernel_registry.h"
#include "kernels/kernel_types.h"

namespace op {

EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
    reset_weight_size(1);
    reset_input_size(1);
    reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
    const auto& input_tensor = get_input(0);
    int32_t token_num = static_cast<int32_t>(input_tensor.size());

    // 检查 Input (token_ids)
    if (input_tensor.is_empty()) {
        return base::error::InvalidArgument("The input tensor is empty.");
    }

    // 检查 Weight [vocab_size, dim]
    base::Status status =
        check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    // 检查 Output [token_num, dim]
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_num, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return base::error::Success();
}

base::Status EmbeddingLayer::forward() {
    base::Status status = check();
    if (!status) {
        return status;
    }
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK(cuda_config_ != nullptr);
    }
    auto embedding_kernel = kernel::KernelRegistry::instance().get<kernel::EmbeddingKernelFn>(
        "embedding", device_type_);
    if (!embedding_kernel) {
        return base::error::InternalError("Embedding kernel not found for device: " +
                                          std::to_string(static_cast<int>(device_type_)));
    }
    embedding_kernel(get_input(0), get_weight(0), get_output(0), vocab_size_,
                     cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success();
}
}  // namespace op
