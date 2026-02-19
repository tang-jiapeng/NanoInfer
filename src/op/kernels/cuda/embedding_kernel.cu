#include "../kernel_registry.h"

namespace kernel {

__global__ void embedding_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                         const int32_t* input_ptr, const float* weight_ptr,
                                         float* output_ptr) {
    int32_t token_idx = blockIdx.x;
    if (token_idx >= token_num) {
        return;
    }
    int32_t token = input_ptr[token_idx];
    if (token >= vocab_size) {
        return;
    }

    float* output_ptr_start = output_ptr + token_idx * weight_dim;
    const float* weight_ptr_start = weight_ptr + token * weight_dim;

    for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

void embedding_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                         const tensor::Tensor& output, int32_t vocab_size, void* stream) {
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA)
        << "Input tensor must be on CUDA for embedding_kernel_cu";
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);

    const int32_t* in_ptr = input.ptr<int32_t>();
    const float* wei_ptr = weight.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    constexpr int32_t thread_num = 128;
    int32_t block_num = input_num;

    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    embedding_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
}

REGISTER_KERNEL(embedding, kDeviceCUDA, embedding_kernel_cu)

}  // namespace kernel