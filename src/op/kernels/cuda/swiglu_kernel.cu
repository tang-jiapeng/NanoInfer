#include "../kernel_registry.h"

namespace kernel {

__global__ void swiglu_kernel_cu_fp32(int32_t size, const float* in1, const float* in2,
                                      float* out) {
    int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    float x = in1[idx];
    float y = in2[idx];

    // Swish(x) = x * Sigmoid(x) = x / (1 + exp(-x))
    float swish = x / (1.0f + __expf(-x));

    out[idx] = swish * y;
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK_EQ(input2.is_empty(), false);
    CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

    CHECK_EQ(output.is_empty(), false);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    int size = static_cast<int32_t>(input1.size());
    int threads = 128;
    int blocks = (size + threads - 1) / threads;
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
}

REGISTER_KERNEL(swiglu, kDeviceCUDA, swiglu_kernel_cu);

}  // namespace kernel