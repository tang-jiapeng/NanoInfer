/**
 * @file swiglu_kernel.cu
 * @brief CUDA SwiGLU 激活算子
 *
 * 逐元素计算：Output[i] = Swish(Input1[i]) * Input2[i]
 *   Swish(x) = x / (1 + exp(-x))
 * 使用 __expf 快速指数函数加速。
 */
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief SwiGLU 激活 CUDA Kernel
 *
 * 逐元素计算：out[idx] = Swish(in1[idx]) × in2[idx]
 *   Swish(x) = x × σ(x) = x / (1 + exp(-x))
 * 使用 __expf 硬件快速指数，精度 ~2 ULP，性能优于 expf。
 *
 * @param size  元素总数
 * @param in1   Gate 分支输入（施加 Swish）
 * @param in2   Up 分支输入（直通乘法）
 * @param out   输出
 */
__global__ void swiglu_kernel_cu_fp32(int32_t size, const float* in1, const float* in2,
                                      float* out) {
    int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) {
        return;
    }

    float x = in1[idx];
    float y = in2[idx];

    // Swish(x) = x * Sigmoid(x) = x / (1 +
    // exp(-x))
    float swish = x / (1.0f + __expf(-x));

    out[idx] = swish * y;
}

/**
 * @brief SwiGLU Host 包装函数
 *
 * 配置 Grid = ceil(size / 128), Block = 128，启动 swiglu_kernel_cu_fp32。
 *
 * @param input1  Gate 分支 Tensor，CUDA 设备
 * @param input2  Up 分支 Tensor，CUDA 设备，shape 须与 input1 一致
 * @param output  输出 Tensor，CUDA 设备
 * @param stream  CUDA Stream
 */
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