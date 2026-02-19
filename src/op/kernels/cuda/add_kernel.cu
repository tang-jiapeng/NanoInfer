/**
 * @file add_kernel.cu
 * @brief CUDA 向量加法算子
 *
 * 简单的逐元素加法 Kernel：Output[i] = Input1[i] + Input2[i]
 * Grid-Stride 设计，512 线程/Block。
 */
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief 逐元素加法 CUDA Kernel
 *
 * 每个线程处理一个元素：output[idx] = input1[idx] + input2[idx]。
 * Grid-Stride 设计，当 size > GridDim×BlockDim 时自动循环。
 *
 * @param size    元素总数
 * @param input1  输入向量 1
 * @param input2  输入向量 2
 * @param output  输出向量（与 input 等长）
 */
__global__ void add_kernel_cu_fp32(int32_t size, const float* input1, const float* input2,
                                   float* output) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

/**
 * @brief 向量加法 Host 包装函数
 *
 * 配置 Grid = ceil(size / 512), Block = 512，启动 add_kernel_cu_fp32。
 *
 * @param input1  输入 Tensor 1，CUDA 设备
 * @param input2  输入 Tensor 2，CUDA 设备，shape 须与 input1 一致
 * @param output  输出 Tensor，CUDA 设备
 * @param stream  CUDA Stream（可为 nullptr 使用默认流）
 */
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    int32_t size = static_cast<int32_t>(input1.size());
    CHECK_EQ(size, input2.size());
    CHECK_EQ(size, output.size());

    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;

    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
}

REGISTER_KERNEL(add, kDeviceCUDA, add_kernel_cu)

}  // namespace kernel