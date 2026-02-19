/**
 * @file embedding_kernel.cu
 * @brief CUDA Embedding 查表算子
 *
 * 每个 Block 处理一个 Token 的 Embedding 查表：
 *   Output[token_idx] = Weight[Input[token_idx]]
 * Block 内多线程并行复制 weight_dim 个元素（128 线程/Block）。
 */
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief Embedding 查表 CUDA Kernel
 *
 * 每个 Block 处理一个 Token：读取 input_ptr[token_idx] 得到 token ID，
 * 然后从 weight_ptr 中按行复制对应 Embedding 向量到 output_ptr。
 * Block 内线程以 stride 方式并行复制 weight_dim 个 float。
 *
 * Grid : (token_num)  — 1 Block / Token
 * Block: 128 threads
 *
 * @param vocab_size   词表大小（用于越界保护）
 * @param token_num    Token 总数（= Grid 大小）
 * @param weight_dim   Embedding 维度
 * @param input_ptr    Token ID 数组 [token_num]
 * @param weight_ptr   Embedding 权重矩阵 [vocab_size, weight_dim]
 * @param output_ptr   输出矩阵 [token_num, weight_dim]
 */
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

/**
 * @brief Embedding 查表 Host 包装函数
 *
 * 从 Tensor 中提取裸指针和维度信息，配置 Grid = token_num, Block = 128，
 * 启动 embedding_kernel_cu_fp32。
 *
 * @param input       Token ID Tensor [token_num]，Int32，CUDA 设备
 * @param weight      Embedding 权重 Tensor [vocab_size, weight_dim]，CUDA 设备
 * @param output      输出 Tensor [token_num, weight_dim]，CUDA 设备
 * @param vocab_size  词表大小
 * @param stream      CUDA Stream
 */
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