#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "../kernel_registry.h"

namespace kernel {

template <int32_t BLOCK_DIM>
__global__ void row_rmsnorm_f32(const float* input, const float* weight, float* output,
                                int hidden_dim, float eps) {
    // 1. 定位当前处理的行 (Token)
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // 指针偏移：Input/Output 随行移动，Weight 固定
    const float* row_in = input + row_idx * hidden_dim;
    float* row_out = output + row_idx * hidden_dim;
    // weight 不需要偏移，所有行共享 [hidden_dim]

    // 2. 计算 Sum of Squares (利用 float4 向量化)
    float sum_sq = 0.0f;

    // 向量化部分
    const int pack_size = 4;
    const int pack_num = hidden_dim / pack_size;
    const int pack_remainder = hidden_dim % pack_size;

    const float4* row_in_f4 = reinterpret_cast<const float4*>(row_in);

    for (int i = tid; i < pack_num; i += BLOCK_DIM) {
        float4 val = row_in_f4[i];
        sum_sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
    }

    // 尾部处理 (处理 hidden_dim 不能被 4 整除的情况)
    for (int i = pack_num * pack_size + tid; i < hidden_dim; i += BLOCK_DIM) {
        float val = row_in[i];
        sum_sq += val * val;
    }

    // 3. Block Reduce 求和
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum_sq);

    // 4. 计算 Scale (由线程 0 计算并广播)
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(block_sum / static_cast<float>(hidden_dim) + eps);
    }
    __syncthreads();

    // 5. 应用 Scale 和 Weight 并写入 Output
    const float4* weight_f4 = reinterpret_cast<const float4*>(weight);
    float4* row_out_f4 = reinterpret_cast<float4*>(row_out);

    // 向量化写入
    for (int i = tid; i < pack_num; i += BLOCK_DIM) {
        float4 in_val = row_in_f4[i];
        float4 wei_val = weight_f4[i];

        float4 out_val;
        out_val.x = in_val.x * inv_rms * wei_val.x;
        out_val.y = in_val.y * inv_rms * wei_val.y;
        out_val.z = in_val.z * inv_rms * wei_val.z;
        out_val.w = in_val.w * inv_rms * wei_val.w;

        row_out_f4[i] = out_val;
    }

    // 尾部写入
    for (int i = pack_num * pack_size + tid; i < hidden_dim; i += BLOCK_DIM) {
        row_out[i] = row_in[i] * inv_rms * weight[i];
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    // Input Shape: [total_tokens, hidden_dim] (Flattened)
    // Weight Shape: [hidden_dim]
    int32_t total_tokens = static_cast<int32_t>(input.get_dim(0));
    int32_t hidden_dim = static_cast<int32_t>(input.get_dim(1));

    // 简单的维度校验
    CHECK_EQ(weight.get_dim(0), hidden_dim) << "Weight dim mismatch";
    CHECK_EQ(input.size(), output.size());

    const float eps = 1e-6f;

    // Grid Size = total_tokens (每行一个 Block)
    constexpr int threads_per_block = 128;  // 一般 128 或 256 足够处理 hidden_dim=4096
    dim3 grid(total_tokens);
    dim3 block(threads_per_block);

    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);

    // 调用 Kernel
    row_rmsnorm_f32<threads_per_block><<<grid, block, 0, stream_>>>(
        input.ptr<float>(), weight.ptr<float>(), const_cast<float*>(output.ptr<float>()), hidden_dim, eps);
}

REGISTER_KERNEL(rmsnorm, kDeviceCUDA, rmsnorm_kernel_cu);

}  // namespace kernel