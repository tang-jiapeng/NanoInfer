/**
 * @file paged_attention_kernel.cu
 * @brief PagedAttention V1 CUDA Kernel（Decode 阶段专用）
 *
 * 算法概述：
 *   在 Decode 阶段，每个序列每步只产生 1 个新 Query token，需要与该序列
 *   历史所有 context_len 个 K/V token 做 Attention。Key/Value 以非连续的
 *   "物理块" 形式存储在显存池中（即 PagedAttention 机制）。
 *
 * 计算流程（对每个 Query Head）：
 *   1. 将 Query 向量加载到 Shared Memory
 *   2. 遍历 KV Cache：查 block_table 将逻辑位置 t → 物理块地址，
 *      计算 score[t] = Q · K[t]
 *   3. Softmax(score * scale)
 *   4. 加权求和 output = Σ prob[t] * V[t]
 *
 * KV Cache 物理布局：
 *   [num_blocks, block_size, num_kv_heads, head_size]
 *   stride_block = block_size * num_kv_heads * head_size
 *   stride_token = num_kv_heads * head_size
 *   stride_head  = head_size
 *
 * 线程映射：
 *   Grid  = (num_heads, batch_size)    → 每个 Block 处理 1 个 (head, seq) 对
 *   Block = 128 threads                → 并行遍历序列维度 & Head 维度
 *
 * GQA 支持：
 *   num_heads > num_kv_heads 时，多个 Query Head 共享同一个 KV Head
 *   kv_head_idx = head_idx / (num_heads / num_kv_heads)
 *
 * 局限性（V1 版本）：
 *   Softmax 的 logits 全部存在 Dynamic Shared Memory 中，
 *   大小为 max_context_len * sizeof(float)。当序列非常长时
 *   （如 >12K）可能超出 Shared Memory 上限，需要 V2（Block-wise Reduction）。
 */
#include <cuda_runtime.h>
#include <cfloat>
#include <cub/block/block_reduce.cuh>
#include "../kernel_registry.h"
namespace kernel {

/**
 * @brief 原地 Softmax（Block 协作版）
 *
 * 在 Shared Memory 中对 logits[0..size-1] 执行 Scaled Softmax：
 *   1. 每个元素乘以 scale（即 1/√head_size），同时找全局最大值
 *   2. 所有元素减去最大值后 exp，求和
 *   3. 除以求和值完成归一化
 *
 * 使用 CUB BlockReduce 做 Max 和 Sum 的 Block 级别 Reduction。
 *
 * @tparam BLOCK_SIZE 线程块大小（必须与 launch 配置一致）
 */
template <int BLOCK_SIZE>
__device__ void softmax_inplace(float* __restrict__ logits, int size, float scale) {
    const int tid = threadIdx.x;

    // ---- Phase 1: 缩放并求全局最大值（数值稳定性） ----
    float local_max = -FLT_MAX;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        logits[i] *= scale;
        local_max = fmaxf(local_max, logits[i]);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float global_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

    __shared__ float shared_max;
    if (tid == 0) shared_max = global_max;
    __syncthreads();
    global_max = shared_max;

    // ---- Phase 2: exp(x - max) 并求和 ----
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        float val = __expf(logits[i] - global_max);
        logits[i] = val;
        local_sum += val;
    }

    __syncthreads();
    float global_sum = BlockReduce(temp_storage).Sum(local_sum);

    __shared__ float shared_sum;
    if (tid == 0) shared_sum = global_sum;
    __syncthreads();
    global_sum = shared_sum;

    // ---- Phase 3: 归一化 ----
    float inv_sum = 1.0f / (global_sum + 1e-6f);
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        logits[i] *= inv_sum;
    }
}

/**
 * @brief PagedAttention V1 核心 Kernel
 *
 * 每个 CUDA Block 负责一个 (Query Head, Sequence) 对的完整 Attention 计算：
 *   1. 加载 Q 到 Shared Memory（向量化 float4 加载）
 *   2. 遍历 context_len 个历史 token，查 block_table 做地址转换，
 *      计算 score = Q · K（点积，循环展开优化）
 *   3. Softmax
 *   4. 遍历 V，加权求和写入 output
 *
 * @tparam BLOCK_SIZE    线程块大小（128）
 * @tparam HEAD_SIZE     编译期 Head 维度（64/80/96/128/256）
 * @tparam KV_BLOCK_SIZE KV Cache 的 Page 大小（16/32）
 */
template <int BLOCK_SIZE, int HEAD_SIZE, int KV_BLOCK_SIZE>
__global__ void paged_attention_kernel_v1(
    float* __restrict__ output, const float* __restrict__ query, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens, int32_t num_kv_heads, int32_t max_blocks_per_seq,
    float scale, int32_t stride_block, int32_t stride_token, int32_t stride_head) {
    const int head_idx = blockIdx.x;  // 当前 Query Head 索引
    const int seq_idx = blockIdx.y;   // 当前序列索引
    const int tid = threadIdx.x;

    const int seq_len = context_lens[seq_idx];
    if (seq_len == 0) return;

    // ---- GQA 映射：Query Head → KV Head ----
    // 例如 num_heads=32, num_kv_heads=8 → 每 4 个 Q Head 共享 1 个 KV Head
    const int num_heads = gridDim.x;
    const int heads_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / heads_per_kv;

    // ---- 将 Q 加载到 Shared Memory ----
    __shared__ float smem_q[HEAD_SIZE];
    const int q_offset = seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

    // 向量化加载：每次读 float4（16 bytes），减少 Global Memory 访问次数
    const float4* q_ptr_4 = reinterpret_cast<const float4*>(query + q_offset);
    float4* smem_q_4 = reinterpret_cast<float4*>(smem_q);
    for (int i = tid; i < HEAD_SIZE / 4; i += BLOCK_SIZE) {
        smem_q_4[i] = q_ptr_4[i];
    }
    __syncthreads();

    // ---- 计算 Attention Score: score[t] = Q · K[t] ----
    // logits 存储在 Dynamic Shared Memory 中（大小 = max_context_len * 4B）
    extern __shared__ float smem_logits[];

    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        // 逻辑位置 t → 物理地址：
        //   逻辑块号 = t / KV_BLOCK_SIZE
        //   块内偏移 = t % KV_BLOCK_SIZE
        //   物理块号 = block_table[seq_idx * max_blocks_per_seq + 逻辑块号]
        const int log_block_idx = t / KV_BLOCK_SIZE;
        const int block_offset = t % KV_BLOCK_SIZE;
        const int phys_block_idx = block_table[seq_idx * max_blocks_per_seq + log_block_idx];

        // 根据 Cache 布局 [num_blocks, block_size, num_kv_heads, head_size]
        // 计算偏移
        int64_t base_offset = static_cast<int64_t>(phys_block_idx) * stride_block +
                              static_cast<int64_t>(block_offset) * stride_token +
                              static_cast<int64_t>(kv_head_idx) * stride_head;

        const float* k_ptr = k_cache + base_offset;

        // 点积：Q · K[t]
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < HEAD_SIZE; ++i) {
            score += smem_q[i] * k_ptr[i];
        }
        smem_logits[t] = score;
    }
    __syncthreads();

    // ---- Softmax(score * scale) ----
    softmax_inplace<BLOCK_SIZE>(smem_logits, seq_len, scale);
    __syncthreads();

    // ---- 加权求和: output = Σ prob[t] * V[t] ----
    // 外层：每个线程负责 output 向量的一部分维度
    // 内层：遍历整个序列做加权求和
    float acc_val = 0.0f;
    for (int i = tid; i < HEAD_SIZE; i += BLOCK_SIZE) {
        acc_val = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            float prob = smem_logits[t];

            const int log_block_idx = t / KV_BLOCK_SIZE;
            const int block_offset = t % KV_BLOCK_SIZE;
            const int phys_block_idx = block_table[seq_idx * max_blocks_per_seq + log_block_idx];

            int64_t base_offset = static_cast<int64_t>(phys_block_idx) * stride_block +
                                  static_cast<int64_t>(block_offset) * stride_token +
                                  static_cast<int64_t>(kv_head_idx) * stride_head;

            const float* v_ptr = v_cache + base_offset;
            acc_val += prob * v_ptr[i];
        }
        output[q_offset + i] = acc_val;
    }
}

// 启动宏：简化模板参数 Dispatch
#define LAUNCH_ATTENTION_V1(HEAD_SIZE_VAL, BLOCK_SIZE_VAL)                                     \
    paged_attention_kernel_v1<128, HEAD_SIZE_VAL, BLOCK_SIZE_VAL>                              \
        <<<grid, 128, smem_size, cuda_stream>>>(                                               \
            const_cast<float*>(output.ptr<float>()), query.ptr<float>(), k_cache.ptr<float>(), \
            v_cache.ptr<float>(), block_table.ptr<int32_t>(), context_lens.ptr<int32_t>(),     \
            num_kv_heads, max_blocks_per_seq, scale, stride_block, stride_token, stride_head)

/**
 * @brief PagedAttention V1 Host 入口
 *
 * 根据 head_size 和 block_size 做二级 switch-case 编译期模板 Dispatch。
 * HEAD_SIZE / KV_BLOCK_SIZE 作为模板参数，使 Kernel 内部循环展开和
 * Shared Memory 分配在编译期确定。
 *
 * Dynamic Shared Memory 大小 = max_context_len * sizeof(float)，
 * 用于存放 Attention Score（V1 的 Shared Memory 瓶颈所在）。
 */
void paged_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& output,
                               const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                               const tensor::Tensor& block_table,
                               const tensor::Tensor& context_lens, int32_t max_context_len,
                               int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                               int32_t block_size, float scale, void* stream) {
    int32_t batch_size = static_cast<int32_t>(query.get_dim(0));
    int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));

    // Cache 布局 strides: [num_blocks, block_size, num_kv_heads, head_size]
    int32_t stride_head = head_size;
    int32_t stride_token = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token;

    // Grid: 每个 Block 处理一个 (head, seq) 对
    dim3 grid(num_heads, batch_size);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Dynamic Shared Memory：存放 logits[max_context_len]
    // 4096 tokens × 4B = 16KB (安全), 8192 × 4B = 32KB (安全)
    // 超长序列可能溢出 → 需 V2 (Block-wise Reduction)
    size_t smem_size = max_context_len * sizeof(float);

    // 二级 Dispatch: block_size → head_size
    switch (block_size) {
        case 16:
            switch (head_size) {
                case 64:
                    LAUNCH_ATTENTION_V1(64, 16);
                    break;
                case 80:
                    LAUNCH_ATTENTION_V1(80, 16);
                    break;
                case 96:
                    LAUNCH_ATTENTION_V1(96, 16);
                    break;
                case 112:
                    LAUNCH_ATTENTION_V1(112, 16);
                    break;
                case 128:
                    LAUNCH_ATTENTION_V1(128, 16);
                    break;
                case 256:
                    LAUNCH_ATTENTION_V1(256, 16);
                    break;
                default:
                    LOG(FATAL) << "Unsupported Head Size: " << head_size << " for Block Size 16";
            }
            break;

        case 32:
            switch (head_size) {
                case 64:
                    LAUNCH_ATTENTION_V1(64, 32);
                    break;
                case 80:
                    LAUNCH_ATTENTION_V1(80, 32);
                    break;
                case 96:
                    LAUNCH_ATTENTION_V1(96, 32);
                    break;
                case 112:
                    LAUNCH_ATTENTION_V1(112, 32);
                    break;
                case 128:
                    LAUNCH_ATTENTION_V1(128, 32);
                    break;
                case 256:
                    LAUNCH_ATTENTION_V1(256, 32);
                    break;
                default:
                    LOG(FATAL) << "Unsupported Head Size: " << head_size << " for Block Size 32";
            }
            break;

        default:
            LOG(FATAL) << "Unsupported Block Size: " << block_size
                       << ". Only 16 and 32 are supported.";
    }
}

REGISTER_KERNEL(paged_attention, kDeviceCUDA, paged_attention_kernel_cu);
}  // namespace kernel