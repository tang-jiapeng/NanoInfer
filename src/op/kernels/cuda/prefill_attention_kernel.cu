/**
 * @file prefill_attention_kernel.cu
 * @brief Chunked Prefill Attention CUDA 实现（Prefill 阶段专用）
 *
 * 整体流程（7 步流水线，参考 vLLM Chunked Prefill）：
 *   Step 0: KV Write    — 将当前 chunk 的 K/V 写入 Paged Cache
 *   Step 1: KV Gather   — 从 Paged Cache 收集 [0, context_len) 的全部 K/V
 *   Step 2: Reshape     — Q/K/V 重排为 [num_heads, seq, head_size]（含 GQA
 * repeat）
 * Step 3: GEMM Scores — cuBLAS BatchedGEMM 计算 Scores = Q @ K^T Step
 * 4: Softmax     — Chunked Causal Softmax（带 start_pos 偏移的因果掩码） Step
 * 5: GEMM Output — cuBLAS BatchedGEMM 计算 Output = Scores @ V Step 6: Reshape
 * Out — 从 head-major 转回 token-major
 *
 * 为什么叫 "Chunked"：
 *   Prefill 可能将很长的 prompt 分成多个 chunk 逐步处理。每个 chunk 只处理
 *   chunk_len 个 token，但需要看到之前所有已缓存的 key（即 context_len 个）。
 *   因此 Score 矩阵大小为 O(chunk_len × context_len)，而非 O(seq_len²)。
 *
 * Causal Mask 的偏移：
 *   chunk 内第 i 个 query 的绝对位置 = start_pos + i，
 *   它可以看到 key 位置 [0, start_pos + i]，其余设为 -inf。
 *
 * 本文件包含 5 个自定义 CUDA Kernel 和 1 个 Host 编排函数：
 *   1. chunked_causal_softmax_kernel  — 带偏移的 Causal Softmax
 *   2. reshape_qkv_kernel             — [seq, heads*dim] → [heads, seq,
 * dim]（GQA repeat）
 *   3. reshape_output_kernel          — [heads, seq, dim] → [seq, heads*dim]
 *   4. prefill_kv_write_kernel        — 批量 token KV 写入 Paged Cache
 *   5. gather_kv_from_cache_kernel    — 从 Paged Cache Scatter-Gather 连续 K/V
 *   + prefill_attention_kernel_cu     — Host 编排函数（调用上述 5 个 Kernel + 2
 * 次 cuBLAS）
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "../kernel_registry.h"


namespace kernel {

// ============================================================================
// Kernel 1: Chunked Causal Softmax
// ============================================================================
/**
 * @brief 带因果掩码与位置偏移的 Softmax（in-place）
 *
 * 输入/输出: scores [num_heads, chunk_len, context_len]  (row-major, in-place
 * 修改)
 *
 * 对 chunk 内第 i 行 query（绝对位置 = start_pos + i）：
 *   有效范围: key 位置 j ∈ [0, start_pos + i]  → 正常 softmax
 *   无效范围: j > start_pos + i                  → 设为 -inf（掩码）
 *
 * 算法分 3 步（Online Softmax 变体）：
 *   Phase 1: scale × score，同时对有效位置求 max，无效位置写 -FLT_MAX
 *   Phase 2: exp(score - max) 并求 sum，无效位置写 0
 *   Phase 3: score / sum 归一化
 *
 * Reduction 策略：Warp Shuffle + Shared Memory（8 个 Warp → 256 线程）
 *
 * Grid : (chunk_len, num_heads)  — 每个 Block 处理一行（一个 q-k 对）
 * Block: 256 threads
 */
__global__ void chunked_causal_softmax_kernel(float* scores, int32_t chunk_len, int32_t context_len,
                                              int32_t start_pos, float scale) {
    const int row = blockIdx.x;   // chunk 内 query 索引 [0, chunk_len)
    const int head = blockIdx.y;  // head 索引
    const int tid = threadIdx.x;
    const int bs = blockDim.x;  // blockDim (256)

    // 定位到当前 (head, row) 对应的 score 行首
    float* row_ptr = scores + (int64_t)head * chunk_len * context_len + (int64_t)row * context_len;

    // 该 query 的绝对位置 = start_pos + row → 可以看到 key [0, start_pos + row]
    int valid_len = start_pos + row + 1;

    // ---- Phase 1: Scale + Causal Mask + Find Row Max ----
    float local_max = -FLT_MAX;
    for (int j = tid; j < context_len; j += bs) {
        if (j < valid_len) {
            row_ptr[j] *= scale;  // 就地乘以 1/√d_k
            local_max = fmaxf(local_max, row_ptr[j]);
        } else {
            row_ptr[j] = -FLT_MAX;  // Causal Mask: 未来位置置为 -inf
        }
    }

    // Block-wide Max Reduction: Warp Shuffle → Shared Memory → 单线程汇总
    __shared__ float warp_maxes[8];  // 最多 8 个 Warp (256/32)
    __shared__ float smem_max;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    if (lane_id == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = warp_maxes[0];
        for (int w = 1; w < (bs + 31) / 32; w++) m = fmaxf(m, warp_maxes[w]);
        smem_max = m;
    }
    __syncthreads();
    float global_max = smem_max;

    // ---- Phase 2: Exp(score - max) + Sum ----
    float local_sum = 0.0f;
    for (int j = tid; j < valid_len; j += bs) {
        float val = __expf(row_ptr[j] - global_max);  // 减 max 保证数值稳定
        row_ptr[j] = val;
        local_sum += val;
    }
    // 超出 valid_len 的位置清零（exp(-inf - max) 理论为 0，但显式写 0 更安全）
    for (int j = tid + valid_len; j < context_len; j += bs) {
        row_ptr[j] = 0.0f;
    }

    // Block-wide Sum Reduction（同样 Warp Shuffle + Shared Memory）
    __shared__ float warp_sums[8];
    __shared__ float smem_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < (bs + 31) / 32; w++) s += warp_sums[w];
        smem_sum = s;
    }
    __syncthreads();

    // ---- Phase 3: Normalize ----
    float inv_sum = 1.0f / (smem_sum + 1e-6f);  // 加 epsilon 防除零
    for (int j = tid; j < valid_len; j += bs) {
        row_ptr[j] *= inv_sum;
    }
}

// ============================================================================
// Kernel 2: Reshape Q/K/V（带 GQA Head Repeat）
// ============================================================================
/**
 * @brief 将 [seq_len, heads * head_size] reshape 为 [num_heads, seq_len,
 * head_size]
 *
 * 对 Query : is_query=true,  num_input_heads = num_heads → 直接一一映射
 * 对 Key/Value: is_query=false, num_input_heads = num_kv_heads → GQA repeat
 *   kv_head = head / (num_heads / num_kv_heads)
 *   即多个 Q head 共享同一个 KV head, 在输出中将其 **复制**
 * num_heads/num_kv_heads 次
 *
 * 线程映射: 1-D 扁平化，总线程数 = num_heads × seq_len × head_size
 * 每个线程计算 (head, pos, dim) → 读取输入 [pos, kv_head * head_size + dim]
 */
__global__ void reshape_qkv_kernel(const float* input, float* output, int32_t seq_len,
                                   int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                   bool is_query) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_size;
    if (idx >= total) return;

    // 从扁平索引反解 (head, pos, dim)
    int dim = idx % head_size;
    int pos = (idx / head_size) % seq_len;
    int head = idx / (head_size * seq_len);

    // GQA: Q 直接用 head; K/V 映射到对应的 kv_head
    int kv_head = is_query ? head : (head / (num_heads / num_kv_heads));
    int input_heads = is_query ? num_heads : num_kv_heads;

    // 输入 offset: [pos, kv_head * head_size + dim]
    int64_t in_offset = (int64_t)pos * input_heads * head_size + (int64_t)kv_head * head_size + dim;
    output[idx] = input[in_offset];
}

// ============================================================================
// Kernel 3: Reshape Output（逆变换）
// ============================================================================
/**
 * @brief 将 [num_heads, seq_len, head_size] 转回 [seq_len, num_heads *
 * head_size]
 *
 * 这是 reshape_qkv_kernel 的逆操作，用于在 GEMM 后将 head-major 布局
 * 恢复为 token-major 布局，以便后续 Wo 矩阵乘法。
 */
__global__ void reshape_output_kernel(const float* input, float* output, int32_t seq_len,
                                      int32_t num_heads, int32_t head_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_size;
    if (idx >= total) return;

    int dim = idx % head_size;
    int pos = (idx / head_size) % seq_len;
    int head = idx / (head_size * seq_len);

    // 输出 offset: [pos, head * head_size + dim]
    int64_t out_offset = (int64_t)pos * num_heads * head_size + (int64_t)head * head_size + dim;
    output[out_offset] = input[idx];
}

// ============================================================================
// Kernel 4: Prefill KV Write（单序列，批量 tokens）
// ============================================================================
/**
 * @brief 将 chunk 内多个 token 的 K/V 批量写入 Paged Cache
 *
 * 与 paged_kv_writed_kernel.cu 中 Decode 版本的区别：
 *   - Decode: batch_size 个序列各写 1 个 token → Grid.x = batch_size
 *   - Prefill: 1 个序列写 chunk_len 个 token → Grid.x = chunk_len
 *   - Prefill 版 block_table 不需要 batch 维度（单序列直接索引）
 *
 * Grid : (chunk_len, num_kv_heads)
 * Block: min(head_size, 256) threads
 */
__global__ void prefill_kv_write_kernel(const float* __restrict__ k_src,
                                        const float* __restrict__ v_src,
                                        float* __restrict__ k_cache, float* __restrict__ v_cache,
                                        const int32_t* __restrict__ block_table,
                                        const int32_t* __restrict__ positions, int32_t num_kv_heads,
                                        int32_t head_size, int32_t block_size, int32_t stride_block,
                                        int32_t stride_token, int32_t stride_head) {
    const int32_t token_idx = blockIdx.x;  // chunk 内第几个 token
    const int32_t head_idx = blockIdx.y;   // KV Head 索引
    const int32_t tid = threadIdx.x;

    // 逻辑位置 → 物理地址（同 Decode 版，但 block_table 为单序列）
    const int32_t pos = positions[token_idx];
    const int32_t logical_block = pos / block_size;
    const int32_t block_offset = pos % block_size;
    const int32_t physical_block = block_table[logical_block];  // 单序列，直接索引

    // 输入 offset: [chunk_len, num_kv_heads, head_size]
    const int64_t src_offset =
        (int64_t)token_idx * num_kv_heads * head_size + (int64_t)head_idx * head_size;
    // Cache offset: [num_blocks, block_size, num_kv_heads, head_size]
    const int64_t cache_offset = (int64_t)physical_block * stride_block +
                                 (int64_t)block_offset * stride_token +
                                 (int64_t)head_idx * stride_head;

    for (int i = tid; i < head_size; i += blockDim.x) {
        k_cache[cache_offset + i] = k_src[src_offset + i];
        v_cache[cache_offset + i] = v_src[src_offset + i];
    }
}

// ============================================================================
// Kernel 5: Gather KV from Paged Cache
// ============================================================================
/**
 * @brief 从 Paged Cache 中 Scatter-Gather 出连续 K/V 缓冲区
 *
 * Paged Cache 中 token 分散在不同的物理 Block 中，cuBLAS GEMM 需要连续内存。
 * 本 Kernel 将 [0, context_len) 内所有 token 的 K（或 V）收集为：
 *   output: [context_len, num_kv_heads * head_size]  (连续 row-major)
 *
 * 地址映射: pos → logical_block → physical_block → cache offset
 *
 * 线程映射: 1-D 扁平化，总元素 = context_len × kv_dim
 * 每个线程读取 cache 的一个 float 并写到 output 对应位置
 */
__global__ void gather_kv_from_cache_kernel(const float* __restrict__ cache,
                                            float* __restrict__ output,
                                            const int32_t* __restrict__ block_table,
                                            int32_t context_len, int32_t num_kv_heads,
                                            int32_t head_size, int32_t block_size,
                                            int32_t stride_block, int32_t stride_token,
                                            int32_t stride_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_dim = num_kv_heads * head_size;
    int total = context_len * kv_dim;
    if (idx >= total) return;

    // 从扁平索引反解 (pos, head, dim)
    int d = idx % kv_dim;    // kv_dim 内的偏移
    int pos = idx / kv_dim;  // token 位置 [0, context_len)

    int h = d / head_size;   // KV head 索引
    int hd = d % head_size;  // head 内维度偏移

    // 逻辑位置 → 物理地址
    int logical_block = pos / block_size;
    int block_offset = pos % block_size;
    int physical_block = block_table[logical_block];

    int64_t cache_offset = (int64_t)physical_block * stride_block +
                           (int64_t)block_offset * stride_token + (int64_t)h * stride_head + hd;

    output[idx] = cache[cache_offset];
}

// ============================================================================
// Host 编排函数: Chunked Prefill Attention
// ============================================================================
/**
 * @brief Prefill 阶段的 Attention 编排（7 步流水线）
 *
 * 整体数据流:
 *   输入 Q: [chunk_len, q_dim]        — 当前 chunk 的 Query
 *   输入 K: [chunk_len, kv_dim]       — 当前 chunk 的 Key（已经过 RoPE）
 *   输入 V: [chunk_len, kv_dim]       — 当前 chunk 的 Value
 *   输出  : [chunk_len, q_dim]        — Attention 结果
 *
 * 中间变量（全部 cudaMallocAsync / cudaFreeAsync 管理）：
 *   k_gathered, v_gathered: [context_len, kv_dim]                       — Step
 * 1 q_reshaped:             [num_heads, chunk_len, head_size]           — Step
 * 2 k_reshaped, v_reshaped: [num_heads, context_len, head_size]        — Step 2
 * (含 GQA repeat) scores:                 [num_heads, chunk_len, context_len]
 * — Step 3 out_reshaped:           [num_heads, chunk_len, head_size] — Step 5
 *
 * cuBLAS Row-Major 技巧：
 *   cuBLAS 只支持 Column-Major，但通过利用 C^T = (B^T @ A^T)^T / C^T = B @ A^T
 *   可以直接在 Row-Major 布局上调用。具体推导见 Step 3/5 的注释。
 *
 * @param context_len 当前序列已缓存的总 token 数（含本 chunk）
 * @param config      CudaConfig*，包含 stream 和 cuBLAS handle
 */
void prefill_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& key,
                                 const tensor::Tensor& value, const tensor::Tensor& output,
                                 const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                 const tensor::Tensor& block_table, const tensor::Tensor& positions,
                                 int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                 int32_t block_size, int32_t context_len, void* config) {
    CHECK(config != nullptr);
    CHECK(static_cast<const CudaConfig*>(config)->cublas_handle != nullptr);

    cudaStream_t stream = static_cast<const CudaConfig*>(config)->stream;
    cublasHandle_t handle = static_cast<const CudaConfig*>(config)->cublas_handle;
    cublasSetStream(handle, stream);

    int32_t chunk_len = query.get_dim(0);  // 当前 chunk 的 token 数
    int32_t q_dim = num_heads * head_size;
    int32_t kv_dim = num_kv_heads * head_size;
    int32_t start_pos = context_len - chunk_len;  // chunk 的起始绝对位置

    // Cache strides: [num_blocks, block_size, num_kv_heads, head_size]
    int32_t c_stride_head = head_size;
    int32_t c_stride_token = num_kv_heads * c_stride_head;
    int32_t c_stride_block = block_size * c_stride_token;

    // ===== Step 0: 将当前 chunk 的 K/V 写入 Paged Cache =====
    // 写入后 Cache 中 [0, context_len) 的数据才完整
    {
        dim3 grid(chunk_len, num_kv_heads);
        int32_t threads = (head_size > 128) ? 256 : 128;
        prefill_kv_write_kernel<<<grid, threads, 0, stream>>>(
            key.ptr<float>(), value.ptr<float>(), const_cast<float*>(k_cache.ptr<float>()),
            const_cast<float*>(v_cache.ptr<float>()), block_table.ptr<int32_t>(),
            positions.ptr<int32_t>(), num_kv_heads, head_size, block_size, c_stride_block,
            c_stride_token, c_stride_head);
    }

    // ===== Step 1: 从 Paged Cache Gather 全部 K/V [0, context_len) =====
    // Paged Cache 中 token 分散在不同物理 Block，GEMM 需要连续内存
    float *k_gathered, *v_gathered;
    size_t kv_gather_bytes = (size_t)context_len * kv_dim * sizeof(float);
    cudaMallocAsync(&k_gathered, kv_gather_bytes, stream);
    cudaMallocAsync(&v_gathered, kv_gather_bytes, stream);

    {
        int total_elems = context_len * kv_dim;
        int threads = 256;
        int blocks = (total_elems + threads - 1) / threads;

        gather_kv_from_cache_kernel<<<blocks, threads, 0, stream>>>(
            k_cache.ptr<float>(), k_gathered, block_table.ptr<int32_t>(), context_len, num_kv_heads,
            head_size, block_size, c_stride_block, c_stride_token, c_stride_head);

        gather_kv_from_cache_kernel<<<blocks, threads, 0, stream>>>(
            v_cache.ptr<float>(), v_gathered, block_table.ptr<int32_t>(), context_len, num_kv_heads,
            head_size, block_size, c_stride_block, c_stride_token, c_stride_head);
    }

    // ===== Step 2: Reshape Q, K, V（含 GQA Head Repeat）=====
    // Q:        [chunk_len, q_dim]    → [num_heads, chunk_len, head_size]
    // 直接映射 K_gather: [context_len, kv_dim] → [num_heads, context_len,
    // head_size]    GQA repeat V_gather: [context_len, kv_dim] → [num_heads,
    // context_len, head_size]    GQA repeat

    float *q_reshaped, *k_reshaped, *v_reshaped;
    size_t q_size = (size_t)num_heads * chunk_len * head_size * sizeof(float);
    size_t kv_size = (size_t)num_heads * context_len * head_size * sizeof(float);
    cudaMallocAsync(&q_reshaped, q_size, stream);
    cudaMallocAsync(&k_reshaped, kv_size, stream);
    cudaMallocAsync(&v_reshaped, kv_size, stream);

    {
        int threads = 256;

        int q_total = num_heads * chunk_len * head_size;
        int q_blocks = (q_total + threads - 1) / threads;
        reshape_qkv_kernel<<<q_blocks, threads, 0, stream>>>(
            query.ptr<float>(), q_reshaped, chunk_len, num_heads, num_heads, head_size, true);

        int kv_total = num_heads * context_len * head_size;
        int kv_blocks = (kv_total + threads - 1) / threads;
        reshape_qkv_kernel<<<kv_blocks, threads, 0, stream>>>(
            k_gathered, k_reshaped, context_len, num_heads, num_kv_heads, head_size, false);
        reshape_qkv_kernel<<<kv_blocks, threads, 0, stream>>>(
            v_gathered, v_reshaped, context_len, num_heads, num_kv_heads, head_size, false);
    }

    // ===== Step 3: Batched GEMM — Scores = Q @ K^T =====
    //   Q:      [num_heads, chunk_len,   head_size]
    //   K^T:    [num_heads, head_size,   context_len]
    //   Scores: [num_heads, chunk_len,   context_len]
    //
    // cuBLAS Row-Major 技巧:
    //   Row-Major: C(M×N) = A(M×K) @ B(K×N)^T
    //   等价 Col-Major: C^T(N×M) = B(N×K) @ A^T(K×M)
    //   因此调用: cublas(OP_T, OP_N, N=context_len, M=chunk_len, K=head_size,
    //                    B=K_ptr, lda=head_size, A=Q_ptr, ldb=head_size,
    //                    C=Scores, ldc=context_len)

    float* scores;
    size_t scores_bytes = (size_t)num_heads * chunk_len * context_len * sizeof(float);
    cudaMallocAsync(&scores, scores_bytes, stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, context_len, chunk_len, head_size,
                              &alpha, k_reshaped, head_size, (int64_t)context_len * head_size,
                              q_reshaped, head_size, (int64_t)chunk_len * head_size, &beta, scores,
                              context_len, (int64_t)chunk_len * context_len, num_heads);

    // ===== Step 4: Chunked Causal Softmax =====
    // scale = 1/√head_size，带 start_pos 偏移的因果掩码
    float scale = 1.0f / sqrtf(static_cast<float>(head_size));
    dim3 softmax_grid(chunk_len, num_heads);
    chunked_causal_softmax_kernel<<<softmax_grid, 256, 0, stream>>>(scores, chunk_len, context_len,
                                                                    start_pos, scale);

    // ===== Step 5: Batched GEMM — Output = Scores @ V =====
    //   Scores: [num_heads, chunk_len,   context_len]
    //   V:      [num_heads, context_len, head_size]
    //   Output: [num_heads, chunk_len,   head_size]
    //
    // cuBLAS Row-Major 技巧:
    //   Row-Major: C(M×N) = A(M×K) @ B(K×N)
    //   等价 Col-Major: C^T(N×M) = B^T(N×K) @ A^T(K×M)
    //   但由于 B(V) 已经是 (context_len, head_size) Row-Major
    //     = (head_size, context_len) Col-Major
    //   因此调用: cublas(OP_N, OP_N, N=head_size, M=chunk_len, K=context_len,
    //                    B=V_ptr, lda=head_size, A=Scores, ldb=context_len,
    //                    C=Out, ldc=head_size)

    float* out_reshaped;
    cudaMallocAsync(&out_reshaped, q_size, stream);  // 与 Q 同尺寸

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, chunk_len, context_len,
                              &alpha, v_reshaped, head_size, (int64_t)context_len * head_size,
                              scores, context_len, (int64_t)chunk_len * context_len, &beta,
                              out_reshaped, head_size, (int64_t)chunk_len * head_size, num_heads);

    // ===== Step 6: Reshape 回 token-major 布局 =====
    // [num_heads, chunk_len, head_size] → [chunk_len, num_heads * head_size]
    {
        int total = num_heads * chunk_len * head_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_output_kernel<<<blocks, threads, 0, stream>>>(
            out_reshaped, const_cast<float*>(output.ptr<float>()), chunk_len, num_heads, head_size);
    }

    // ===== Cleanup: 释放所有临时 GPU 缓冲区 =====
    cudaFreeAsync(k_gathered, stream);
    cudaFreeAsync(v_gathered, stream);
    cudaFreeAsync(q_reshaped, stream);
    cudaFreeAsync(k_reshaped, stream);
    cudaFreeAsync(v_reshaped, stream);
    cudaFreeAsync(scores, stream);
    cudaFreeAsync(out_reshaped, stream);
}

REGISTER_KERNEL(prefill_attention, kDeviceCUDA, prefill_attention_kernel_cu);

}  // namespace kernel
