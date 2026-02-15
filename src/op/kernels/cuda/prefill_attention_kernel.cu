#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cfloat>
#include <cmath>

#include "nanoinfer/base/cuda_config.h"
#include "prefill_attention_kernel.cuh"

namespace kernel {

// ============================================================================
// Kernel 1: Chunked Causal Softmax
// ============================================================================
// Scores: [num_heads, chunk_len, context_len] (row-major)
// 对 chunk 中第 i 行 query (绝对位置 = start_pos + i):
//   有效 key 范围: [0, start_pos + i], 其余设为 -inf
// Grid: (chunk_len, num_heads)  每个 block 处理一行
// Block: 256 threads

__global__ void chunked_causal_softmax_kernel(float* scores, int32_t chunk_len,
                                              int32_t context_len,
                                              int32_t start_pos, float scale) {
  const int row = blockIdx.x;   // chunk 内 query 索引 (0..chunk_len-1)
  const int head = blockIdx.y;  // 当前 head
  const int tid = threadIdx.x;
  const int bs = blockDim.x;

  float* row_ptr = scores + (int64_t)head * chunk_len * context_len +
                   (int64_t)row * context_len;

  // 绝对位置 = start_pos + row → 可以看到 [0, start_pos + row]
  int valid_len = start_pos + row + 1;

  // Step 1: Scale + Mask + Find Max
  float local_max = -FLT_MAX;
  for (int j = tid; j < context_len; j += bs) {
    if (j < valid_len) {
      row_ptr[j] *= scale;
      local_max = fmaxf(local_max, row_ptr[j]);
    } else {
      row_ptr[j] = -FLT_MAX;
    }
  }

  // Block reduction for max
  __shared__ float warp_maxes[8];
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

  // Step 2: Exp + Sum
  float local_sum = 0.0f;
  for (int j = tid; j < valid_len; j += bs) {
    float val = __expf(row_ptr[j] - global_max);
    row_ptr[j] = val;
    local_sum += val;
  }
  for (int j = tid + valid_len; j < context_len; j += bs) {
    row_ptr[j] = 0.0f;
  }

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

  // Step 3: Normalize
  float inv_sum = 1.0f / (smem_sum + 1e-6f);
  for (int j = tid; j < valid_len; j += bs) {
    row_ptr[j] *= inv_sum;
  }
}

// ============================================================================
// Kernel 2: Reshape Q/K/V (带 GQA head repeat)
// ============================================================================
// 输入: [seq_len, num_input_heads * head_size]  (row-major)
// 输出: [num_heads, seq_len, head_size]
// 当 is_query=true:  num_input_heads = num_heads, kv_head = head
// 当 is_query=false: num_input_heads = num_kv_heads, kv_head = head / gqa_ratio

__global__ void reshape_qkv_kernel(const float* input, float* output,
                                   int32_t seq_len, int32_t num_heads,
                                   int32_t num_kv_heads, int32_t head_size,
                                   bool is_query) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * seq_len * head_size;
  if (idx >= total) return;

  int dim = idx % head_size;
  int pos = (idx / head_size) % seq_len;
  int head = idx / (head_size * seq_len);

  int kv_head = is_query ? head : (head / (num_heads / num_kv_heads));
  int input_heads = is_query ? num_heads : num_kv_heads;

  int64_t in_offset = (int64_t)pos * input_heads * head_size +
                      (int64_t)kv_head * head_size + dim;
  output[idx] = input[in_offset];
}

// ============================================================================
// Kernel 3: 从 [num_heads, seq_len, head_size] 转回 [seq_len, num_heads *
// head_size]
// ============================================================================
__global__ void reshape_output_kernel(const float* input, float* output,
                                      int32_t seq_len, int32_t num_heads,
                                      int32_t head_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * seq_len * head_size;
  if (idx >= total) return;

  int dim = idx % head_size;
  int pos = (idx / head_size) % seq_len;
  int head = idx / (head_size * seq_len);

  int64_t out_offset =
      (int64_t)pos * num_heads * head_size + (int64_t)head * head_size + dim;
  output[out_offset] = input[idx];
}

// ============================================================================
// Kernel 4: Prefill KV Write (单序列, 批量 tokens)
// ============================================================================
// Grid: (chunk_len, num_kv_heads)
__global__ void prefill_kv_write_kernel_cu(
    const float* __restrict__ k_src, const float* __restrict__ v_src,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ positions, int32_t num_kv_heads,
    int32_t head_size, int32_t block_size, int32_t stride_block,
    int32_t stride_token, int32_t stride_head) {
  const int32_t token_idx = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;

  const int32_t pos = positions[token_idx];
  const int32_t logical_block = pos / block_size;
  const int32_t block_offset = pos % block_size;
  const int32_t physical_block = block_table[logical_block];

  const int64_t src_offset = (int64_t)token_idx * num_kv_heads * head_size +
                             (int64_t)head_idx * head_size;
  const int64_t cache_offset = (int64_t)physical_block * stride_block +
                               (int64_t)block_offset * stride_token +
                               (int64_t)head_idx * stride_head;

  for (int i = tid; i < head_size; i += blockDim.x) {
    k_cache[cache_offset + i] = k_src[src_offset + i];
    v_cache[cache_offset + i] = v_src[src_offset + i];
  }
}

// ============================================================================
// Kernel 5: 从 Paged KV Cache 中 Gather 连续 K/V
// ============================================================================
// 将分散在不同 Block 中的 K/V 收集为连续 buffer
// cache:  [num_blocks, block_size, num_kv_heads, head_size]
// output: [context_len, num_kv_heads * head_size]  (连续)
// Grid/Block: 1-D, 总线程 = context_len * kv_dim

__global__ void gather_kv_from_cache_kernel(
    const float* __restrict__ cache, float* __restrict__ output,
    const int32_t* __restrict__ block_table, int32_t context_len,
    int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t stride_block, int32_t stride_token, int32_t stride_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int kv_dim = num_kv_heads * head_size;
  int total = context_len * kv_dim;
  if (idx >= total) return;

  int d = idx % kv_dim;    // 维度偏移 (head * head_size + dim)
  int pos = idx / kv_dim;  // token 位置 [0, context_len)

  int h = d / head_size;
  int hd = d % head_size;

  int logical_block = pos / block_size;
  int block_offset = pos % block_size;
  int physical_block = block_table[logical_block];

  int64_t cache_offset = (int64_t)physical_block * stride_block +
                         (int64_t)block_offset * stride_token +
                         (int64_t)h * stride_head + hd;

  output[idx] = cache[cache_offset];
}

// ============================================================================
// Host 函数: Chunked Prefill Attention
// ============================================================================
// 策略 (vLLM Chunked Prefill):
// 1. 将当前 chunk 的 K/V 写入 Paged Cache
// 2. 从 Paged Cache Gather 全部已缓存的 K/V [0, context_len)
// 3. Reshape Q [chunk_len] 和 gathered K/V [context_len] (含 GQA repeat)
// 4. cuBLAS: Scores = Q @ K^T → [num_heads, chunk_len, context_len]
// 5. Chunked Causal Softmax (start_pos 偏移)
// 6. cuBLAS: Output = Scores @ V → [num_heads, chunk_len, head_size]
// 7. Reshape back
//
// 内存: Score 矩阵为 O(chunk_len × context_len), 不再是 O(seq_len²)

void prefill_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& key,
    const tensor::Tensor& value, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& positions,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, int32_t context_len, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(config->cublas_handle != nullptr);

  cudaStream_t stream = config->stream;
  cublasHandle_t handle = config->cublas_handle;
  cublasSetStream(handle, stream);

  int32_t chunk_len = query.get_dim(0);  // 当前 chunk 的 token 数
  int32_t q_dim = num_heads * head_size;
  int32_t kv_dim = num_kv_heads * head_size;
  int32_t start_pos = context_len - chunk_len;  // chunk 起始绝对位置

  // Cache strides: [num_blocks, block_size, num_kv_heads, head_size]
  int32_t c_stride_head = head_size;
  int32_t c_stride_token = num_kv_heads * c_stride_head;
  int32_t c_stride_block = block_size * c_stride_token;

  // ==== Step 0: Write chunk K/V to Paged Cache ====
  {
    dim3 grid(chunk_len, num_kv_heads);
    int32_t threads = (head_size > 128) ? 256 : 128;
    prefill_kv_write_kernel_cu<<<grid, threads, 0, stream>>>(
        key.ptr<float>(), value.ptr<float>(),
        const_cast<float*>(k_cache.ptr<float>()),
        const_cast<float*>(v_cache.ptr<float>()), block_table.ptr<int32_t>(),
        positions.ptr<int32_t>(), num_kv_heads, head_size, block_size,
        c_stride_block, c_stride_token, c_stride_head);
  }

  // ==== Step 1: Gather ALL K/V from Paged Cache [0, context_len) ====
  float *k_gathered, *v_gathered;
  size_t kv_gather_bytes = (size_t)context_len * kv_dim * sizeof(float);
  cudaMallocAsync(&k_gathered, kv_gather_bytes, stream);
  cudaMallocAsync(&v_gathered, kv_gather_bytes, stream);

  {
    int total_elems = context_len * kv_dim;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;

    gather_kv_from_cache_kernel<<<blocks, threads, 0, stream>>>(
        k_cache.ptr<float>(), k_gathered, block_table.ptr<int32_t>(),
        context_len, num_kv_heads, head_size, block_size, c_stride_block,
        c_stride_token, c_stride_head);

    gather_kv_from_cache_kernel<<<blocks, threads, 0, stream>>>(
        v_cache.ptr<float>(), v_gathered, block_table.ptr<int32_t>(),
        context_len, num_kv_heads, head_size, block_size, c_stride_block,
        c_stride_token, c_stride_head);
  }

  // ==== Step 2: Reshape Q, gathered K, gathered V ====
  // Q:        [chunk_len, q_dim]     → [num_heads, chunk_len, head_size]
  // K_gather: [context_len, kv_dim]  → [num_heads, context_len, head_size] (GQA
  // repeat) V_gather: [context_len, kv_dim]  → [num_heads, context_len,
  // head_size] (GQA repeat)

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
        query.ptr<float>(), q_reshaped, chunk_len, num_heads, num_heads,
        head_size, true);

    int kv_total = num_heads * context_len * head_size;
    int kv_blocks = (kv_total + threads - 1) / threads;
    reshape_qkv_kernel<<<kv_blocks, threads, 0, stream>>>(
        k_gathered, k_reshaped, context_len, num_heads, num_kv_heads, head_size,
        false);
    reshape_qkv_kernel<<<kv_blocks, threads, 0, stream>>>(
        v_gathered, v_reshaped, context_len, num_heads, num_kv_heads, head_size,
        false);
  }

  // ==== Step 3: Batched GEMM: Scores = Q @ K^T ====
  // Q:      [num_heads, chunk_len, head_size]
  // K^T:    [num_heads, head_size, context_len]
  // Scores: [num_heads, chunk_len, context_len]

  float* scores;
  size_t scores_bytes =
      (size_t)num_heads * chunk_len * context_len * sizeof(float);
  cudaMallocAsync(&scores, scores_bytes, stream);

  float alpha = 1.0f, beta = 0.0f;
  // Row-major C = A @ B^T  →  col-major: C^T = B @ A^T
  // cuBLAS(CUBLAS_OP_T, CUBLAS_OP_N, N=context_len, M=chunk_len, K=head_size,
  //        K_ptr, lda=head_size, Q_ptr, ldb=head_size, Scores, ldc=context_len)
  cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, context_len, chunk_len, head_size,
      &alpha, k_reshaped, head_size, (int64_t)context_len * head_size,
      q_reshaped, head_size, (int64_t)chunk_len * head_size, &beta, scores,
      context_len, (int64_t)chunk_len * context_len, num_heads);

  // ==== Step 4: Chunked Causal Softmax ====
  float scale = 1.0f / sqrtf(static_cast<float>(head_size));
  dim3 softmax_grid(chunk_len, num_heads);
  chunked_causal_softmax_kernel<<<softmax_grid, 256, 0, stream>>>(
      scores, chunk_len, context_len, start_pos, scale);

  // ==== Step 5: Batched GEMM: Output = Scores @ V ====
  // Scores: [num_heads, chunk_len, context_len]
  // V:      [num_heads, context_len, head_size]
  // Output: [num_heads, chunk_len, head_size]

  float* out_reshaped;
  cudaMallocAsync(&out_reshaped, q_size, stream);  // same size as Q

  // Row-major C = A @ B  →  col-major: C^T = B^T @ A^T
  // cuBLAS(CUBLAS_OP_N, CUBLAS_OP_N, N=head_size, M=chunk_len, K=context_len,
  //        V_ptr, lda=head_size, Scores_ptr, ldb=context_len, Out,
  //        ldc=head_size)
  cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, head_size, chunk_len, context_len,
      &alpha, v_reshaped, head_size, (int64_t)context_len * head_size, scores,
      context_len, (int64_t)chunk_len * context_len, &beta, out_reshaped,
      head_size, (int64_t)chunk_len * head_size, num_heads);

  // ==== Step 6: Reshape output back ====
  // [num_heads, chunk_len, head_size] → [chunk_len, num_heads * head_size]
  {
    int total = num_heads * chunk_len * head_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_output_kernel<<<blocks, threads, 0, stream>>>(
        out_reshaped, const_cast<float*>(output.ptr<float>()), chunk_len,
        num_heads, head_size);
  }

  // ==== Cleanup ====
  cudaFreeAsync(k_gathered, stream);
  cudaFreeAsync(v_gathered, stream);
  cudaFreeAsync(q_reshaped, stream);
  cudaFreeAsync(k_reshaped, stream);
  cudaFreeAsync(v_reshaped, stream);
  cudaFreeAsync(scores, stream);
  cudaFreeAsync(out_reshaped, stream);
}

}  // namespace kernel
