#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cfloat>
#include <cmath>
#include "nanoinfer/base/cuda_config.h"
#include "prefill_attention_kernel.cuh"

namespace kernel {

// ============================================================================
// Kernel 1: Causal Mask + Scale + Row-wise Softmax
// ============================================================================
// Scores: [num_heads, seq_len, seq_len] (row-major)
// 对每一行做: mask (j > i 的位置设为 -inf) -> scale -> softmax
// Grid: (seq_len, num_heads)  每个 block 处理一行
// Block: 256 threads

__global__ void causal_softmax_kernel(float* scores, int32_t seq_len,
                                      float scale) {
  const int row = blockIdx.x;   // 当前 query 位置 (0..seq_len-1)
  const int head = blockIdx.y;  // 当前 head
  const int tid = threadIdx.x;
  const int blockSize = blockDim.x;

  // scores 偏移: head * seq_len * seq_len + row * seq_len
  float* row_ptr =
      scores + (int64_t)head * seq_len * seq_len + (int64_t)row * seq_len;

  // 有效长度: 因果 mask 下, query at position row 只能看到 [0..row]
  int valid_len = row + 1;

  // Step 1: Scale + Mask + Find Max
  float local_max = -FLT_MAX;
  for (int j = tid; j < seq_len; j += blockSize) {
    if (j < valid_len) {
      row_ptr[j] *= scale;
      local_max = fmaxf(local_max, row_ptr[j]);
    } else {
      row_ptr[j] = -FLT_MAX;
    }
  }

  // Warp reduction for max
  __shared__ float smem_max;
  // Simple block reduction
  for (int offset = blockSize / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
    local_max = fmaxf(local_max, other);
  }
  // Write warp 0 leaders to shared memory, then reduce across warps
  __shared__ float warp_maxes[8];  // max 8 warps for blockDim <= 256
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  if (lane_id == 0) warp_maxes[warp_id] = local_max;
  __syncthreads();
  if (tid == 0) {
    float m = warp_maxes[0];
    for (int w = 1; w < (blockSize + 31) / 32; w++) {
      m = fmaxf(m, warp_maxes[w]);
    }
    smem_max = m;
  }
  __syncthreads();
  float global_max = smem_max;

  // Step 2: Exp + Sum
  float local_sum = 0.0f;
  for (int j = tid; j < valid_len; j += blockSize) {
    float val = __expf(row_ptr[j] - global_max);
    row_ptr[j] = val;
    local_sum += val;
  }
  // Zero out masked positions (already -inf -> exp ~ 0, but be explicit)
  for (int j = tid + valid_len; j < seq_len; j += blockSize) {
    row_ptr[j] = 0.0f;
  }

  // Block sum reduction
  __shared__ float smem_sum;
  __shared__ float warp_sums[8];
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
  }
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    for (int w = 0; w < (blockSize + 31) / 32; w++) {
      s += warp_sums[w];
    }
    smem_sum = s;
  }
  __syncthreads();

  // Step 3: Normalize
  float inv_sum = 1.0f / (smem_sum + 1e-6f);
  for (int j = tid; j < valid_len; j += blockSize) {
    row_ptr[j] *= inv_sum;
  }
}

// ============================================================================
// Kernel 2: 扩展 K/V (GQA head repeat)
// ============================================================================
// 将 [seq_len, num_kv_heads, head_size] -> [num_heads, seq_len, head_size]
// (通过 head repeat 实现 GQA)
// 同时做转置: 输入是 [seq_len, num_kv_heads * head_size]
// 输出     是 [num_heads, seq_len, head_size]

__global__ void reshape_qkv_kernel(
    const float* input,  // [seq_len, total_dim]
    float* output,       // [num_heads, seq_len, head_size]
    int32_t seq_len, int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    bool is_query) {
  // Global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_heads * seq_len * head_size;
  if (idx >= total) return;

  // Decode: idx -> (head, pos, dim)
  int dim = idx % head_size;
  int pos = (idx / head_size) % seq_len;
  int head = idx / (head_size * seq_len);

  int kv_head = is_query ? head : (head / (num_heads / num_kv_heads));
  int input_heads = is_query ? num_heads : num_kv_heads;

  // input offset: pos * (input_heads * head_size) + kv_head * head_size + dim
  int64_t in_offset = (int64_t)pos * input_heads * head_size +
                      (int64_t)kv_head * head_size + dim;

  output[idx] = input[in_offset];
}

// ============================================================================
// Kernel 3: 从 [num_heads, seq_len, head_size] 转回 [seq_len, num_heads *
// head_size]
// ============================================================================
__global__ void reshape_output_kernel(
    const float* input,  // [num_heads, seq_len, head_size]
    float* output,       // [seq_len, num_heads * head_size]
    int32_t seq_len, int32_t num_heads, int32_t head_size) {
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
// Kernel 4: Prefill KV Write
// ============================================================================
// 专用于 prefill 阶段: 所有 token 属于同一个序列 (batch=1)
// 不同于 decode 阶段的 paged_kv_write_kernel (每个 grid.x 是不同的序列)
// Grid: (seq_len, num_kv_heads)
__global__ void prefill_kv_write_kernel_cu(
    const float* __restrict__ k_src,  // [seq_len, num_kv_heads * head_size]
    const float* __restrict__ v_src,
    float* __restrict__ k_cache,  // [num_blocks, block_size, num_kv_heads,
                                  // head_size]
    float* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,  // [1, max_blocks_per_seq]
    const int32_t* __restrict__ positions,    // [seq_len]
    int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t stride_block, int32_t stride_token, int32_t stride_head) {
  const int32_t token_idx = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;

  const int32_t pos = positions[token_idx];
  const int32_t logical_block = pos / block_size;
  const int32_t block_offset = pos % block_size;

  // block_table 只有 1 行 (batch=1), 直接用 logical_block 索引
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
// Host 函数: Prefill Attention
// ============================================================================
// 策略:
// 1. Reshape Q/K/V from [seq_len, dim] -> [num_heads, seq_len, head_size]
// 2. 用 cublasSgemmStridedBatched 计算 Q @ K^T -> Scores [num_heads, seq_len,
// seq_len]
// 3. Causal mask + softmax
// 4. 用 cublasSgemmStridedBatched 计算 Scores @ V -> Output [num_heads,
// seq_len, head_size]
// 5. Reshape output back
// 6. 同时将 K/V 写入 Paged Cache

void prefill_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& key,
    const tensor::Tensor& value, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& positions,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    int32_t block_size, const CudaConfig* config) {
  CHECK(config != nullptr);
  CHECK(config->cublas_handle != nullptr);

  cudaStream_t stream = config->stream;
  cublasHandle_t handle = config->cublas_handle;
  cublasSetStream(handle, stream);

  int32_t seq_len = query.get_dim(0);  // total_tokens for this prefill
  int32_t q_dim = num_heads * head_size;
  int32_t kv_dim = num_kv_heads * head_size;

  // ---- Step 0: Write K/V to Paged Cache ----
  // 使用专用的 prefill KV write kernel (所有 token 属于 batch=1 的同一序列)
  {
    int32_t stride_head = head_size;
    int32_t stride_token = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token;

    dim3 grid(seq_len, num_kv_heads);
    int32_t threads = 128;
    if (head_size > 128) threads = 256;

    prefill_kv_write_kernel_cu<<<grid, threads, 0, stream>>>(
        key.ptr<float>(), value.ptr<float>(),
        const_cast<float*>(k_cache.ptr<float>()),
        const_cast<float*>(v_cache.ptr<float>()), block_table.ptr<int32_t>(),
        positions.ptr<int32_t>(), num_kv_heads, head_size, block_size,
        stride_block, stride_token, stride_head);
  }

  // ---- Step 1: Reshape Q, K, V ----
  // Q: [seq_len, q_dim] -> [num_heads, seq_len, head_size]
  // K: [seq_len, kv_dim] -> [num_heads, seq_len, head_size] (with GQA repeat)
  // V: same as K

  float *q_reshaped, *k_reshaped, *v_reshaped;
  size_t qkv_size = (size_t)num_heads * seq_len * head_size * sizeof(float);
  cudaMallocAsync(&q_reshaped, qkv_size, stream);
  cudaMallocAsync(&k_reshaped, qkv_size, stream);
  cudaMallocAsync(&v_reshaped, qkv_size, stream);

  int total_elements = num_heads * seq_len * head_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  reshape_qkv_kernel<<<blocks, threads, 0, stream>>>(
      query.ptr<float>(), q_reshaped, seq_len, num_heads, num_heads, head_size,
      true);
  reshape_qkv_kernel<<<blocks, threads, 0, stream>>>(
      key.ptr<float>(), k_reshaped, seq_len, num_heads, num_kv_heads, head_size,
      false);
  reshape_qkv_kernel<<<blocks, threads, 0, stream>>>(
      value.ptr<float>(), v_reshaped, seq_len, num_heads, num_kv_heads,
      head_size, false);

  // ---- Step 2: Batched GEMM: Scores = Q @ K^T ----
  // Q:      [num_heads, seq_len, head_size] (M=seq_len, K=head_size)
  // K^T:    [num_heads, head_size, seq_len] (K=head_size, N=seq_len)
  // Scores: [num_heads, seq_len, seq_len]   (M=seq_len, N=seq_len)

  float* scores;
  size_t scores_size = (size_t)num_heads * seq_len * seq_len * sizeof(float);
  cudaMallocAsync(&scores, scores_size, stream);

  float alpha = 1.0f, beta = 0.0f;
  // cuBLAS is column-major, so we compute: Scores^T = K * Q^T
  // which gives us Scores in row-major
  // Actually for row-major: C = A * B^T
  // cublasSgemmStridedBatched(..., CUBLAS_OP_T, CUBLAS_OP_N, ...)
  // We want: Scores[h] = Q[h] @ K[h]^T
  // In col-major cuBLAS: C^T = (Q @ K^T)^T = K @ Q^T
  // So: cublas(CUBLAS_OP_T, CUBLAS_OP_N, N=seq_len, M=seq_len, K=head_size,
  //           K_ptr, lda=head_size, Q_ptr, ldb=head_size, Scores_ptr,
  //           ldc=seq_len)
  cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_T,  // op(A) = K^T
      CUBLAS_OP_N,  // op(B) = Q
      seq_len,      // N (cols of C in col-major = cols of Scores)
      seq_len,      // M (rows of C in col-major = rows of Scores)
      head_size,    // K
      &alpha, k_reshaped, head_size,
      (int64_t)seq_len * head_size,  // A = K, lda=head_size, stride
      q_reshaped, head_size,
      (int64_t)seq_len * head_size,  // B = Q, ldb=head_size, stride
      &beta, scores, seq_len,
      (int64_t)seq_len * seq_len,  // C = Scores, ldc=seq_len, stride
      num_heads                    // batch count
  );

  // ---- Step 3: Causal Mask + Softmax ----
  float scale = 1.0f / sqrtf(static_cast<float>(head_size));
  dim3 softmax_grid(seq_len, num_heads);
  causal_softmax_kernel<<<softmax_grid, 256, 0, stream>>>(scores, seq_len,
                                                          scale);

  // ---- Step 4: Batched GEMM: Output = Scores @ V ----
  // Scores: [num_heads, seq_len, seq_len] (M=seq_len, K=seq_len)
  // V:      [num_heads, seq_len, head_size] (K=seq_len, N=head_size)
  // Output: [num_heads, seq_len, head_size] (M=seq_len, N=head_size)

  float* out_reshaped;
  cudaMallocAsync(&out_reshaped, qkv_size, stream);

  // Row-major: Output = Scores @ V
  // Col-major cuBLAS: C^T = (S@V)^T = V^T @ S^T
  // cublas(CUBLAS_OP_N, CUBLAS_OP_N, N=head_size, M=seq_len, K=seq_len,
  //        V_ptr, lda=head_size, Scores_ptr, ldb=seq_len, Out_ptr,
  //        ldc=head_size)
  cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_N,  // op(A) = V
      CUBLAS_OP_N,  // op(B) = Scores
      head_size,    // N
      seq_len,      // M
      seq_len,      // K
      &alpha, v_reshaped, head_size, (int64_t)seq_len * head_size,  // A = V
      scores, seq_len, (int64_t)seq_len * seq_len,  // B = Scores
      &beta, out_reshaped, head_size,
      (int64_t)seq_len * head_size,  // C = Output
      num_heads);

  // ---- Step 5: Reshape output back ----
  // [num_heads, seq_len, head_size] -> [seq_len, num_heads * head_size]
  reshape_output_kernel<<<blocks, threads, 0, stream>>>(
      out_reshaped, const_cast<float*>(output.ptr<float>()), seq_len, num_heads,
      head_size);

  // ---- Cleanup ----
  cudaFreeAsync(q_reshaped, stream);
  cudaFreeAsync(k_reshaped, stream);
  cudaFreeAsync(v_reshaped, stream);
  cudaFreeAsync(scores, stream);
  cudaFreeAsync(out_reshaped, stream);
}

}  // namespace kernel
