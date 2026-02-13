#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cfloat>
#include <cub/block/block_reduce.cuh>

#include "paged_attention_kernel.cuh"
namespace kernel {

template <int BLOCK_SIZE>
__device__ void softmax_inplace(float* __restrict__ logits, int size,
                                float scale) {
  const int tid = threadIdx.x;

  // 1. Find Max (利用 scale 预处理，防止溢出)
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

  // 2. Exp & Sum
  float local_sum = 0.0f;
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    float val = __expf(logits[i] - global_max);
    logits[i] = val;
    local_sum += val;
  }

  __syncthreads();  // Reuse temp_storage
  float global_sum = BlockReduce(temp_storage).Sum(local_sum);

  __shared__ float shared_sum;
  if (tid == 0) shared_sum = global_sum;
  __syncthreads();
  global_sum = shared_sum;

  // 3. Normalize
  float inv_sum = 1.0f / (global_sum + 1e-6f);
  for (int i = tid; i < size; i += BLOCK_SIZE) {
    logits[i] *= inv_sum;
  }
}

// =========================================================================================
// Kernel: Paged Attention V1
// 模板参数：
// BLOCK_SIZE: CUDA 线程块大小 (128)
// HEAD_SIZE: Head 维度 (64, 80, 96, 128, 256)
// KV_BLOCK_SIZE: PagedBlock 大小 (16, 32)
// =========================================================================================
template <int BLOCK_SIZE, int HEAD_SIZE, int KV_BLOCK_SIZE>
__global__ void paged_attention_kernel_v1(
    float* __restrict__ output, const float* __restrict__ query,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ context_lens, int32_t num_kv_heads,
    int32_t max_blocks_per_seq, float scale, int32_t stride_block,
    int32_t stride_token, int32_t stride_head) {
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;

  // 1. 获取当前序列长度
  const int seq_len = context_lens[seq_idx];
  if (seq_len == 0) return;

  // 2. GQA 映射: 当前 Query Head 对应哪个 KV Head
  const int num_heads = gridDim.x;
  const int heads_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / heads_per_kv;

  // 3. 加载 Query 到 Shared Memory
  // HEAD_SIZE 是编译期常量，此处申请静态 Shared Memory 可能会超，所以我们只对
  // Query 用静态 更好的方式是 Q 用寄存器(如果 HEAD_SIZE 小) 或
  // Shared。这里为了通用性用 Shared。
  __shared__ float smem_q[HEAD_SIZE];

  const int q_offset = seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

  // 向量化加载 float4
  // 假设 HEAD_SIZE 是 4 的倍数 (Llama/Qwen 都是)
  const float4* q_ptr_4 = reinterpret_cast<const float4*>(query + q_offset);
  float4* smem_q_4 = reinterpret_cast<float4*>(smem_q);

  for (int i = tid; i < HEAD_SIZE / 4; i += BLOCK_SIZE) {
    smem_q_4[i] = q_ptr_4[i];
  }
  // 处理不能被 4 整除的尾部 (虽然 HEAD_SIZE 通常是 32 的倍数)
  // 这里为了性能，假设 HEAD_SIZE % 4 == 0
  __syncthreads();

  // 4. 计算 Attention Scores (Q * K^T)
  // Logits 存储在动态 Shared Memory 中
  extern __shared__ float smem_logits[];

  // 遍历序列中的所有 Token
  for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
    // 逻辑坐标 -> 物理坐标
    const int log_block_idx = t / KV_BLOCK_SIZE;
    const int block_offset = t % KV_BLOCK_SIZE;
    const int phys_block_idx =
        block_table[seq_idx * max_blocks_per_seq + log_block_idx];

    // 计算物理地址
    int64_t base_offset = static_cast<int64_t>(phys_block_idx) * stride_block +
                          static_cast<int64_t>(block_offset) * stride_token +
                          static_cast<int64_t>(kv_head_idx) * stride_head;

    const float* k_ptr = k_cache + base_offset;

    // 点积计算
    float score = 0.0f;

// 循环展开优化
#pragma unroll
    for (int i = 0; i < HEAD_SIZE; ++i) {
      score += smem_q[i] * k_ptr[i];
    }
    smem_logits[t] = score;
  }
  __syncthreads();

  // 5. Softmax
  softmax_inplace<BLOCK_SIZE>(smem_logits, seq_len, scale);
  __syncthreads();

  // 6. 聚合 Value (O = Score * V)
  float acc_val = 0.0f;

  // 外层循环：Head 维度并行 (每个线程负责计算 output 的一部分)
  for (int i = tid; i < HEAD_SIZE; i += BLOCK_SIZE) {
    acc_val = 0.0f;
    // 内层循环：对序列长度求和
    for (int t = 0; t < seq_len; ++t) {
      float prob = smem_logits[t];

      const int log_block_idx = t / KV_BLOCK_SIZE;
      const int block_offset = t % KV_BLOCK_SIZE;
      const int phys_block_idx =
          block_table[seq_idx * max_blocks_per_seq + log_block_idx];

      int64_t base_offset =
          static_cast<int64_t>(phys_block_idx) * stride_block +
          static_cast<int64_t>(block_offset) * stride_token +
          static_cast<int64_t>(kv_head_idx) * stride_head;

      const float* v_ptr = v_cache + base_offset;

      acc_val += prob * v_ptr[i];
    }

    // 写入 Global Memory
    output[q_offset + i] = acc_val;
  }
}

// =========================================================================================
// Kernel Dispatcher (核心修改)
// =========================================================================================

// 定义宏来简化 switch-case
#define LAUNCH_ATTENTION_V1(HEAD_SIZE_VAL, BLOCK_SIZE_VAL)                     \
  paged_attention_kernel_v1<128, HEAD_SIZE_VAL, BLOCK_SIZE_VAL>                \
      <<<grid, 128, smem_size, cuda_stream>>>(                                 \
          const_cast<float*>(output.ptr<float>()), query.ptr<float>(),         \
          k_cache.ptr<float>(), v_cache.ptr<float>(),                          \
          block_table.ptr<int32_t>(), context_lens.ptr<int32_t>(),             \
          num_kv_heads, max_blocks_per_seq, scale, stride_block, stride_token, \
          stride_head)

void paged_attention_kernel(
    const tensor::Tensor& query, const tensor::Tensor& output,
    const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
    const tensor::Tensor& block_table, const tensor::Tensor& context_lens,
    int32_t max_context_len, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_size, int32_t block_size, float scale, void* stream) {
  int32_t batch_size = static_cast<int32_t>(query.get_dim(0));
  int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));

  int32_t stride_head = head_size;
  int32_t stride_token = num_kv_heads * stride_head;
  int32_t stride_block = block_size * stride_token;

  dim3 grid(num_heads, batch_size);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  // 计算 Shared Memory 大小
  // 需要存储 logits [max_context_len]
  // 注意：如果 Context 很长 (如 > 4096)，Shared Mem 可能会不够。
  // V1 版本受限于 Shared Mem 大小。这也是为什么会有 V2 (Block-based
  // reduction)。 这里我们先做个检查。一般 GPU Shared Mem 有 48KB - 164KB。
  // max_context_len * 4 bytes. 4096 * 4 = 16KB (安全)。
  // 8192 * 4 = 32KB (安全)。
  size_t smem_size = max_context_len * sizeof(float);

  // 双重 Switch Dispatch
  // 外层 Block Size
  switch (block_size) {
    case 16:
      // 内层 Head Size
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
          break;  // Llama-2/3, Qwen Standard
        case 256:
          LAUNCH_ATTENTION_V1(256, 16);
          break;
        default:
          LOG(FATAL) << "Unsupported Head Size: " << head_size
                     << " for Block Size 16";
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
          break;  // Llama-2/3, Qwen Large Block
        case 256:
          LAUNCH_ATTENTION_V1(256, 32);
          break;
        default:
          LOG(FATAL) << "Unsupported Head Size: " << head_size
                     << " for Block Size 32";
      }
      break;

    default:
      LOG(FATAL) << "Unsupported Block Size: " << block_size
                 << ". Only 16 and 32 are supported.";
  }
}
}