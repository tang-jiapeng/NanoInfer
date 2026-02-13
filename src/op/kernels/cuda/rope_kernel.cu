#include "rope_kernel.cuh"

namespace kernel {
// Device 函数：执行旋转计算
__device__ void rope_calc(float cos_val, float sin_val, float* vec,
                          int32_t idx) {
  // 使用 float2 向量化读取，一次读两个 float (x, y)
  float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
  float2 vec_value = *vec_ptr;

  // RoPE 公式:
  // x_new = x * cos - y * sin
  // y_new = x * sin + y * cos
  *vec_ptr = make_float2(vec_value.x * cos_val - vec_value.y * sin_val,
                         vec_value.x * sin_val + vec_value.y * cos_val);
}

/**
 * @brief Batched RoPE Kernel
 * 支持 Continuous Batching，每个 Token 有独立的位置索引
 */
__global__ void rope_kernel_cu_fp32(int32_t total_tokens, int32_t dim,
                                    int32_t kv_dim, int32_t head_size,
                                    const float* input_q, const float* input_k,
                                    const int32_t* input_pos,  // [total_tokens]
                                    const float* sin_cache,
                                    const float* cos_cache) {
  // 全局线程索引
  int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  // 我们按 Pair (一对数值) 处理，所以总 Pair 数是 total_tokens * (dim / 2)
  int32_t half_dim = dim / 2;
  if (idx >= total_tokens * half_dim) {
    return;
  }

  // 1. 确定当前处理的是哪个 Token 以及该 Token 内的哪个维度对
  int32_t token_idx = idx / half_dim;
  int32_t dim_pair_idx = idx % half_dim;
  int32_t real_dim_idx = dim_pair_idx * 2;  // 实际 float 索引 (0, 2, 4...)

  // 2. 获取当前 Token 的位置索引 (关键改动: 从数组读取)
  int32_t pos = input_pos[token_idx];

  // 3. 获取 Sin/Cos
  // RoPE 是在 Head 内部旋转，所以需要对 head_size 取模找到对应频率
  int32_t head_dim = real_dim_idx % head_size;
  // Cache Shape: [max_seq_len, head_size] (简化版，有些实现是 [max_seq_len,
  // head_dim/2]) 假设 Cache 里存的是预计算好的值，索引方式需与 Cache
  // 生成逻辑一致 这里假设 sin_cache 也是按 float 存的，且已经 duplicate
  // 过了或者就是按 pair 存的 通常 RoPE Cache 生成时：cache[pos, i] 对应 head
  // 中第 i 个分量的频率值
  float sin_val = sin_cache[pos * head_size + head_dim];
  float cos_val = cos_cache[pos * head_size + head_dim];

  // 4. 对 Query 进行旋转
  // Q Shape: [total_tokens, dim]
  // 指针偏移 = token_idx * dim + real_dim_idx
  int64_t q_offset = static_cast<int64_t>(token_idx) * dim + real_dim_idx;
  rope_calc(cos_val, sin_val, const_cast<float*>(input_q), q_offset);

  // 5. 对 Key 进行旋转 (如果当前维度在 KV Dim 范围内)
  // 注意：GQA/MQA 场景下 kv_dim 可能小于 dim
  if (real_dim_idx < kv_dim) {
    // K Shape: [total_tokens, kv_dim]
    // 指针偏移 = token_idx * kv_dim + real_dim_idx
    int64_t k_offset = static_cast<int64_t>(token_idx) * kv_dim + real_dim_idx;
    rope_calc(cos_val, sin_val, const_cast<float*>(input_k), k_offset);
  }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                    const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k,
                    const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache,
                    const tensor::Tensor& cos_cache, void* stream) {
  CHECK(!input_q.is_empty());
  CHECK(!input_k.is_empty());
  CHECK(!input_pos.is_empty());
  CHECK(input_q.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(input_pos.device_type() == base::DeviceType::kDeviceCUDA);

  // 计算总 Token 数
  // input_q: [total_tokens, dim]
  // input_pos: [total_tokens]
  int32_t total_tokens = static_cast<int32_t>(input_pos.size());
  CHECK_EQ(input_q.size(), total_tokens * dim);

  // 配置 Grid
  // 总共需要处理 total_tokens * (dim / 2) 个 Pair
  int32_t total_pairs = total_tokens * (dim / 2);
  int32_t threads = 128;
  int32_t blocks = (total_pairs + threads - 1) / threads;

  cudaStream_t stream_ = static_cast<cudaStream_t>(stream);

  rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
      total_tokens, dim, kv_dim, head_size, input_q.ptr<float>(),
      input_k.ptr<float>(), input_pos.ptr<int32_t>(), sin_cache.ptr<float>(),
      cos_cache.ptr<float>());
}
}