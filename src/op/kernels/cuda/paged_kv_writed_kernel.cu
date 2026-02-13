#include "paged_kv_write_kernel.cuh"

namespace kernel {
/**
 * @brief Paged KV Cache 写入 Kernel
 * * 布局假设 (根据 kernels_interface.h):
 * Cache: [num_blocks, block_size, num_kv_heads, head_size]
 * * Grid: (batch_size, num_kv_heads)
 * Block: min(head_size, 512)
 */
__global__ void paged_kv_write_kernel_cu(
    const float* __restrict__ k_src,  // [batch_size, num_kv_heads, head_size]
    const float* __restrict__ v_src,  // [batch_size, num_kv_heads, head_size]
    float* __restrict__ k_cache,      // [num_blocks, block_size, num_kv_heads,
                                      // head_size]
    float* __restrict__ v_cache,      // [num_blocks, block_size, num_kv_heads,
                                      // head_size]
    const int32_t* __restrict__ block_table,  // [batch_size,
                                              // max_blocks_per_seq]
    const int32_t* __restrict__ positions,    // [batch_size]
    int32_t num_kv_heads, int32_t head_size, int32_t block_size,
    int32_t max_blocks_per_seq,
    int32_t stride_block,           // num_kv_heads * head_size * block_size
    int32_t stride_token_in_block,  // num_kv_heads * head_size
    int32_t stride_head             // head_size
) {
  // Grid 维度映射
  const int32_t seq_idx = blockIdx.x;   // 第几个请求
  const int32_t head_idx = blockIdx.y;  // 第几个 KV Head
  const int32_t tid = threadIdx.x;      // 向量维度索引

  // 1. 获取当前 Token 的位置信息
  const int32_t pos = positions[seq_idx];
  const int32_t logical_block_idx = pos / block_size;
  const int32_t block_offset = pos % block_size;

  // 2. 获取物理 Block ID
  // block_table shape: [batch_size, max_blocks_per_seq]
  const int32_t physical_block_idx =
      block_table[seq_idx * max_blocks_per_seq + logical_block_idx];

  // 3. 计算源数据 (Source) 偏移
  // Input Layout: [batch_size, num_kv_heads, head_size]
  // src_offset 指向当前 Head 的向量起始位置
  const int64_t src_offset =
      static_cast<int64_t>(seq_idx) * num_kv_heads * head_size +
      static_cast<int64_t>(head_idx) * head_size;

  // 4. 计算目标数据 (Cache) 偏移
  // Cache Layout: [num_blocks, block_size, num_kv_heads, head_size]
  const int64_t cache_base_offset =
      static_cast<int64_t>(physical_block_idx) * stride_block +
      static_cast<int64_t>(block_offset) * stride_token_in_block +
      static_cast<int64_t>(head_idx) * stride_head;

  // 5. 并行拷贝数据
  // 一个 Block 处理一个 Head 的向量拷贝
  for (int i = tid; i < head_size; i += blockDim.x) {
    k_cache[cache_base_offset + i] = k_src[src_offset + i];
    v_cache[cache_base_offset + i] = v_src[src_offset + i];
  }
}

void paged_kv_write_kernel(const tensor::Tensor& k, const tensor::Tensor& v,
                           const tensor::Tensor& k_cache,
                           const tensor::Tensor& v_cache,
                           const tensor::Tensor& block_table,
                           const tensor::Tensor& input_pos,
                           int32_t num_kv_heads, int32_t head_size,
                           int32_t block_size, void* stream) {
  // 维度检查
  CHECK(k.device_type() == base::DeviceType::kDeviceCUDA);
  CHECK(k_cache.device_type() == base::DeviceType::kDeviceCUDA);

  int32_t batch_size = static_cast<int32_t>(k.get_dim(0));
  // max_blocks_per_seq 从 block_table 的维度获取 [batch, max_blocks]
  int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));

  // 计算 Cache 的 Strides (根据 [num_blocks, block_size, num_kv_heads,
  // head_size])
  int32_t stride_head = head_size;
  int32_t stride_token_in_block = num_kv_heads * stride_head;
  int32_t stride_block = block_size * stride_token_in_block;

  // Grid: (Batch, Num_Heads)
  dim3 grid(batch_size, num_kv_heads);
  // Block: 覆盖 Head Size
  int32_t threads = 128;
  if (head_size > 128) threads = 256;
  if (head_size > 256) threads = 512;

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  paged_kv_write_kernel_cu<<<grid, threads, 0, cuda_stream>>>(
      k.ptr<float>(), v.ptr<float>(), const_cast<float*>(k_cache.ptr<float>()),
      const_cast<float*>(v_cache.ptr<float>()), block_table.ptr<int32_t>(),
      input_pos.ptr<int32_t>(), num_kv_heads, head_size, block_size,
      max_blocks_per_seq, stride_block, stride_token_in_block, stride_head);
}
}  // namespace kernel