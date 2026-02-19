/**
 * @file paged_kv_writed_kernel.cu
 * @brief Paged KV Cache Write CUDA Kernel（Decode 阶段专用）
 *
 * 功能：将当前步新生成的 K/V 向量写入 Paged KV Cache 的正确物理位置。
 *
 * 地址计算流程（每个 token 独立）：
 *   1. 从 positions[seq_idx] 获取当前 token 的绝对位置 pos
 *   2. 逻辑块号 = pos / block_size
 *   3. 块内偏移 = pos % block_size
 *   4. 物理块号 = block_table[seq_idx][逻辑块号]
 *   5. Cache 物理地址 = 物理块号 * stride_block + 块内偏移 * stride_token +
 * head * stride_head
 *
 * 数据布局：
 *   输入 K/V :  [batch_size, num_kv_heads, head_size]       (连续)
 *   Cache    :  [num_blocks, block_size, num_kv_heads, head_size]  (分页)
 *
 * 线程映射：
 *   Grid  = (batch_size, num_kv_heads)   → 每个 Block 负责一个 (seq, head) 对
 *   Block = min(head_size, 512) threads  → 并行拷贝 head_size 个 float
 */
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief KV Cache 写入 Kernel
 *
 * 每个 CUDA Block 将一个 (seq, kv_head) 对的 K/V 向量从输入拷贝到 Cache。
 * 核心逻辑是 **逻辑位置 → 物理地址** 的转换。
 */
__global__ void paged_kv_write_kernel(
    const float* __restrict__ k_src,          // [batch_size, num_kv_heads, head_size]
    const float* __restrict__ v_src,          // [batch_size, num_kv_heads, head_size]
    float* __restrict__ k_cache,              // [num_blocks, block_size, num_kv_heads,
                                              // head_size]
    float* __restrict__ v_cache,              // [num_blocks, block_size, num_kv_heads,
                                              // head_size]
    const int32_t* __restrict__ block_table,  // [batch_size,
                                              // max_blocks_per_seq]
    const int32_t* __restrict__ positions,    // [batch_size]
    int32_t num_kv_heads, int32_t head_size, int32_t block_size, int32_t max_blocks_per_seq,
    int32_t stride_block,           // = block_size * num_kv_heads * head_size
    int32_t stride_token_in_block,  // = num_kv_heads * head_size
    int32_t stride_head             // = head_size
) {
    const int32_t seq_idx = blockIdx.x;
    const int32_t head_idx = blockIdx.y;
    const int32_t tid = threadIdx.x;

    // ---- 逻辑位置 → 物理地址 ----
    const int32_t pos = positions[seq_idx];
    const int32_t logical_block_idx = pos / block_size;
    const int32_t block_offset = pos % block_size;
    const int32_t physical_block_idx =
        block_table[seq_idx * max_blocks_per_seq + logical_block_idx];

    // 输入偏移: [batch_size, num_kv_heads, head_size] 中定位到 (seq, head)
    const int64_t src_offset = static_cast<int64_t>(seq_idx) * num_kv_heads * head_size +
                               static_cast<int64_t>(head_idx) * head_size;

    // Cache 偏移: [num_blocks, block_size, num_kv_heads, head_size] 中定位到
    // (block, slot, head)
    const int64_t cache_base_offset = static_cast<int64_t>(physical_block_idx) * stride_block +
                                      static_cast<int64_t>(block_offset) * stride_token_in_block +
                                      static_cast<int64_t>(head_idx) * stride_head;

    // 并行拷贝 head_size 个 float
    for (int i = tid; i < head_size; i += blockDim.x) {
        k_cache[cache_base_offset + i] = k_src[src_offset + i];
        v_cache[cache_base_offset + i] = v_src[src_offset + i];
    }
}

/// @brief Host 入口：配置 Grid/Block 并启动 Kernel
void paged_kv_write_kernel_cu(const tensor::Tensor& k, const tensor::Tensor& v,
                              const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                              const tensor::Tensor& block_table, const tensor::Tensor& input_pos,
                              int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                              void* stream) {
    CHECK(k.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(k_cache.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t batch_size = static_cast<int32_t>(k.get_dim(0));
    int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));

    // Cache strides（根据 [num_blocks, block_size, num_kv_heads, head_size]）
    int32_t stride_head = head_size;
    int32_t stride_token_in_block = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token_in_block;

    dim3 grid(batch_size, num_kv_heads);
    int32_t threads = 128;
    if (head_size > 128) threads = 256;
    if (head_size > 256) threads = 512;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    paged_kv_write_kernel<<<grid, threads, 0, cuda_stream>>>(
        k.ptr<float>(), v.ptr<float>(), const_cast<float*>(k_cache.ptr<float>()),
        const_cast<float*>(v_cache.ptr<float>()), block_table.ptr<int32_t>(),
        input_pos.ptr<int32_t>(), num_kv_heads, head_size, block_size, max_blocks_per_seq,
        stride_block, stride_token_in_block, stride_head);
}

REGISTER_KERNEL(paged_kv_write, kDeviceCUDA, paged_kv_write_kernel_cu);

}  // namespace kernel