/**
 * @file paged_kv_write_kernel.cpp
 * @brief CPU Paged KV Cache 写入算子
 *
 * 将当前步生成的 Key/Value 写入全局的不连续 KV Cache（Paged 结构）：
 *   1. 根据 Token 位置计算逻辑 Block 及 Block 内偏移
 *   2. 通过 Block Table 查找物理 Block ID
 *   3. memcpy 复制每个 KV Head 的数据到 Cache 目标位置
 */
#include <cstring>
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU Paged KV Cache 写入
 *
 * 将当前步生成的 K/V 写入全局 Paged Cache：
 *   1. 根据 Token 位置算出逻辑 Block 及 Block 内偏移
 *   2. 通过 Block Table 查找物理 Block ID
 *   3. memcpy 复制每个 KV Head 的 head_size 个 float
 *
 * @param k            Key Tensor [batch_size, num_kv_heads × head_size]
 * @param v            Value Tensor，shape 同 k
 * @param k_cache      Key Cache [num_blocks, block_size, num_kv_heads, head_size]
 * @param v_cache      Value Cache，布局同 k_cache
 * @param block_table  Block Table [batch_size, max_blocks_per_seq]，Int32
 * @param input_pos    每个序列当前 Token 的位置索引 [batch_size]，Int32
 * @param num_kv_heads KV Head 数
 * @param head_size    单个 Head 维度
 * @param block_size   每个物理 Block 容纳的 Token 数
 * @param stream       未使用
 */
void paged_kv_write_kernel_cpu(const tensor::Tensor& k, const tensor::Tensor& v,
                               const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                               const tensor::Tensor& block_table, const tensor::Tensor& input_pos,
                               int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                               [[maybe_unused]] void* stream) {
    int32_t batch_size = static_cast<int32_t>(k.get_dim(0));
    int32_t max_blocks_per_seq = static_cast<int32_t>(block_table.get_dim(1));

    // Cache strides: [num_blocks, block_size, num_kv_heads, head_size]
    int32_t stride_head = head_size;
    int32_t stride_token = num_kv_heads * stride_head;
    int32_t stride_block = block_size * stride_token;

    const float* k_src = k.ptr<float>();
    const float* v_src = v.ptr<float>();
    float* k_dst = const_cast<float*>(k_cache.ptr<float>());
    float* v_dst = const_cast<float*>(v_cache.ptr<float>());
    const int32_t* bt = block_table.ptr<int32_t>();
    const int32_t* pos = input_pos.ptr<int32_t>();

    for (int32_t seq = 0; seq < batch_size; ++seq) {
        int32_t position = pos[seq];
        int32_t logical_block = position / block_size;
        int32_t block_offset = position % block_size;
        int32_t physical_block = bt[seq * max_blocks_per_seq + logical_block];

        for (int32_t h = 0; h < num_kv_heads; ++h) {
            int64_t src_offset = static_cast<int64_t>(seq) * num_kv_heads * head_size +
                                 static_cast<int64_t>(h) * head_size;
            int64_t cache_offset = static_cast<int64_t>(physical_block) * stride_block +
                                   static_cast<int64_t>(block_offset) * stride_token +
                                   static_cast<int64_t>(h) * stride_head;

            std::memcpy(k_dst + cache_offset, k_src + src_offset, head_size * sizeof(float));
            std::memcpy(v_dst + cache_offset, v_src + src_offset, head_size * sizeof(float));
        }
    }
}

REGISTER_KERNEL(paged_kv_write, kDeviceCPU, paged_kv_write_kernel_cpu)

}  // namespace kernel
