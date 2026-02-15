#include "paged_kv_write_kernel.h"
#include <cstring>

namespace kernel {

void paged_kv_write_kernel_cpu(const tensor::Tensor& k, const tensor::Tensor& v,
                               const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                               const tensor::Tensor& block_table, const tensor::Tensor& input_pos,
                               int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                               void* stream) {
    UNUSED(stream);

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

}  // namespace kernel
