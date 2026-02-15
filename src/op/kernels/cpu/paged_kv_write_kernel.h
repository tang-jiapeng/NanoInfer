#ifndef NANO_INFER_PAGED_KV_WRITE_KERNEL_CPU_H
#define NANO_INFER_PAGED_KV_WRITE_KERNEL_CPU_H
#include "nanoinfer/tensor/tensor.h"

namespace kernel {
void paged_kv_write_kernel_cpu(const tensor::Tensor& k, const tensor::Tensor& v,
                               const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                               const tensor::Tensor& block_table, const tensor::Tensor& input_pos,
                               int32_t num_kv_heads, int32_t head_size, int32_t block_size,
                               void* stream);
}  // namespace kernel
#endif
