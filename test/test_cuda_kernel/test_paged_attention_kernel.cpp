#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"


class PagedAttentionTest : public ::testing::Test {
   protected:
    void SetUp() override {
        allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

TEST_F(PagedAttentionTest, LlamaStyleParams) {
    // === 模拟 Llama-2-7B 的配置 ===
    // Head Size = 128
    // Block Size = 16
    // GQA: Query Heads = 4, KV Heads = 2 (倍数=2)
    int32_t batch_size = 1;
    int32_t num_heads = 4;
    int32_t num_kv_heads = 2;
    int32_t head_size = 128;  // <--- 关键：测试 128 维度
    int32_t block_size = 16;
    int32_t max_blocks = 4;
    int32_t context_len = 10;  // 测试一个稍长的序列
    float scale = 1.0f / sqrt(head_size);

    // 1. 准备输入
    // Query: 全 1.0
    std::vector<float> h_q(batch_size * num_heads * head_size, 1.0f);

    // Cache: 填充满 0.1
    // 这样 Q * K = 128 * 1.0 * 0.1 = 12.8
    // Logits = 12.8 * scale = 12.8 / 11.31 = 1.13
    // Softmax 均匀分布 (因为 K 是一样的) -> 概率 = 1 / 10 = 0.1
    // V = 0.2
    // Output = Sum(0.1 * 0.2) * 10 = 0.2
    int32_t cache_size = max_blocks * block_size * num_kv_heads * head_size;
    std::vector<float> h_k_cache(cache_size, 0.1f);
    std::vector<float> h_v_cache(cache_size, 0.2f);  // Value = 0.2

    // Metadata
    // 假设 10 个 Token 都在 Block 0 中 (因为 Block Size 16 > 10)
    std::vector<int32_t> h_block_table = {0, -1, -1, -1};
    std::vector<int32_t> h_lens = {context_len};

    // 2. Tensors
    tensor::Tensor t_q(base::DataType::kDataTypeFp32, batch_size, num_heads, head_size, true,
                       allocator_);
    tensor::Tensor t_out(base::DataType::kDataTypeFp32, batch_size, num_heads, head_size, true,
                         allocator_);
    tensor::Tensor t_k_c(base::DataType::kDataTypeFp32, max_blocks, block_size, num_kv_heads,
                         head_size, true, allocator_);
    tensor::Tensor t_v_c(base::DataType::kDataTypeFp32, max_blocks, block_size, num_kv_heads,
                         head_size, true, allocator_);
    tensor::Tensor t_tbl(base::DataType::kDataTypeInt32, batch_size, max_blocks, true, allocator_);
    tensor::Tensor t_lens(base::DataType::kDataTypeInt32, batch_size, true, allocator_);

    cudaMemcpy(t_q.ptr<void>(), h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_k_c.ptr<void>(), h_k_cache.data(), h_k_cache.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_v_c.ptr<void>(), h_v_cache.data(), h_v_cache.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_tbl.ptr<void>(), h_block_table.data(), h_block_table.size() * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(t_lens.ptr<void>(), h_lens.data(), h_lens.size() * 4, cudaMemcpyHostToDevice);

    // 3. Run
    auto pa_cu = kernel::KernelRegistry::instance().get<kernel::PagedAttentionKernelFn>(
        "paged_attention", base::DeviceType::kDeviceCUDA);
    pa_cu(t_q, t_out, t_k_c, t_v_c, t_tbl, t_lens, context_len, num_heads, num_kv_heads, head_size,
          block_size, scale, nullptr);
    cudaDeviceSynchronize();

    // 4. Verify
    std::vector<float> res(h_q.size());
    cudaMemcpy(res.data(), t_out.ptr<void>(), res.size() * 4, cudaMemcpyDeviceToHost);

    // Expect 0.2
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_NEAR(res[i], 0.2f, 1e-3) << "Mismatch at index " << i;
    }
}