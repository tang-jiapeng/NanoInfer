#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/tensor/tensor.h"

class PagedKVWriteTest : public ::testing::Test {
   protected:
    void SetUp() override {
        allocator_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

TEST_F(PagedKVWriteTest, BasicWrite) {
    // === 1. 参数设置 ===
    int32_t batch_size = 2;
    int32_t num_kv_heads = 2;
    int32_t head_size = 4;  // 方便肉眼 Debug
    int32_t block_size = 4;
    int32_t max_blocks = 2;     // 每个 Seq 最多 2 个 Block
    int32_t total_blocks = 10;  // 全局 Cache 有 10 个 Block

    // === 2. 准备数据 ===
    // Seq 0: Pos = 5 (逻辑 Block 1, Offset 1) -> 映射到物理 Block 3
    // Seq 1: Pos = 0 (逻辑 Block 0, Offset 0) -> 映射到物理 Block 7
    std::vector<int32_t> h_pos = {5, 0};

    // Block Table [batch, max_blocks]
    // Seq 0: [2, 3] (逻辑0->物理2, 逻辑1->物理3)
    // Seq 1: [7, 8] (逻辑0->物理7, 逻辑1->物理8)
    std::vector<int32_t> h_block_table = {2, 3, 7, 8};

    // K/V Data [batch, heads, dim] = [2, 2, 4]
    // Token 0 (Pos 5): Head0 全是 5.0, Head1 全是 5.1
    // Token 1 (Pos 0): Head0 全是 0.0, Head1 全是 0.1
    std::vector<float> h_k(batch_size * num_kv_heads * head_size);
    // Fill data
    for (int j = 0; j < head_size; ++j) {
        // Seq 0
        h_k[0 * num_kv_heads * head_size + 0 * head_size + j] = 5.0f;
        h_k[0 * num_kv_heads * head_size + 1 * head_size + j] = 5.1f;
        // Seq 1
        h_k[1 * num_kv_heads * head_size + 0 * head_size + j] = 0.0f;
        h_k[1 * num_kv_heads * head_size + 1 * head_size + j] = 0.1f;
    }
    std::vector<float> h_v = h_k;  // V 和 K 数据一样以便验证

    // === 3. 创建 Tensor ===
    tensor::Tensor t_k(base::DataType::kDataTypeFp32, batch_size, num_kv_heads, head_size, true,
                       allocator_);
    tensor::Tensor t_v(base::DataType::kDataTypeFp32, batch_size, num_kv_heads, head_size, true,
                       allocator_);
    tensor::Tensor t_pos(base::DataType::kDataTypeInt32, batch_size, true, allocator_);
    tensor::Tensor t_block_table(base::DataType::kDataTypeInt32, batch_size, max_blocks, true,
                                 allocator_);

    // Cache: [num_blocks, block_size, num_kv_heads, head_size]
    tensor::Tensor t_k_cache(base::DataType::kDataTypeFp32, total_blocks, block_size, num_kv_heads,
                             head_size, true, allocator_);
    tensor::Tensor t_v_cache(base::DataType::kDataTypeFp32, total_blocks, block_size, num_kv_heads,
                             head_size, true, allocator_);

    // Init Cache to -1.0
    cudaMemset(t_k_cache.ptr<void>(), 0xFF, t_k_cache.byte_size());  // fill NaN/garbage roughly

    // Copy Inputs
    cudaMemcpy(t_k.ptr<void>(), h_k.data(), h_k.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_v.ptr<void>(), h_v.data(), h_v.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_pos.ptr<void>(), h_pos.data(), h_pos.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t_block_table.ptr<void>(), h_block_table.data(), h_block_table.size() * 4,
               cudaMemcpyHostToDevice);

    // === 4. Run Kernel ===
    auto kv_write_cu = kernel::KernelRegistry::instance().get<kernel::PagedKVWriteKernelFn>(
        "paged_kv_write", base::DeviceType::kDeviceCUDA);
    kv_write_cu(t_k, t_v, t_k_cache, t_v_cache, t_block_table, t_pos, num_kv_heads, head_size,
                block_size, nullptr);
    cudaDeviceSynchronize();

    // === 5. Verify ===
    std::vector<float> res_cache(t_k_cache.size());
    cudaMemcpy(res_cache.data(), t_k_cache.ptr<void>(), t_k_cache.byte_size(),
               cudaMemcpyDeviceToHost);

    // 验证 Seq 0 (Pos 5) -> 应该写到了 物理 Block 3, Offset 1 (5 % 4)
    // Block 3 Base = 3 * (4 * 2 * 4) = 3 * 32 = 96
    // Offset 1 Base = 1 * (2 * 4) = 8
    // Addr = 96 + 8 = 104
    // Head 0: Addr 104 -> Expect 5.0
    // Head 1: Addr 104 + 4 = 108 -> Expect 5.1

    int seq0_target_base =
        3 * (block_size * num_kv_heads * head_size) + 1 * (num_kv_heads * head_size);

    EXPECT_FLOAT_EQ(res_cache[seq0_target_base + 0], 5.0f);  // Head 0
    EXPECT_FLOAT_EQ(res_cache[seq0_target_base + 4], 5.1f);  // Head 1

    // 验证 Seq 1 (Pos 0) -> 应该写到了 物理 Block 7, Offset 0
    // Block 7 Base = 7 * 32 = 224
    int seq1_target_base =
        7 * (block_size * num_kv_heads * head_size) + 0 * (num_kv_heads * head_size);

    EXPECT_FLOAT_EQ(res_cache[seq1_target_base + 0], 0.0f);  // Head 0
    EXPECT_FLOAT_EQ(res_cache[seq1_target_base + 4], 0.1f);  // Head 1
}