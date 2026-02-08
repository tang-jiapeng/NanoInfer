#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <vector>
#include "nanoinfer/base/alloc.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/kv_cache_manager.h"

using namespace engine;

class KVCacheManagerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 配置参数
        // 总显存: 10 个 Block
        // Block 大小: 16 tokens
        // 模型: 2 Layers, 2 KV Heads, Head Dim 64
        num_blocks_ = 10;
        block_size_ = 16;
        num_layers_ = 2;
        num_kv_heads_ = 2;
        head_size_ = 64;

        manager_ = std::make_unique<KVCacheManager>(num_blocks_, block_size_, num_layers_,
                                                    num_kv_heads_, head_size_);

        // 使用 CPU Allocator 进行测试 (不需要 GPU 环境即可验证逻辑)
        allocator_ = base::CPUDeviceAllocatorFactory::get_instance();

        auto status = manager_->init(allocator_);
        ASSERT_TRUE(status) << "Init failed: " << status.get_err_msg();
    }

    int32_t num_blocks_;
    int32_t block_size_;
    int32_t num_layers_;
    int32_t num_kv_heads_;
    int32_t head_size_;
    std::unique_ptr<KVCacheManager> manager_;
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

// 1. 初始化检查
TEST_F(KVCacheManagerTest, Initialization) {
    EXPECT_EQ(manager_->get_total_block_num(), num_blocks_);
    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_);
    EXPECT_FLOAT_EQ(manager_->get_utilization(), 0.0f);

    // 检查 Tensor 是否分配成功
    tensor::Tensor& k_cache = manager_->get_key_cache(0);
    EXPECT_FALSE(k_cache.is_empty());
    EXPECT_NE(k_cache.ptr<float>(), nullptr);
    // Shape: [num_blocks, num_kv_heads, block_size, head_size]
    EXPECT_EQ(k_cache.get_dim(0), num_blocks_);
    EXPECT_EQ(k_cache.get_dim(1), num_kv_heads_);
    EXPECT_EQ(k_cache.get_dim(2), block_size_);
    EXPECT_EQ(k_cache.get_dim(3), head_size_);
}

// 2. 基础分配测试
TEST_F(KVCacheManagerTest, BasicAllocation) {
    int32_t seq_id = 1;
    // 申请 30 个 token -> 需要 ceil(30/16) = 2 个 blocks
    int32_t num_tokens = 30;

    auto status = manager_->allocate_sequence(seq_id, num_tokens);
    ASSERT_TRUE(status);

    EXPECT_TRUE(manager_->is_sequence_allocated(seq_id));
    EXPECT_EQ(manager_->get_num_blocks(seq_id), 2);
    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_ - 2);

    // 验证容量
    EXPECT_EQ(manager_->get_sequence_capacity(seq_id), 2 * block_size_);
}

// 3. 动态扩展 - 不需要新块
TEST_F(KVCacheManagerTest, ExtendSequenceInternal) {
    int32_t seq_id = 1;
    // 初始: 20 tokens -> 2 blocks (Capacity 32)
    manager_->allocate_sequence(seq_id, 20);
    int32_t initial_blocks = manager_->get_num_blocks(seq_id);
    EXPECT_EQ(initial_blocks, 2);

    // 扩展: +10 tokens -> total 30. 仍然在 Capacity 32 范围内
    int32_t new_blocks_alloc = 0;
    auto status = manager_->extend_sequence(seq_id, 10, &new_blocks_alloc);

    ASSERT_TRUE(status);
    EXPECT_EQ(new_blocks_alloc, 0);  // 不需要分配新块
    EXPECT_EQ(manager_->get_num_blocks(seq_id), 2);
}

// 4. 动态扩展 - 需要分配新块
TEST_F(KVCacheManagerTest, ExtendSequenceExternal) {
    int32_t seq_id = 1;
    // 初始: 16 tokens -> 1 block (Capacity 16)
    manager_->allocate_sequence(seq_id, 16);
    EXPECT_EQ(manager_->get_num_blocks(seq_id), 1);

    // 扩展: +1 token -> total 17. 需要第 2 个 block
    int32_t new_blocks_alloc = 0;
    auto status = manager_->extend_sequence(seq_id, 1, &new_blocks_alloc);

    ASSERT_TRUE(status);
    EXPECT_EQ(new_blocks_alloc, 1);
    EXPECT_EQ(manager_->get_num_blocks(seq_id), 2);
    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_ - 2);
}

// 5. 释放序列
TEST_F(KVCacheManagerTest, FreeSequence) {
    manager_->allocate_sequence(1, 32);  // 2 blocks
    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_ - 2);

    auto status = manager_->free_sequence(1);
    ASSERT_TRUE(status);

    EXPECT_FALSE(manager_->is_sequence_allocated(1));
    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_);  // 内存归还
}

// 6. 显存耗尽 (OOM) 测试
TEST_F(KVCacheManagerTest, OutOfMemory) {
    // 总共 10 个块
    // 申请 9 个块 (9 * 16 = 144 tokens)
    manager_->allocate_sequence(1, 144);
    EXPECT_EQ(manager_->get_free_block_num(), 1);

    // 尝试申请 2 个块 -> 失败
    auto status = manager_->allocate_sequence(2, 32);
    EXPECT_FALSE(status);
    // 这里具体错误码取决于 BlockManager 实现，通常是 InternalError 或 OOM

    // 尝试扩展现有序列超出限制 -> 失败
    // Seq 1 当前 9 块，剩 1 块。尝试扩展 2 块 (+32 tokens)
    int32_t new_alloc = 0;
    status = manager_->extend_sequence(1, 32, &new_alloc);
    EXPECT_FALSE(status);
}

// 7. GPU Block Table 生成测试
TEST_F(KVCacheManagerTest, BlockTableTensorGeneration) {
    // Seq 1: 3 blocks
    manager_->allocate_sequence(1, 48);
    // Seq 2: 1 block
    manager_->allocate_sequence(2, 5);

    std::vector<int32_t> batch_ids = {1, 2};
    tensor::Tensor table_tensor;
    auto status = manager_->get_block_table_tensor(batch_ids, table_tensor);

    ASSERT_TRUE(status);
    EXPECT_FALSE(table_tensor.is_empty());

    // 验证维度 [batch_size, max_blocks_per_seq]
    EXPECT_EQ(table_tensor.dims().size(), 2);
    EXPECT_EQ(table_tensor.get_dim(0), 2);
    EXPECT_EQ(table_tensor.get_dim(1), manager_->get_max_blocks_per_seq());

    // 验证数据
    int32_t* ptr = table_tensor.ptr<int32_t>();
    ASSERT_NE(ptr, nullptr);

    // 由于 Block 分配 ID 是动态的，我们只能验证其属性
    // Seq 1 应有 3 个有效 ID，其余为 -1
    int32_t max_blocks = manager_->get_max_blocks_per_seq();
    int32_t count_seq1 = 0;
    for (int i = 0; i < max_blocks; ++i) {
        if (ptr[i] != -1) count_seq1++;
    }
    EXPECT_EQ(count_seq1, 3);

    // Seq 2 (偏移 max_blocks) 应有 1 个有效 ID
    int32_t count_seq2 = 0;
    int32_t* ptr_seq2 = ptr + max_blocks;
    for (int i = 0; i < max_blocks; ++i) {
        if (ptr_seq2[i] != -1) count_seq2++;
    }
    EXPECT_EQ(count_seq2, 1);
}

// 8. 异常参数处理
TEST_F(KVCacheManagerTest, InvalidOperations) {
    // 扩展不存在的序列
    int32_t alloc = 0;
    auto status = manager_->extend_sequence(999, 10, &alloc);
    EXPECT_FALSE(status);

    // 释放不存在的序列
    status = manager_->free_sequence(999);
    EXPECT_FALSE(status);

    // 重复分配
    manager_->allocate_sequence(1, 10);
    status = manager_->allocate_sequence(1, 10);
    EXPECT_FALSE(status);
}

// 9. Reset 功能
TEST_F(KVCacheManagerTest, Reset) {
    manager_->allocate_sequence(1, 32);
    manager_->allocate_sequence(2, 16);
    EXPECT_NE(manager_->get_free_block_num(), num_blocks_);

    manager_->reset();

    EXPECT_EQ(manager_->get_free_block_num(), num_blocks_);
    EXPECT_FALSE(manager_->is_sequence_allocated(1));
    EXPECT_FALSE(manager_->is_sequence_allocated(2));
}
