#include <glog/logging.h>
#include <gtest/gtest.h>
#include <set>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/block_manager.h"

using namespace engine;

class BlockManagerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 设置默认参数：10个块，每个块16个token
        num_blocks_ = 10;
        block_size_ = 16;
        block_manager_ = std::make_unique<BlockManager>(num_blocks_, block_size_);
    }

    int32_t num_blocks_;
    int32_t block_size_;
    std::unique_ptr<BlockManager> block_manager_;
};

// 测试初始化状态
TEST_F(BlockManagerTest, Initialization) {
    EXPECT_EQ(block_manager_->get_total_block_num(), num_blocks_);
    EXPECT_EQ(block_manager_->get_free_block_num(), num_blocks_);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 0);
    EXPECT_FLOAT_EQ(block_manager_->get_utilization(), 0.0f);
    EXPECT_EQ(block_manager_->get_block_size(), block_size_);
}

// 测试单个块分配
TEST_F(BlockManagerTest, AllocateSingleBlock) {
    int32_t block_id = -1;
    auto status = block_manager_->allocate(block_id);

    EXPECT_TRUE(status);  // base::Status 转换 bool 为 true 表示成功
    EXPECT_GE(block_id, 0);
    EXPECT_LT(block_id, num_blocks_);

    // 验证分配后的状态
    EXPECT_EQ(block_manager_->get_free_block_num(), num_blocks_ - 1);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 1);
    EXPECT_TRUE(block_manager_->is_allocated(block_id));

    // 验证利用率
    EXPECT_FLOAT_EQ(block_manager_->get_utilization(), 1.0f / num_blocks_);
}

// 测试批量分配
TEST_F(BlockManagerTest, AllocateBatch) {
    int32_t alloc_num = 5;
    std::vector<int32_t> blocks;
    auto status = block_manager_->allocate(alloc_num, blocks);

    EXPECT_TRUE(status);
    EXPECT_EQ(blocks.size(), alloc_num);
    EXPECT_EQ(block_manager_->get_free_block_num(), num_blocks_ - alloc_num);

    // 验证所有分配的块都是唯一的
    std::set<int32_t> unique_blocks(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_blocks.size(), alloc_num);

    for (int32_t id : blocks) {
        EXPECT_TRUE(block_manager_->is_allocated(id));
    }
}

// 测试内存耗尽 (OOM) 情况
TEST_F(BlockManagerTest, OutOfMemory) {
    // 1. 先分配所有块
    std::vector<int32_t> blocks;
    auto status = block_manager_->allocate(num_blocks_, blocks);
    EXPECT_TRUE(status);
    EXPECT_EQ(block_manager_->get_free_block_num(), 0);

    // 2. 尝试再分配一个块，应失败
    int32_t block_id = -1;
    status = block_manager_->allocate(block_id);
    EXPECT_FALSE(status);  // 期望失败
    EXPECT_EQ(status.get_err_code(),
              base::StatusCode::kInternalError);  // 假设 OOM 返回 InternalError

    // 3. 尝试批量分配，应失败
    std::vector<int32_t> extra_blocks;
    status = block_manager_->allocate(1, extra_blocks);
    EXPECT_FALSE(status);
}

// 测试释放单个块
TEST_F(BlockManagerTest, FreeSingleBlock) {
    int32_t block_id = -1;
    block_manager_->allocate(block_id);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 1);

    auto status = block_manager_->free(block_id);
    EXPECT_TRUE(status);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 0);
    EXPECT_FALSE(block_manager_->is_allocated(block_id));
}

// 测试释放多个块
TEST_F(BlockManagerTest, FreeBatch) {
    std::vector<int32_t> blocks;
    block_manager_->allocate(5, blocks);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 5);

    auto status = block_manager_->free(blocks);
    EXPECT_TRUE(status);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 0);
}

// 测试重复释放 (Double Free)
TEST_F(BlockManagerTest, DoubleFree) {
    int32_t block_id = -1;
    block_manager_->allocate(block_id);

    // 第一次释放
    auto status = block_manager_->free(block_id);
    EXPECT_TRUE(status);

    // 第二次释放同一个 ID，应报错
    status = block_manager_->free(block_id);
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);
}

// 测试释放无效 ID
TEST_F(BlockManagerTest, FreeInvalidId) {
    auto status = block_manager_->free(9999);  // 超出范围
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);
}

// 测试重置功能
TEST_F(BlockManagerTest, Reset) {
    // 分配一半
    std::vector<int32_t> blocks;
    block_manager_->allocate(num_blocks_ / 2, blocks);
    EXPECT_NE(block_manager_->get_free_block_num(), num_blocks_);

    // 重置
    block_manager_->reset();

    // 验证是否恢复初始状态
    EXPECT_EQ(block_manager_->get_free_block_num(), num_blocks_);
    EXPECT_EQ(block_manager_->get_allocated_block_num(), 0);
    EXPECT_FLOAT_EQ(block_manager_->get_utilization(), 0.0f);

    // 验证之前的 ID 是否被标记为未分配
    for (int32_t id : blocks) {
        EXPECT_FALSE(block_manager_->is_allocated(id));
    }
}

// 测试分配顺序（验证栈式行为：后进先出）
// 注意：这依赖于 BlockManager 的具体实现细节 (Stack)，如果改为 Queue 则此测试需调整
TEST_F(BlockManagerTest, AllocationOrder) {
    // 初始化时是逆序压栈，所以 pop 应该是 0, 1, 2...
    int32_t id1, id2;
    block_manager_->allocate(id1);
    block_manager_->allocate(id2);

    EXPECT_EQ(id1, 0);
    EXPECT_EQ(id2, 1);

    // 释放 id2 (1)，它被压回栈顶
    block_manager_->free(id2);

    // 再次分配，应该拿到栈顶的 id2 (1)
    int32_t id3;
    block_manager_->allocate(id3);
    EXPECT_EQ(id3, 1);
}
