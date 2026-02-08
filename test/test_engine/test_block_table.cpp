#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/block_table.h"
#include "nanoinfer/tensor/tensor.h"

using namespace engine;

class BlockTableTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 创建一个非线程安全的 BlockTable 用于测试
        block_table_ = std::make_unique<BlockTable>(false);
    }

    std::unique_ptr<BlockTable> block_table_;
};

// 1. 测试基本的序列分配和查询
TEST_F(BlockTableTest, AllocateAndGet) {
    int32_t seq_id = 1;
    std::vector<int32_t> initial_blocks = {10, 20, 30};

    // 分配
    auto status = block_table_->allocate_sequence(seq_id, initial_blocks);
    ASSERT_TRUE(status) << status.get_err_msg();

    // 验证存在性
    EXPECT_TRUE(block_table_->has_sequence(seq_id));
    EXPECT_EQ(block_table_->get_num_sequences(), 1);

    // 验证块数量
    EXPECT_EQ(block_table_->get_num_blocks(seq_id), 3);

    // 验证内容
    std::vector<int32_t> blocks;
    status = block_table_->get_blocks(seq_id, blocks);
    ASSERT_TRUE(status);
    EXPECT_EQ(blocks, initial_blocks);
}

// 2. 测试追加块 (Append)
TEST_F(BlockTableTest, AppendBlocks) {
    int32_t seq_id = 1;
    block_table_->allocate_sequence(seq_id, {1, 2});

    // 追加单个
    auto status = block_table_->append_block(seq_id, 3);
    ASSERT_TRUE(status);
    EXPECT_EQ(block_table_->get_num_blocks(seq_id), 3);

    // 追加多个
    status = block_table_->append_blocks(seq_id, {4, 5});
    ASSERT_TRUE(status);
    EXPECT_EQ(block_table_->get_num_blocks(seq_id), 5);

    // 验证完整序列
    std::vector<int32_t> blocks;
    block_table_->get_blocks(seq_id, blocks);
    std::vector<int32_t> expected = {1, 2, 3, 4, 5};
    EXPECT_EQ(blocks, expected);
}

// 3. 测试释放序列 (Free)
TEST_F(BlockTableTest, FreeSequence) {
    int32_t seq_id = 99;
    std::vector<int32_t> initial_blocks = {100, 101};
    block_table_->allocate_sequence(seq_id, initial_blocks);

    // 释放
    std::vector<int32_t> freed_blocks;
    auto status = block_table_->free_sequence(seq_id, freed_blocks);
    ASSERT_TRUE(status);

    // 验证返回的块是否正确 (用于归还给 BlockManager)
    EXPECT_EQ(freed_blocks, initial_blocks);

    // 验证状态
    EXPECT_FALSE(block_table_->has_sequence(seq_id));
    EXPECT_EQ(block_table_->get_num_sequences(), 0);

    // 再次释放应失败
    status = block_table_->free_sequence(seq_id, freed_blocks);
    EXPECT_FALSE(status);
}

// 4. 测试 GPU Tensor 格式转换 (核心功能)
TEST_F(BlockTableTest, ToGPUFormat) {
    // 构造场景:
    // Seq 1: [1, 2]       (2 blocks)
    // Seq 2: [10, 11, 12] (3 blocks)
    // Seq 3: [20]         (1 block)
    block_table_->allocate_sequence(1, {1, 2});
    block_table_->allocate_sequence(2, {10, 11, 12});
    block_table_->allocate_sequence(3, {20});

    // 我们只把 Seq 1 和 Seq 3 放入当前 Batch (跳过 Seq 2 测试筛选功能)
    std::vector<int32_t> batch_seq_ids = {1, 3};
    int32_t max_blocks = 4;  // Max blocks per seq set to 4

    tensor::Tensor gpu_table;
    auto status = block_table_->to_gpu_format(batch_seq_ids, max_blocks, gpu_table);
    ASSERT_TRUE(status) << status.get_err_msg();

    // 验证 Tensor 属性
    ASSERT_FALSE(gpu_table.is_empty());
    EXPECT_EQ(gpu_table.device_type(), base::DeviceType::kDeviceCPU);  // 初始生成在 CPU
    EXPECT_EQ(gpu_table.data_type(), base::DataType::kDataTypeInt32);

    // 验证 Shape: [num_seqs, max_blocks] -> [2, 4]
    const auto& dims = gpu_table.dims();
    ASSERT_EQ(dims.size(), 2);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 4);

    // 验证数据内容 (Padding 应为 -1)
    // Row 0 (Seq 1): [1, 2, -1, -1]
    // Row 1 (Seq 3): [20, -1, -1, -1]
    int32_t* ptr = gpu_table.ptr<int32_t>();
    ASSERT_NE(ptr, nullptr);

    // Row 0
    EXPECT_EQ(ptr[0], 1);
    EXPECT_EQ(ptr[1], 2);
    EXPECT_EQ(ptr[2], -1);
    EXPECT_EQ(ptr[3], -1);

    // Row 1 (Offset = 4)
    EXPECT_EQ(ptr[4], 20);
    EXPECT_EQ(ptr[5], -1);
    EXPECT_EQ(ptr[6], -1);
    EXPECT_EQ(ptr[7], -1);
}

// 5. 测试异常处理
TEST_F(BlockTableTest, ErrorHandling) {
    // 1. 重复创建
    block_table_->allocate_sequence(1, {10});
    auto status = block_table_->allocate_sequence(1, {20});
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);

    // 2. 操作不存在的序列
    status = block_table_->append_block(999, 1);
    EXPECT_FALSE(status);

    // 3. to_gpu_format 越界检查 (Sequence 长度超过 max_blocks)
    block_table_->allocate_sequence(2, {1, 2, 3});
    tensor::Tensor t;
    status = block_table_->to_gpu_format({2}, 2, t);  // max=2, but len=3
    EXPECT_FALSE(status);
    LOG(INFO) << "Expected error: " << status.get_err_msg();
}