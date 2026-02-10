#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "nanoinfer/engine/scheduler.h"

using namespace engine;

class SchedulerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 默认配置:
        // Max Batch Size = 2 (一次最多跑 2 个请求)
        // Max Sequences = 4  (系统最多容纳 4 个并发，含 Running + Waiting)
        max_batch_size_ = 2;
        max_sequences_ = 4;
        scheduler_ = std::make_unique<Scheduler>(max_batch_size_, max_sequences_);
    }

    int32_t max_batch_size_;
    int32_t max_sequences_;
    std::unique_ptr<Scheduler> scheduler_;
};

// 1. 测试最基础的调度逻辑
TEST_F(SchedulerTest, BasicScheduling) {
    // 添加 2 个请求
    int64_t id1 = scheduler_->add_request("Prompt A", {1, 2}, 10);
    int64_t id2 = scheduler_->add_request("Prompt B", {3, 4}, 10);

    // 初始状态: 都在 Waiting
    EXPECT_EQ(scheduler_->num_waiting(), 2);
    EXPECT_EQ(scheduler_->num_running(), 0);
    EXPECT_TRUE(scheduler_->has_work());

    // 执行调度
    auto batch = scheduler_->schedule_next_batch();

    // 验证结果: 两个都应该被调度 (Capacity=2)
    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(scheduler_->num_waiting(), 0);
    EXPECT_EQ(scheduler_->num_running(), 2);

    // 验证状态变迁
    EXPECT_EQ(batch.requests[0]->request_id(), id1);
    EXPECT_EQ(batch.requests[0]->state(), RequestState::kRunning);
    EXPECT_EQ(batch.requests[1]->request_id(), id2);
}

// 2. 测试 Batch Size 限制
TEST_F(SchedulerTest, BatchSizeLimit) {
    // Max Batch = 2, 添加 3 个请求
    int64_t id1 = scheduler_->add_request("A", {1}, 10);
    int64_t id2 = scheduler_->add_request("B", {1}, 10);
    int64_t id3 = scheduler_->add_request("C", {1}, 10);

    auto batch = scheduler_->schedule_next_batch();

    // 预期: 只调度前 2 个 (id1, id2)
    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(scheduler_->num_running(), 2);
    EXPECT_EQ(scheduler_->num_waiting(), 1);  // id3 还在等

    EXPECT_EQ(batch.requests[0]->request_id(), id1);
    EXPECT_EQ(batch.requests[1]->request_id(), id2);

    // 再次调度，Running 的继续跑，Batch 满了，id3 依然进不来
    auto batch2 = scheduler_->schedule_next_batch();
    EXPECT_EQ(batch2.size(), 2);
    EXPECT_EQ(batch2.requests[0]->request_id(), id1);
    EXPECT_EQ(batch2.requests[1]->request_id(), id2);
    EXPECT_EQ(scheduler_->num_waiting(), 1);
}

// 3. 测试连续批处理逻辑 (Running 优先 + 填补空缺)
TEST_F(SchedulerTest, ContinuousScheduling) {
    // 场景:
    // T0: [req1, req2] 正在运行
    // T1: req1 完成 -> 释放 1 个槽位
    // T2: 调度 -> 应该包含 [req2 (继续), req3 (新加入)]

    int64_t id1 = scheduler_->add_request("A", {1}, 10);
    int64_t id2 = scheduler_->add_request("B", {1}, 10);
    int64_t id3 = scheduler_->add_request("C", {1}, 10);

    // Step 1: 调度前两个
    auto batch1 = scheduler_->schedule_next_batch();
    EXPECT_EQ(batch1.size(), 2);
    // 验证包含 id1, id2
    bool has_id1 = false, has_id2 = false;
    for (auto& req : batch1.requests) {
        if (req->request_id() == id1) has_id1 = true;
        if (req->request_id() == id2) has_id2 = true;
    }
    EXPECT_TRUE(has_id1 && has_id2);

    // Step 2: 模拟 id1 执行完成
    scheduler_->update_after_step({id1});

    // 验证 id1 状态
    auto req1 = scheduler_->get_request(id1);
    EXPECT_TRUE(req1->is_finished());
    EXPECT_EQ(scheduler_->num_running(), 1);  // 只剩 id2

    // Step 3: 下一轮调度
    auto batch2 = scheduler_->schedule_next_batch();

    // 预期: Batch 大小为 2 (id2 + id3)
    EXPECT_EQ(batch2.size(), 2);

    // 验证 id2 必须在 (Running Priority)
    EXPECT_EQ(batch2.requests[0]->request_id(), id2);
    // 验证 id3 补位成功
    EXPECT_EQ(batch2.requests[1]->request_id(), id3);
}

// 4. 测试最大并发序列数限制 (Max Sequences)
TEST_F(SchedulerTest, MaxSequenceLimit) {
    // 重置 Scheduler，设置极小的 Max Sequences = 2，但 Max Batch 很大 = 10
    scheduler_ = std::make_unique<Scheduler>(10, 2);

    int64_t id1 = scheduler_->add_request("A", {1}, 10);
    int64_t id2 = scheduler_->add_request("B", {1}, 10);
    int64_t id3 = scheduler_->add_request("C", {1}, 10);

    // 虽然 Batch Size 允许跑 10 个，但 Max Seqs 限制了只能跑 2 个
    auto batch = scheduler_->schedule_next_batch();

    EXPECT_EQ(batch.size(), 2);
    EXPECT_EQ(scheduler_->num_running(), 2);
    EXPECT_EQ(scheduler_->num_waiting(), 1);
}

// 5. 测试完成清理逻辑
TEST_F(SchedulerTest, ClearFinished) {
    int64_t id1 = scheduler_->add_request("A", {1}, 10);
    scheduler_->schedule_next_batch();

    // 标记完成
    scheduler_->update_after_step({id1});

    // update_after_step 只是将其从 Running 列表移除，但对象还在 map 中
    EXPECT_NE(scheduler_->get_request(id1), nullptr);
    EXPECT_EQ(scheduler_->get_stats().num_finished, 1);

    // 显式清理
    scheduler_->clear_finished_requests();

    // 应该查不到了
    EXPECT_EQ(scheduler_->get_request(id1), nullptr);
    EXPECT_EQ(scheduler_->get_stats().num_finished, 0);
}
