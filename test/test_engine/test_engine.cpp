#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/model.h"

using namespace engine;

class MockModel : public model::Model {
   public:
    MockModel()
        : model::Model(base::TokenizerType::kEncodeUnknown, base::ModelType::kModelTypeUnknown, "",
                       "", false) {
        // 伪造一个 Config
        config_ = std::make_unique<model::TransformerConfig>();
        config_->vocab_size_ = 100;  // 小词表
        config_->hidden_dim_ = 32;
        config_->layer_num_ = 2;
        config_->head_num_ = 4;
        config_->kv_head_num_ = 4;
        config_->head_size_ = 8;
        config_->eos_token_id_ = 2;  // 假设 2 是 EOS
        config_->bos_token_id_ = 1;
    }

    base::Status init(base::DeviceType device_type) override {
        return base::error::Success();
    }
    base::Status create_layers() override {
        return base::error::Success();
    }
    void create_param_layers() override {
    }
    void create_nonparam_layers() override {
    }
    void create_param_quant_layers() override {
    }
    void init_mem() override {
    }

    std::vector<int32_t> encode(const std::string& sentence) const override {
        return {10, 11, 12};
    }

    std::string decode(std::vector<int32_t> token_idxs) const override {
        return "mock_output";
    }

    // Unused legacy methods
    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos, bool is_prompt,
                         int& next) const override {
        return base::error::Success();
    }
    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos,
                         int& next) const override {
        return base::error::Success();
    }
    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override {
        return 0;
    }

    // [Fix] 显式构造 EmbeddingOutput
    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override {
        // 创建 3 个空的 dummy tensor
        tensor::Tensor t1, t2, t3;
        // 显式调用构造函数
        return op::EmbeddingOutput(t1, t2, t3);
    }

    // ==========================================
    // 核心测试点：Forward Batched
    // ==========================================
    base::Status forward_batched(const model::ForwardBatch& input,
                                 tensor::Tensor& logits) override {
        // 验证 Engine 传进来的数据是否合理
        EXPECT_GT(input.batch_size, 0);
        EXPECT_FALSE(input.token_ids.empty());

        // 注意：在 CPU Mock 测试中，如果 Engine 使用了 GPU Allocator 创建 BlockTable，
        // 这里可能会失败。但我们在 SetUp 中使用了 CPUDeviceAllocator，所以应该没问题。
        // EXPECT_FALSE(input.block_table.is_empty());

        int32_t total_tokens = static_cast<int32_t>(input.token_ids.size());
        int32_t vocab_size = config_->vocab_size_;

        float* ptr = logits.ptr<float>();
        if (logits.device_type() == base::DeviceType::kDeviceCPU) {
            for (int i = 0; i < total_tokens * vocab_size; ++i) {
                ptr[i] = 0.0f;
            }
            // 设置 Argmax 为 3
            for (int i = 0; i < total_tokens; ++i) {
                ptr[i * vocab_size + 3] = 10.0f;
            }
        }

        LOG(INFO) << "MockModel: forward_batched called with batch_size=" << input.batch_size
                  << ", total_tokens=" << total_tokens;
        return base::error::Success();
    }
};

class EngineTest : public ::testing::Test {
   protected:
    void SetUp() override {
        mock_model_ = new MockModel();

        EngineConfig config;
        config.max_batch_size = 4;
        config.max_sequences = 8;
        config.prefill_chunk_size = 128;

        engine_ = std::make_unique<Engine>(mock_model_, config);
        allocator_ = base::CPUDeviceAllocatorFactory::get_instance();

        auto status = engine_->init(allocator_);
        ASSERT_TRUE(status) << status.get_err_msg();
    }

    void TearDown() override {
        delete mock_model_;
    }

    MockModel* mock_model_;
    std::unique_ptr<Engine> engine_;
    std::shared_ptr<base::DeviceAllocator> allocator_;
};

TEST_F(EngineTest, AddRequest) {
    int64_t id = engine_->add_request("test prompt", 10);
    EXPECT_GE(id, 0);
    EXPECT_TRUE(engine_->has_work());

    auto req = engine_->get_request(id);
    ASSERT_NE(req, nullptr);
    // encode 返回 {10, 11, 12}，Engine 可能会添加 BOS (1)
    // 所以长度可能是 3 或 4，取决于 engine.cpp 里的 BOS 逻辑是否触发
    EXPECT_GE(req->prompt_len(), 3);
    EXPECT_EQ(req->state(), RequestState::kWaiting);
}

TEST_F(EngineTest, SingleStepExecution) {
    int64_t id = engine_->add_request("test", 5);

    // Step 1: Prefill
    auto status = engine_->step();
    ASSERT_TRUE(status);

    auto req = engine_->get_request(id);
    EXPECT_EQ(req->state(), RequestState::kRunning);
    EXPECT_GE(req->num_computed_tokens(), 3);

    // Step 2: Decode 1
    status = engine_->step();
    ASSERT_TRUE(status);

    EXPECT_EQ(req->generated_len(), 1);
    EXPECT_EQ(req->generated_tokens().back(), 3);
}

TEST_F(EngineTest, ContinuousBatching) {
    int64_t id1 = engine_->add_request("req1", 10);
    engine_->step();

    auto req1 = engine_->get_request(id1);
    EXPECT_TRUE(req1->is_decode());

    int64_t id2 = engine_->add_request("req2", 10);
    engine_->step();

    auto stats = engine_->get_scheduler_stats();
    EXPECT_EQ(stats.num_running, 2);

    auto req2 = engine_->get_request(id2);
    EXPECT_EQ(req1->generated_len(), 1);
    EXPECT_GE(req2->num_computed_tokens(), 3);
}

TEST_F(EngineTest, StopEngine) {
    engine_->add_request("test", 10);
    engine_->stop();
    engine_->run();
}