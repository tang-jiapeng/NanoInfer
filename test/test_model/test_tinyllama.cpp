#include <gtest/gtest.h>
#include <glog/logging.h>
#include "nanoinfer/model/llama.h"
#include "nanoinfer/base/base.h"

// 请修改为你的实际路径
const std::string MODEL_PATH = "./models/llama2/llama2_fp32.bin";
const std::string TOKEN_PATH = "./models/llama2/tokenizer.model";

class TestLLamaModel : public model::LLamaModel {
public:
    using model::LLamaModel::LLamaModel; // 继承构造函数

    // 增加一个 Public 方法来获取权重指针
    const float* get_raw_weights_ptr() const {
        if (!raw_model_data_) return nullptr;
        // weight_data 是 void* 或 int8_t*，强转为 float*
        return reinterpret_cast<const float*>(raw_model_data_->weight_data);
    }
};

class TinyLlamaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 使用 TestLLamaModel 实例化
        model_ = std::make_unique<TestLLamaModel>(
            base::TokenizerType::kEncodeSpe, 
            TOKEN_PATH, 
            MODEL_PATH, 
            false // is_quant_model
        );
    }

    // 这里必须是 TestLLamaModel 类型，不能是 LLamaModel
    std::unique_ptr<TestLLamaModel> model_;
};

// 1. 测试元数据是否符合 TinyLlama 1.1B 的参数
TEST_F(TinyLlamaTest, MetadataCheck) {
    auto status = model_->init(base::DeviceType::kDeviceCPU);
    ASSERT_EQ(status, base::error::Success());

    const auto& config = model_->config();
    
    LOG(INFO) << "Checking TinyLlama parameters...";
    
    // TinyLlama-1.1B Specs
    EXPECT_EQ(config.dim_, 2048);           // Hidden Size
    EXPECT_EQ(config.hidden_dim_, 5632);    // Intermediate Size
    EXPECT_EQ(config.layer_num_, 22);       // Layers
    EXPECT_EQ(config.head_num_, 32);        // Query Heads
    EXPECT_EQ(config.kv_head_num_, 4);      // KV Heads (GQA 验证点!)
    
    // Vocab size 可能会有正负之分 (取决于 is_shared_weight 逻辑)
    // 脚本里如果 shared，vocab_size 是正数；不 shared 是负数。
    // Model::generate_model_infos 里取了 abs
    EXPECT_EQ(std::abs(config.vocab_size_), 32000); 
    
    EXPECT_EQ(config.head_size_, 64);       // 2048 / 32
    EXPECT_EQ(config.kv_mul_, 8);           // 32 / 4 = 8 (GQA Factor)
}

// 2. 测试 Tokenizer (验证 Hello world)
TEST_F(TinyLlamaTest, TokenizerCheck) {
    // 即使不加载模型权重，也可以测 Tokenizer（前提是 init 允许）
    // 为了安全，我们先完整 init
    model_->init(base::DeviceType::kDeviceCPU);
    
    std::string text = "Hello world";
    std::vector<int32_t> ids = model_->encode(text);
    
    LOG(INFO) << "Encoded 'Hello world':";
    for (auto id : ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Llama Tokenizer (SentencePiece) 预期:
    // BOS (1) + "Hello" (15043) + " world" (3186)
    // 注意：有些实现会自动加 BOS，有些不会，取决于 SpeEncodeLayer 的参数
    
    ASSERT_GE(ids.size(), 2);
    
    // 检查是否包含 Hello (15043) 和 world (3186)
    // 允许有 BOS (1) 在前面
    bool found_hello = false;
    bool found_world = false;
    for (auto id : ids) {
        if (id == 15043) found_hello = true;
        if (id == 3186) found_world = true;
    }
    
    EXPECT_TRUE(found_hello) << "Token ID for 'Hello' (15043) not found";
    EXPECT_TRUE(found_world) << "Token ID for ' world' (3186) not found";
}

TEST_F(TinyLlamaTest, WeightValueCheck) {
    auto status = model_->init(base::DeviceType::kDeviceCPU);
    ASSERT_EQ(status, base::error::Success());

    // 获取权重指针
    const float* weights = model_->get_raw_weights_ptr();
    ASSERT_NE(weights, nullptr) << "Weight pointer is null, mmap failed?";

    // Python 脚本提取的 Ground Truth
    // (6.28829e-06, 4.351139e-06, 4.32133e-06, 9.65595e-06, 4.79817e-06)
    std::vector<float> expected_weights = {
        6.288290023803711e-06f,
        4.351139068603516e-06f,
        4.32133674621582e-06f,
        9.655952453613281e-06f,
        4.798173904418945e-06f
    };

    LOG(INFO) << "Comparing first 5 weights with Python Ground Truth...";
    
    for (size_t i = 0; i < expected_weights.size(); ++i) {
        // 使用 EXPECT_NEAR 进行浮点数比较，容忍度设为 1e-10
        // 如果这里失败，说明文件读取偏移量错位了
        EXPECT_NEAR(weights[i], expected_weights[i], 1e-10) 
            << "Mismatch at weight index [" << i << "]";
            
        LOG(INFO) << "Idx " << i << ": Actual=" << weights[i] << ", Expected=" << expected_weights[i];
    }
}