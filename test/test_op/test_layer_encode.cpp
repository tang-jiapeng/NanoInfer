/**
 * @file test_layer_encode.cpp
 * @brief BpeEncodeLayer 单元测试
 *
 * 测试范围：
 *   1. Metadata       — bos_id / eos_id / vocab_size 与 LLaMA-3 规范值匹配
 *   2. StopTokens     — is_sentence_ending 对 <|end_of_text|> 和 <|eot_id|> 返回 true
 *   3. Encode/Decode roundtrip — encode 后 decode（去掉 BOS）应还原原文
 *   4. BOS 注入        — has_bos=true 时首 token 为 bos_id
 *   5. Decode 单/多一致性 — decode(id) 与 decode({id}) 结果相同
 *   6. 空串编码        — has_bos=true 时编码空串只含 BOS token
 *
 * 前置条件：
 *   需要 ./models/llama3/tokenizer.json 文件，不存在时所有用例自动跳过。
 *   路径可通过编译时宏 -DTEST_LLAMA3_TOKENIZER_PATH=xxx 覆盖。
 *
 * LLaMA-3 tokenizer 已知值（Meta 官方发布，固定不变）：
 *   bos_id  = 128000  (<|begin_of_text|>)
 *   eos_id  = 128001  (<|end_of_text|>)
 *   eot_id  = 128009  (<|eot_id|>)
 *   vocab_size = 128256
 */
#include <gtest/gtest.h>
#include <fstream>
#include "nanoinfer/op/encode.h"

// 可通过 -DTEST_LLAMA3_TOKENIZER_PATH="..." 在编译期覆盖
#ifndef TEST_LLAMA3_TOKENIZER_PATH
#define TEST_LLAMA3_TOKENIZER_PATH "./models/llama3/tokenizer.json"
#endif

static const std::string kTokenizerPath = TEST_LLAMA3_TOKENIZER_PATH;

// LLaMA-3 官方固定 token ID
static constexpr int32_t kBosId = 128000;
static constexpr int32_t kEosId = 128001;
static constexpr int32_t kEotId = 128009;
static constexpr int32_t kVocabSize = 128256;
// 普通内容 token，不是停止符（"Hello" 对应的 BPE token）
static constexpr int32_t kNormalToken = 9906;

// ===========================================================================
// BpeEncodeLayerTest
// ===========================================================================
class BpeEncodeLayerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // 文件不存在时直接跳过，不让测试 Fail
        std::ifstream f(kTokenizerPath);
        if (!f.is_open()) {
            GTEST_SKIP() << "Tokenizer file not found: " << kTokenizerPath
                         << "\nPlace LLaMA-3 tokenizer.json at that path to run these tests.";
        }

        // has_bos=true, has_eos=false（推理时通常只加 BOS）
        enc_ = std::make_unique<op::BpeEncodeLayer>(kTokenizerPath, /*has_bos=*/true,
                                                    /*has_eos=*/false);
    }

    std::unique_ptr<op::BpeEncodeLayer> enc_;
};

// ---------------------------------------------------------------------------
// 1. Metadata
// ---------------------------------------------------------------------------
TEST_F(BpeEncodeLayerTest, BosId) {
    EXPECT_EQ(enc_->bos_id(), kBosId);
}

TEST_F(BpeEncodeLayerTest, EosId) {
    EXPECT_EQ(enc_->eos_id(), kEosId);
}

TEST_F(BpeEncodeLayerTest, VocabSize) {
    EXPECT_EQ(enc_->vocab_size(), kVocabSize);
}

// ---------------------------------------------------------------------------
// 2. StopTokens
// ---------------------------------------------------------------------------
// <|end_of_text|> 是停止符
TEST_F(BpeEncodeLayerTest, IsEndingForEos) {
    EXPECT_TRUE(enc_->is_sentence_ending(kEosId));
}

// <|eot_id|> 也是停止符（Assistant 发言结束）
TEST_F(BpeEncodeLayerTest, IsEndingForEot) {
    EXPECT_TRUE(enc_->is_sentence_ending(kEotId));
}

// 普通 token 不是停止符
TEST_F(BpeEncodeLayerTest, NotEndingForNormalToken) {
    EXPECT_FALSE(enc_->is_sentence_ending(kNormalToken));
}

// bos_id 本身也不是停止符
TEST_F(BpeEncodeLayerTest, NotEndingForBos) {
    EXPECT_FALSE(enc_->is_sentence_ending(kBosId));
}

// ---------------------------------------------------------------------------
// 3. Encode / Decode roundtrip
// ---------------------------------------------------------------------------
// 编码后去掉 BOS，再解码，应还原原始字符串
TEST_F(BpeEncodeLayerTest, RoundtripSimple) {
    const std::string sentence = "Hello world";
    auto ids = enc_->encode(sentence);

    ASSERT_FALSE(ids.empty());
    ASSERT_EQ(ids.front(), kBosId);

    // 去掉 BOS 再解码
    std::vector<int32_t> content(ids.begin() + 1, ids.end());
    EXPECT_EQ(enc_->decode(content), sentence);
}

TEST_F(BpeEncodeLayerTest, RoundtripMultiWord) {
    const std::string sentence = "The quick brown fox jumps over the lazy dog";
    auto ids = enc_->encode(sentence);

    ASSERT_GT(ids.size(), 1u);
    ASSERT_EQ(ids.front(), kBosId);

    std::vector<int32_t> content(ids.begin() + 1, ids.end());
    EXPECT_EQ(enc_->decode(content), sentence);
}

// 中英混合（测试非 ASCII 字符路径）
TEST_F(BpeEncodeLayerTest, RoundtripUnicode) {
    const std::string sentence = "Hello 世界";
    auto ids = enc_->encode(sentence);

    ASSERT_GT(ids.size(), 1u);
    ASSERT_EQ(ids.front(), kBosId);

    std::vector<int32_t> content(ids.begin() + 1, ids.end());
    EXPECT_EQ(enc_->decode(content), sentence);
}

// ---------------------------------------------------------------------------
// 4. BOS 注入
// ---------------------------------------------------------------------------
// has_bos=true 时，encode 输出的首元素必须是 bos_id
TEST_F(BpeEncodeLayerTest, EncodePrependsBos) {
    auto ids = enc_->encode("anything");
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids.front(), kBosId);
}

// has_bos=false 时，首元素不应是 bos_id
TEST_F(BpeEncodeLayerTest, EncodeNoBos) {
    op::BpeEncodeLayer enc_no_bos(kTokenizerPath, /*has_bos=*/false, /*has_eos=*/false);
    auto ids = enc_no_bos.encode("Hello");
    ASSERT_FALSE(ids.empty());
    EXPECT_NE(ids.front(), kBosId);
}

// has_eos=true 时，encode 输出的尾元素必须是 eos_id
TEST_F(BpeEncodeLayerTest, EncodeAppendsEos) {
    op::BpeEncodeLayer enc_with_eos(kTokenizerPath, /*has_bos=*/false, /*has_eos=*/true);
    auto ids = enc_with_eos.encode("Hello");
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids.back(), kEosId);
}

// ---------------------------------------------------------------------------
// 5. decode 单/多 token 一致性
// ---------------------------------------------------------------------------
// decode(id) 应等价于 decode({id})
TEST_F(BpeEncodeLayerTest, DecodeSingleConsistency) {
    // 先 encode 一个已知短词，取第一个内容 token
    auto ids = enc_->encode("Hello");
    ASSERT_GE(ids.size(), 2u);  // at least BOS + 1 content token
    const int32_t first_token = ids[1];

    EXPECT_EQ(enc_->decode(first_token), enc_->decode(std::vector<int32_t>{first_token}));
}

// ---------------------------------------------------------------------------
// 6. 空串编码
// ---------------------------------------------------------------------------
// has_bos=true 时，空串编码结果只含 BOS token，无其他内容
TEST_F(BpeEncodeLayerTest, EncodeEmptyStringWithBos) {
    auto ids = enc_->encode("");
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], kBosId);
}

// has_bos=false 时，空串编码结果为空
TEST_F(BpeEncodeLayerTest, EncodeEmptyStringNoBos) {
    op::BpeEncodeLayer enc_no_bos(kTokenizerPath, /*has_bos=*/false, /*has_eos=*/false);
    auto ids = enc_no_bos.encode("");
    EXPECT_TRUE(ids.empty());
}
