/**
 * @file encode.h
 * @brief Tokenizer 编解码层：SentencePiece 和 tiktoken BPE 两种实现
 *
 * 类层次：
 *   EncodeLayerBase
 *   ├── SpeEncodeLayer  — Google SentencePiece（LLaMA-2 等模型）
 *   └── BpeEncodeLayer  — tiktoken BPE（LLaMA-3 等模型）
 */
#ifndef NANO_INFER_ENCODE_H
#define NANO_INFER_ENCODE_H

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include <sentencepiece_processor.h>
#include "layer.h"
#include "nlohmann/json.hpp"
#include "tiktoken.h"
#include "unordered_dense.h"

namespace op {

/**
 * @brief Tokenizer 抽象基类
 *
 * 提供文本 ↔ Token ID 的编解码接口，作为预/后处理模块在 CPU 上运行
 */
class EncodeLayerBase : public Layer {
   public:
    explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
        : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
          has_bos_(has_bos),
          has_eos_(has_eos),
          token_model_path_(std::move(token_model_path)) {
    }

    /// @brief 编码：文本→Token ID 序列
    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    /// @brief 解码：单个 Token ID→字符串
    virtual std::string decode(int32_t token_id) const = 0;

    /// @brief 解码：Token ID 序列→完整字符串
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    /// @brief 判断是否为 EOS Token
    virtual bool is_sentence_ending(int32_t token_id) const = 0;

    virtual int32_t vocab_size() const = 0;

    virtual int32_t bos_id() const = 0;

    virtual int32_t eos_id() const = 0;

    /// @brief 第二停止符（BPE 模型用，如 <|eot_id|>），默认 -1 表示不存在
    virtual int32_t stop_token2() const {
        return -1;
    }

   protected:
    bool has_bos_ = true;
    bool has_eos_ = false;
    std::string token_model_path_;
};

/**
 * @brief 基于 SentencePiece 的 Tokenizer 实现
 *
 * 适用于 LLaMA-2 等使用 .model 文件格式的模型。
 * 通过 Google SentencePiece 库完成编解码。
 */
class SpeEncodeLayer : public EncodeLayerBase {
   public:
    explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

    int32_t bos_id() const override;

    int32_t eos_id() const override;

   private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};

/**
 * @brief 基于 tiktoken BPE 的 Tokenizer 实现
 *
 * 适用于 LLaMA-3 等使用 tiktoken.json 格式的模型。
 * 词表和特殊 token（BOS/EOS/停止符）从 JSON 文件中加载。
 * 支持 <|end_of_text|> 和 <|eot_id|> 两种停止 token。
 */
class BpeEncodeLayer : public EncodeLayerBase {
   public:
    explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

    int32_t bos_id() const override;

    int32_t eos_id() const override;

   private:
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t stop_token1_ = -1;  ///< <|end_of_text|>
    int32_t stop_token2_ = -1;  ///< <|eot_id|>
    int32_t num_token_ = 0;
    std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

}  // namespace op

#endif