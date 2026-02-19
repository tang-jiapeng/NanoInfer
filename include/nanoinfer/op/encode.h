/**
 * @file encode.h
 * @brief Tokenizer 编解码层 (SentencePiece 实现)
 */
#ifndef NANO_INFER_ENCODE_H
#define NANO_INFER_ENCODE_H

#include <sentencepiece_processor.h>
#include "layer.h"

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

   protected:
    bool has_bos_ = true;
    bool has_eos_ = false;
    std::string token_model_path_;
};

/// @brief 基于 SentencePiece 的 Tokenizer 实现
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

}  // namespace op

#endif