#ifndef NANO_INFER_ENCODE_H
#define NANO_INFER_ENCODE_H

#include <sentencepiece_processor.h>
#include "layer.h"

namespace op {

/**
 * @brief 编码层基类
 *
 * 定义了 Tokenizer 的通用接口，用于文本与 Token ID 之间的相互转换。
 * 继承自 Layer，但运行在 CPU 上，且通常不参与计算图的 Tensor 自动流转，
 * 而是作为预处理（Pre-processing）和后处理（Post-processing）模块使用。
 */
class EncodeLayerBase : public Layer {
   public:
    /**
     * @brief 构造函数
     *
     * @param token_model_path Tokenizer 模型文件路径
     * @param has_bos 是否在编码结果开头添加 BOS (Begin of Sentence) 标记
     * @param has_eos 是否在编码结果末尾添加 EOS (End of Sentence) 标记
     */
    explicit EncodeLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
        : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
          has_bos_(has_bos),
          has_eos_(has_eos),
          token_model_path_(std::move(token_model_path)) {
    }

    /**
     * @brief 编码：将字符串转换为 Token ID 序列
     *
     * @param sentence 输入文本
     * @return std::vector<int32_t> 对应的 Token ID 列表
     */
    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    /**
     * @brief 解码：将单个 Token ID 转换为字符串
     *
     * @param token_id 输入 Token ID
     * @return std::string 对应的文本片段
     */
    virtual std::string decode(int32_t token_id) const = 0;

    /**
     * @brief 解码：将 Token ID 序列转换为完整字符串
     *
     * @param token_ids 输入 Token ID 列表
     * @return std::string 拼接后的完整文本
     */
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    /**
     * @brief 判断 Token 是否为句子结束标记 (EOS)
     * 用于生成循环中的停止条件判断。
     */
    virtual bool is_sentence_ending(int32_t token_id) const = 0;

    /**
     * @brief 获取词表大小 (Vocabulary Size)
     */
    virtual int32_t vocab_size() const = 0;

   private:
    bool has_bos_ = true;           ///< 是否添加 BOS
    bool has_eos_ = false;          ///< 是否添加 EOS
    std::string token_model_path_;  ///< 模型路径
};

/**
 * @brief 基于 SentencePiece 的编码层实现
 *
 * 封装了 Google 的 sentencepiece 库，支持 Llama2 等模型常用的 .model 文件。
 */
class SpeEncodeLayer : public EncodeLayerBase {
   public:
    /**
     * @brief 构造函数
     * 初始化 sentencepiece::SentencePieceProcessor 并加载模型。
     */
    explicit SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);

    /**
     * @brief 编码实现
     * 调用 spe->Encode，并根据 has_bos_/has_eos_ 配置插入特殊 Token。
     */
    std::vector<int32_t> encode(const std::string& sentence) const override;

    /**
     * @brief 单 Token 解码实现
     */
    std::string decode(int32_t token_id) const override;

    /**
     * @brief 序列解码实现
     */
    std::string decode(const std::vector<int32_t>& token_ids) const override;

    /**
     * @brief 判断是否为 EOS
     * 检查 token_id 是否等于 spe->eos_id()。
     */
    bool is_sentence_ending(int32_t token_id) const override;

    /**
     * @brief 获取词表大小
     */
    int32_t vocab_size() const override;

   private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor>
        spe;  ///< SentencePiece 处理器实例
};

}  // namespace op

#endif