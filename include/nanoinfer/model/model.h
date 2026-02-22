/**
 * @file model.h
 * @brief 模型抽象基类：权重加载 / Tokenizer / 批处理 Forward 接口
 */
#ifndef NANO_INFER_MODEL_H
#define NANO_INFER_MODEL_H

#include "config.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/op/embedding.h"
#include "nanoinfer/op/encode.h"
#include "nanoinfer/tensor/tensor.h"
#include "raw_model_data.h"
#include "sentencepiece_processor.h"

namespace model {

/// @brief 批处理 Forward 输入包
struct ForwardBatch {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> seq_ids;

    tensor::Tensor block_table;  ///< [batch_size, max_blocks_per_seq]

    std::vector<int32_t> context_lens;
    int32_t max_context_len = 0;

    int32_t batch_size = 0;
    bool is_prefill = false;
};

/**
 * @brief 模型抽象基类
 *
 * 管理权重加载、Tokenizer、算子构建，提供 Continuous Batching Forward 接口。
 */
class Model {
   public:
    explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                   std::string token_path, std::string model_path, bool is_quant_model);

    /// @brief 初始化：读取权重 → 创建层 → 资源迁移
    virtual base::Status init(base::DeviceType device_type) = 0;

    /// @brief 注入 KV Cache 物理 Tensor（Engine 分配，Model 持有引用）
    virtual void set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                              const std::vector<tensor::Tensor>& value_caches) {
        (void)key_caches;
        (void)value_caches;
    }

    /// @brief 批处理 Forward，输出 Logits [total_tokens, vocab_size]
    virtual base::Status forward_batched(const ForwardBatch& input, tensor::Tensor& logits) {
        return base::error::FunctionNotImplement("forward_batched not implemented");
    }

    const TransformerConfig& config() const {
        return *config_;
    }

    base::ModelType model_type() const;
    const std::string& token_path() const;
    const std::string& model_path() const;

    virtual bool is_sentence_ending(int32_t token_idx) const;

    virtual std::string decode(int32_t token_idx) const;

    virtual std::string decode(std::vector<int32_t> token_idxs) const;

    virtual std::vector<int32_t> encode(const std::string& sentence) const;

   protected:
    /// @brief mmap 读取模型文件并解析文件头
    virtual base::Status read_model_file();

    virtual base::Status create_encode_layer();

    /// @brief 加载流程：create_encode → read_model → create_layers
    virtual base::Status gen_model_from_file();

    /// @brief 从 ModelConfig 推导 kv_dim / kv_mul / head_size 等
    virtual base::Status generate_model_infos(const ModelConfig& config) const;

   private:
    virtual base::Status create_layers() = 0;
    virtual void create_param_layers() = 0;
    virtual void create_nonparam_layers() = 0;
    virtual void create_param_quant_layers() = 0;

   protected:
    int32_t group_size_ = 1;
    bool is_quant_model_ = false;
    std::unique_ptr<TransformerConfig> config_;

    std::string token_path_;
    std::string model_path_;
    std::unique_ptr<op::EncodeLayerBase> encode_layer_;
    std::shared_ptr<RawModelData> raw_model_data_;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
    base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};

}  // namespace model

#endif  // NANO_INFER_MODEL_H