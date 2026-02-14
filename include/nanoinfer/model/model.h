#ifndef NANO_INFER_MODEL_H
#define NANO_INFER_MODEL_H

#include "config.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/op/embedding.h"
#include "nanoinfer/op/encode.h"
#include "nanoinfer/sampler/argmax_sampler.h"
#include "nanoinfer/tensor/tensor.h"
#include "raw_model_data.h"
#include "sentencepiece_processor.h"

namespace model {

struct ForwardBatch {
    // 拼接后的 Token ID 列表 (Batch 中所有请求的当前步 Token)
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;  ///< 拼接后的 Position ID 列表 (用于 RoPE)
    std::vector<int32_t> seq_ids;    ///< 对应的 Sequence ID (用于调试或特定逻辑)

    tensor::Tensor block_table;  ///< GPU Tensor: [batch_size, max_blocks_per_seq]

    // 每个请求的上下文长度 (用于 Attention Mask/Scale)
    std::vector<int32_t> context_lens;
    int32_t max_context_len = 0;  ///< 当前 Batch 中最大的上下文长度 (用于 Kernel 配置)

    int32_t batch_size = 0;   ///< 当前 Batch 的序列数量
    bool is_prefill = false;  ///< 是否为 Prefill 阶段 (一次性处理所有 prompt tokens)
};

/**
 * @brief 模型抽象基类
 *
 * 专注于支持 Continuous Batching 和 PagedAttention 的推理流程
 * 职责：
 * 1. 模型文件加载与解析
 * 2. Tokenizer 管理
 * 3. 权重加载与算子构建
 * 4. 提供批处理前向传播接口
 */
class Model {
   public:
    /**
     * @brief 构造函数
     *
     * @param tokenizer_type 分词器类型 (如 SentencePiece)
     * @param model_type 模型架构类型 (如 Llama2)
     * @param token_path Tokenizer 模型文件路径
     * @param model_path 模型权重文件路径
     * @param is_quant_model 是否为量化模型 (决定加载 FP32 还是 Int8 权重)
     */
    explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                   std::string token_path, std::string model_path, bool is_quant_model);

    /**
     * @brief 模型初始化 (核心流程)
     * 1. 读取权重文件
     * 2. 创建各层对象
     * 3. 资源迁移
     */
    virtual base::Status init(base::DeviceType device_type) = 0;

    // PagedAttention & Continuous Batching 接口

    /**
     * @brief 注入 KV Cache 物理张量
     * * 在 Engine 初始化或分配 KV Cache 后调用。
     * Model 应当保存这些 Tensor 的引用/指针，供 Attention 算子使用。
     * Model 不负责管理这些 Tensor 的生命周期。
     * * @param key_caches 每层的 Key Cache 物理张量 (Pool)
     * @param value_caches 每层的 Value Cache 物理张量 (Pool)
     */
    virtual void set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                              const std::vector<tensor::Tensor>& value_caches) {
        (void)key_caches;
        (void)value_caches;
    }

    /**
     * @brief 执行批处理前向传播
     * * 基于 BlockTable 进行非连续显存访问。
     * * @param input 批处理输入包 (包含 tokens, positions, block_table 等)
     * @param logits [输出] 预测的 Logits 张量 [total_tokens, vocab_size]
     */
    virtual base::Status forward_batched(const ForwardBatch& input, tensor::Tensor& logits) {
        return base::error::FunctionNotImplement("forward_batched not implemented");
    }

    const TransformerConfig& config() const {
        return *config_;
    }

    base::ModelType model_type() const;
    const std::string& token_path() const;
    const std::string& model_path() const;

    /**
     * @brief 判断 Token 是否为结束符 (EOS)
     */
    virtual bool is_sentence_ending(int32_t token_idx) const;

    /**
     * @brief 解码单个 Token ID 为字符串
     */
    virtual std::string decode(int32_t token_idx) const;

    /**
     * @brief 解码 Token ID 序列为完整文本
     */
    virtual std::string decode(std::vector<int32_t> token_idxs) const;

    /**
     * @brief 编码文本为 Token ID 序列
     */
    virtual std::vector<int32_t> encode(const std::string& sentence) const;

   protected:
    /**
     * @brief 读取模型权重文件
     *
     * 使用 mmap 将模型文件映射到内存，并解析文件头 (ModelConfig)。
     */
    virtual base::Status read_model_file();

    /**
     * @brief 创建 Tokenizer 编码层 (如 SpeEncodeLayer)
     */
    virtual base::Status create_encode_layer();

    /**
     * @brief 编排模型加载流程
     *
     * 依次执行：创建 Encode 层 -> 读取模型文件(mmap) -> 创建算子层(create_layers)。
     */
    virtual base::Status gen_model_from_file();

    /**
     * @brief 根据配置生成模型元数据
     *
     * 计算派生参数，如 kv_dim, kv_mul, head_size 等。
     */
    virtual base::Status generate_model_infos(const ModelConfig& config) const;

   private:
    /**
     * @brief 创建所有层
     * 实例化 Attention, FeedForward, RMSNorm 等算子
     */
    virtual base::Status create_layers() = 0;

    /**
     * @brief 创建参数层 (Linear, Embedding 等)
     */
    virtual void create_param_layers() = 0;

    /**
     * @brief 创建非参数层 (RoPE, SwiGLU 等)
     */
    virtual void create_nonparam_layers() = 0;

    /**
     * @brief 创建量化参数层
     */
    virtual void create_param_quant_layers() = 0;

   protected:
    int32_t group_size_ = 1;                     ///< 量化分组大小
    bool is_quant_model_ = false;                ///< 是否为量化模型
    std::unique_ptr<TransformerConfig> config_;  ///< 模型配置信息

    std::string token_path_;
    std::string model_path_;
    std::unique_ptr<op::EncodeLayerBase> encode_layer_;  ///< 分词器实例
    std::shared_ptr<RawModelData> raw_model_data_;       ///< 原始权重数据
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
    base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};

}  // namespace model

#endif  // NANO_INFER_MODEL_H