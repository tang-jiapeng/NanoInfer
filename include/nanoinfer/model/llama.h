#ifndef NANO_INFER_MODEL_LLAMA_H
#define NANO_INFER_MODEL_LLAMA_H

#include "model.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/op/layer.h"

namespace model {

/**
 * @brief Llama 模型层容器
 *
 * 集中管理所有的算子对象。
 * 1. 有参数的层 (Weights)：使用 vector 存储，对应每一层 Transformer Block。
 * 2. 无参数的层 (Stateless)：如 RoPE, Add, SwiGLU，通常全模型复用同一个对象。
 */
struct LLamaLayers {
    std::shared_ptr<op::Layer> add_layer_;     ///< 残差连接加法层 (Add)
    std::shared_ptr<op::Layer> rope_layer_;    ///< 旋转位置编码层 (RoPE)
    std::shared_ptr<op::Layer> swiglu_layer_;  ///< FFN 激活层 (SwiGLU)
    std::shared_ptr<op::Layer> mha_layer_;     ///< 多头注意力计算核心 (MHA)

    // Attention 模块的线性投影层
    std::vector<std::shared_ptr<op::Layer>> wq_layers_;  ///< Query Projection
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;  ///< Key Projection
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;  ///< Value Projection
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;  ///< Output Projection

    // FeedForward 模块的线性层
    std::vector<std::shared_ptr<op::Layer>> w1_layers_;  ///< Gate Projection
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;  ///< Down Projection
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;  ///< Up Projection

    // 这里统一存储了所有 RMSNorm 层
    // 布局顺序：[Attention Norms (0...N-1), FFN Norms (0...N-1), Final Norm]
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;  ///< Attention Norm

    std::shared_ptr<op::Layer> cls_layer_;        ///< 最终分类层 (LM Head / Linear)
    std::shared_ptr<op::Layer> embedding_layer_;  ///< Token Embedding 层

    /**
     * @brief 将所有层迁移到 CUDA
     * 并设置统一的 CUDA Stream 配置。
     */
    void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

/**
 * @brief Llama 模型实现类
 *
 * 实现了标准的 Llama Decoder-Only 架构：
 * Embedding -> [RMSNorm -> Attention -> Add -> RMSNorm -> FFN -> Add] * N -> RMSNorm ->
 * Linear
 */
class LLamaModel : public Model {
   public:
    /**
     * @brief 构造函数
     *
     * @param tokenizer_type 分词器类型
     * @param token_path Tokenizer 模型路径
     * @param model_path 权重文件路径
     * @param is_quant_model 是否为量化模型
     */
    explicit LLamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                        std::string model_path, bool is_quant_model);

    /**
     * @brief 初始化 Llama 模型
     * 调用基类的 init 流程，并额外分配 CUDA Stream 资源。
     */
    base::Status init(base::DeviceType device_type) override;

    /**
     * @brief 对外预测接口
     *
     * 1. 预处理：Embedding。
     * 2. 执行 forward 计算图。
     * 3. 后处理：采样得到 Next Token。
     */
    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         bool is_prompt, int& next) const override;

    /**
     * @brief 核心前向计算图
     *
     * 循环执行 transformer_layers_num 次 Block 计算。
     */
    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         int& next) const override;

    /**
     * @brief 执行 Embedding 查找
     */
    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

   private:
    /**
     * @brief 显存预分配
     * 根据 max_seq_len 和 hidden_size 计算 KV Cache 和中间 Buffer 所需内存并分配。
     */
    void init_mem() override;

    /**
     * @brief 构建计算图中的所有层
     */
    base::Status create_layers() override;

    /**
     * @brief 创建浮点参数层 (Weights)
     */
    void create_param_layers() override;

    /**
     * @brief 创建无参数层 (Non-parametric)
     */
    void create_nonparam_layers() override;

    /**
     * @brief 创建量化参数层 (Quantized Weights)
     */
    void create_param_quant_layers() override;

    /**
     * @brief 执行 Multi-Head Attention 计算
     * 包括：RoPE -> Update KV Cache -> Attention Score -> Context Output
     */
    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

    /**
     * @brief 执行 Attention 之前的 RMSNorm
     */
    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

    /**
     * @brief 执行 Feed-Forward Network (FFN)
     * 流程：RMSNorm (Post-Attn) -> (Gate * Up) -> SwiGLU -> Down -> Add Residual
     */
    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

    /**
     * @brief 执行 QKV 线性投影
     * 输入 -> Wq/Wk/Wv -> Query/Key/Value Tensors
     */
    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

    /**
     * @brief 计算最终的分类 Logits
     * 最后的 RMSNorm -> Linear (Vocab Projection)
     */
    void cls_logits(const tensor::Tensor& input) const;

    /**
     * @brief 后处理采样
     * Logits -> Probabilities -> Next Token ID
     */
    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

   private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;  ///< CUDA 执行配置 (Stream)
    std::unique_ptr<LLamaLayers> llama_layers_;        ///< 所有算子的容器
};

}  // namespace model

#endif  // NANO_INFER_MODEL_LLAMA_H