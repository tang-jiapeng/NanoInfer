#ifndef NANO_INFER_MODEL_LLAMA_H
#define NANO_INFER_MODEL_LLAMA_H

#include "model.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/op/layer.h"
#include "nanoinfer/tensor/tensor.h"
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
    std::shared_ptr<op::Layer> swiglu_layer_;  ///< FFN 激活层 (SwiGLU)
    std::shared_ptr<op::Layer> attn_layer_;  ///< Attention 层 (RoPE + Prefill/Decode Attention)

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

    ~LLamaModel() = default;

    /**
     * @brief 初始化：加载权重，构建计算图
     */
    base::Status init(base::DeviceType device_type) override;

    /**
     * @brief [Core] 批处理前向传播
     * 负责调度每一层的计算：Norm -> QKV -> RoPE -> PagedAttn -> FFN -> ...
     */
    base::Status forward_batched(const ForwardBatch& input, tensor::Tensor& logits) override;

    /**
     * @brief 注入 KV Cache
     * Model 类直接持有 KV Cache 的物理 Tensor，
     * 在 forward_batched 循环中，将对应的 Cache 传递给 Attention 算子。
     */
    void set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                      const std::vector<tensor::Tensor>& value_caches) override;

   private:
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
     * @brief 预计算 RoPE 表 (Sin/Cos Cache)
     */
    void init_rope_cache();

   private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;  ///< CUDA 执行配置 (Stream)
    std::unique_ptr<LLamaLayers> llama_layers_;        ///< 所有算子的容器

    tensor::Tensor sin_cache_;
    tensor::Tensor cos_cache_;

    // KV Cache 引用 (由 Engine 分配，Model 持有引用)
    std::vector<tensor::Tensor> key_caches_;
    std::vector<tensor::Tensor> value_caches_;
};

}  // namespace model

#endif  // NANO_INFER_MODEL_LLAMA_H