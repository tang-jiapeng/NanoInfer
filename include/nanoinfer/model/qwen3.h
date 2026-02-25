#ifndef NANO_INFER_MODEL_QWEN3_H
#define NANO_INFER_MODEL_QWEN3_H

#include "model.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/op/layer.h"
#include "nanoinfer/tensor/tensor.h"

namespace model {

struct Qwen3Layers {
    std::shared_ptr<op::Layer> add_layer_;
    std::shared_ptr<op::Layer> swiglu_layer_;
    std::shared_ptr<op::Layer> attn_layer_;

    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;

    std::vector<std::shared_ptr<op::Layer>> w1_layers_;
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;

    /// 布局: [AttnNorm x N, FFNNorm x N, FinalNorm]
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;

    std::shared_ptr<op::Layer> cls_layer_;
    std::shared_ptr<op::Layer> embedding_layer_;

    /// @brief 迁移所有层到 CUDA 并设置统一 Stream
    void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Qwen3Model : public Model {
   public:
    explicit Qwen3Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                        std::string token_path, std::string model_path, bool is_quant_model);

    ~Qwen3Model() = default;

    base::Status init(base::DeviceType device_type) override;

    /// @brief 批处理 Forward：逐层 Norm → QKV → RoPE → PagedAttn → FFN
    base::Status forward_batched(const ForwardBatch& input, tensor::Tensor& logits) override;

    void set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                      const std::vector<tensor::Tensor>& value_caches) override;

   private:
    base::Status create_layers() override;
    void create_param_layers() override;
    void create_nonparam_layers() override;
    void create_param_quant_layers() override;

    /// @brief 预计算 RoPE Sin/Cos Cache
    void init_rope_cache();

   private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<Qwen3Layers> qwen3_layers_;

    tensor::Tensor sin_cache_;
    tensor::Tensor cos_cache_;

    std::vector<tensor::Tensor> key_caches_;
    std::vector<tensor::Tensor> value_caches_;
};

}  // namespace model

#endif