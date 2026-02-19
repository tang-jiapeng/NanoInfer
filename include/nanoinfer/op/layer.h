/**
 * @file layer.h
 * @brief 算子层体系：BaseLayer → Layer → LayerParam
 */
#ifndef NANO_INFER_LAYER_H
#define NANO_INFER_LAYER_H
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

namespace op {
class Layer;

/// @brief 算子类型枚举
enum class LayerType : uint8_t {
    kLayerUnknown = 0,
    kLayerLinear = 1,      ///< Linear (全连接)
    kLayerEncode = 2,      ///< Tokenizer 编码
    kLayerEmbedding = 3,   ///< Token Embedding
    kLayerRMSNorm = 4,     ///< RMSNorm
    kLayerMatmul = 5,      ///< Matmul (矩阵乘法)
    kLayerRoPe = 6,        ///< RoPE (旋转位置编码)
    kLayerMHA = 7,         ///< Multi-Head Attention
    kLayerSoftmax = 8,     ///< Softmax
    kLayerAdd = 9,         ///< 逐元素加法
    kLayerSwiGLU = 10,     ///< SwiGLU 激活
    kLayerAttention = 11,  ///< Attention (Prefill + Decode)
};

/**
 * @brief 算子抽象基类 (Interface)
 *
 * 定义 init / forward / set_input / set_output / check / set_weight 等统一接口
 */
class BaseLayer {
   public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                       std::string layer_name = "");

    base::DataType data_type() const;

    LayerType layer_type() const;

    /**
     * @brief 算子初始化
     * 用于分配临时缓冲区、预计算常量等准备工作。
     */
    virtual base::Status init() = 0;

    /**
     * @brief 执行前向传播 (核心接口)
     *
     * 具体的计算逻辑在此实现。调用前需确保输入输出 Tensor 已通过 set_input/set_output
     * 设置。
     */
    virtual base::Status forward() = 0;

    /**
     * @brief 前向传播的便捷重载 (单输入单输出)
     * 内部会自动调用 set_input/set_output 然后执行 forward()
     */
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (双输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (三输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (四输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (五输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

    /**
     * @brief 设置指定索引的输入 Tensor
     */
    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

    /**
     * @brief 设置指定索引的输出 Tensor
     */
    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    /// @brief 校验输入输出 Tensor 的维度、类型是否符合要求
    virtual base::Status check() const = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

    /**
     * @brief 从原始指针加载权重，加载后可自动迁移到目标设备
     */
    virtual base::Status set_weight(
        int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name);

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

   protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

/**
 * @brief 通用算子实现类
 *
 * 管理输入输出 Tensor 列表与 CUDA 配置。无参数算子 (Add, SwiGLU 等) 直接继承此类
 */
class Layer : public BaseLayer {
   public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

    base::Status init() override;

    /**
     * @brief 校验单个 Tensor 的基本属性 (非空、设备、数据类型)
     */
    base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                              base::DataType data_type) const;

    /**
     * @brief 校验 Tensor 的属性及其维度
     *
     * @param ... 可变参数，依次列出期望的各维度大小 (int32_t)
     */
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                       base::DataType data_type, ...) const;

    base::Status check() const override;

    base::Status forward() override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& input5, const tensor::Tensor& output1) override;

    void set_input(int32_t idx, const tensor::Tensor& input) override;

    void set_output(int32_t idx, const tensor::Tensor& output) override;

    const tensor::Tensor& get_input(int32_t idx) const override;

    const tensor::Tensor& get_output(int32_t idx) const override;

    tensor::Tensor& get_input(int32_t idx) override;

    tensor::Tensor& get_output(int32_t idx) override;

    size_t input_size() const override;

    size_t output_size() const override;

    void reset_input_size(size_t size);

    void reset_output_size(size_t size);

    /// @brief 将算子内的 Tensor 迁移到 CUDA，子类可重写以处理额外成员
    virtual void to_cuda();

    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

    std::shared_ptr<kernel::CudaConfig> cuda_config() const;

    // ---- Attention 相关配置 (AttentionLayer 重写，默认空实现) ----

    virtual void set_kv_cache(const tensor::Tensor& /*key_cache*/,
                              const tensor::Tensor& /*value_cache*/) {
    }

    virtual void set_rope_cache(const tensor::Tensor& /*sin_cache*/,
                                const tensor::Tensor& /*cos_cache*/) {
    }

    virtual void set_prefill(bool /*is_prefill*/) {
    }

    virtual void set_context_len(int32_t /*context_len*/) {
    }

   protected:
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

/**
 * @brief 带参数的算子基类
 *
 * 在 Layer 基础上增加权重 (Weights) 和量化参数 (Scales / GroupSize) 管理
 */
class LayerParam : public Layer {
   public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                        bool is_quant_layer = false, std::string layer_name = "");

    size_t weight_size() const;

    void reset_weight_size(size_t size);

    tensor::Tensor& get_weight(int32_t idx);

    const tensor::Tensor& get_weight(int32_t idx) const;

    /// @brief 迁移到 CUDA，额外处理 weights_ 和 scales_
    void to_cuda() override;

    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

    /**
     * @brief 从原始指针加载权重
     * @note 量化层自动拆分为 int8 权重 + float Scales
     */
    base::Status set_weight(
        int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

    void set_scales(const tensor::Tensor& scales);

    void set_group_size(int32_t group_size);

    int32_t get_scale_num() const;

   protected:
    int32_t group_size_ = 0;
    bool is_quant_layer_ = false;
    tensor::Tensor scales_;
    std::vector<tensor::Tensor> weights_;
};
}  // namespace op

#endif  // NANO_INFER_LAYER_H
