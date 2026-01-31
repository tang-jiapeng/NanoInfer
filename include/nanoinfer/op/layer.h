#ifndef NANO_INFER_LAYER_H
#define NANO_INFER_LAYER_H
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

namespace op {
class Layer;

/**
 * @brief 算子类型枚举
 * 定义了框架支持的所有算子类型
 */
enum class LayerType : uint8_t {
    kLayerUnknown = 0,
    kLayerLinear = 1,     ///< 线性层 (全连接)
    kLayerEncode = 2,     ///< 编码层 (通常用于 Tokenizer)
    kLayerEmbedding = 3,  ///< 词嵌入层
    kLayerRMSNorm = 4,    ///< Root Mean Square Layer Normalization
    kLayerMatmul = 5,     ///< 矩阵乘法
    kLayerRoPe = 6,       ///< Rotary Positional Embeddings
    kLayerMHA = 7,        ///< Multi-Head Attention
    kLayerSoftmax = 8,    ///< Softmax 激活
    kLayerAdd = 9,        ///< 逐元素加法
    kLayerSwiGLU = 10,    ///< SwiGLU 激活函数
};

/**
 * @brief 算子抽象基类 (Interface)
 *
 * 定义了所有算子必须实现的通用接口，包括初始化、前向传播、权重加载等。
 * 不包含具体的成员变量实现，主要用于多态调用。
 */
class BaseLayer {
   public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                       base::DataType data_type, std::string layer_name = "");

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
    virtual base::Status forward(const tensor::Tensor& input1,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (双输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (三输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (四输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) = 0;

    /**
     * @brief 前向传播的便捷重载 (五输入单输出)
     */
    virtual base::Status forward(const tensor::Tensor& input1,
                                 const tensor::Tensor& input2,
                                 const tensor::Tensor& input3,
                                 const tensor::Tensor& input4,
                                 const tensor::Tensor& input5,
                                 const tensor::Tensor& output1) = 0;

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

    /**
     * @brief 检查算子状态
     * 验证输入输出 Tensor 的维度、类型是否符合当前算子的要求。
     */
    virtual base::Status check() const = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    /**
     * @brief 设置权重 Tensor
     * @param idx 权重索引
     * @param weight 权重数据
     */
    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

    /**
     * @brief 从原始指针加载权重
     *
     * @param idx 权重索引
     * @param dims 权重维度
     * @param weight_ptr 原始数据指针 (Host 端)
     * @param device_type 目标设备类型 (加载后自动迁移)
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
 * 实现了 BaseLayer 中关于输入输出 Tensor 管理、CUDA 配置传递等通用逻辑
 * 无参数的算子 (如 Add, Softmax) 可以直接继承此类
 */
class Layer : public BaseLayer {
   public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type,
                   std::string layer_name = "");

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
    base::Status check_tensor_with_dim(const tensor::Tensor& tensor,
                                       base::DeviceType device_type,
                                       base::DataType data_type, ...) const;

    base::Status check() const override;

    base::Status forward() override;

    base::Status forward(const tensor::Tensor& input1,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& output1) override;

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         const tensor::Tensor& input3, const tensor::Tensor& input4,
                         const tensor::Tensor& input5,
                         const tensor::Tensor& output1) override;

    void set_input(int32_t idx, const tensor::Tensor& input) override;

    void set_output(int32_t idx, const tensor::Tensor& output) override;

    const tensor::Tensor& get_input(int32_t idx) const override;

    const tensor::Tensor& get_output(int32_t idx) const override;

    tensor::Tensor& get_input(int32_t idx) override;

    tensor::Tensor& get_output(int32_t idx) override;

    size_t input_size() const override;

    size_t output_size() const override;

    /**
     * @brief 重置输入 Tensor 列表的大小
     */
    void reset_input_size(size_t size);

    /**
     * @brief 重置输出 Tensor 列表的大小
     */
    void reset_output_size(size_t size);

    /**
     * @brief 将算子内的 Tensor (如输入、输出) 迁移到 CUDA
     * @note 具体的子类可能需要重写此方法以处理内部特有的 Tensor
     */
    virtual void to_cuda();

    /**
     * @brief 设置 CUDA 执行配置 (Stream 等)
     */
    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

    std::shared_ptr<kernel::CudaConfig> cuda_config() const;

   protected:
    std::vector<tensor::Tensor> inputs_;               ///< 输入 Tensor 列表
    std::vector<tensor::Tensor> outputs_;              ///< 输出 Tensor 列表
    std::shared_ptr<kernel::CudaConfig> cuda_config_;  ///< CUDA 上下文
};

/**
 * @brief 带参数的算子基类
 *
 * 继承自 Layer，增加了对权重 (Weights) 和量化参数 (Scales, GroupSize) 的管理
 * 适用于 Linear, Embedding, RMSNorm 等有可学习参数的层
 */
class LayerParam : public Layer {
   public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                        bool is_quant_layer = false, std::string layer_name = "");

    size_t weight_size() const;

    /**
     * @brief 重置权重列表的大小
     * 通常在构造函数中调用，根据算子类型确定需要几个权重张量。
     */
    void reset_weight_size(size_t size);

    tensor::Tensor& get_weight(int32_t idx);

    const tensor::Tensor& get_weight(int32_t idx) const;

    /**
     * @brief 将参数也迁移到 CUDA
     * 重写了 Layer::to_cuda，额外处理 weights_ 和 scales_
     */
    void to_cuda() override;

    /**
     * @brief 设置权重 Tensor (直接拷贝)
     */
    base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

    /**
     * @brief 从原始指针加载权重 (支持量化权重拆分)
     *
     * 如果是量化层 (is_quant_layer_ == true)，该方法会自动将连续的内存块
     * 拆分为 int8 权重 (weights_) 和 float 缩放因子 (scales_)。
     */
    base::Status set_weight(
        int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

    /**
     * @brief 设置量化缩放因子 Tensor
     */
    void set_scales(const tensor::Tensor& scales);

    /**
     * @brief 设置量化分组大小
     */
    void set_group_size(int32_t group_size);

    /**
     * @brief 获取量化缩放因子的数量
     */
    int32_t get_scale_num() const;

   protected:
    int32_t group_size_ = 0;               ///< 量化分组大小 (如 64)
    bool is_quant_layer_ = false;          ///< 是否为量化层
    tensor::Tensor scales_;                ///< 量化缩放因子 (Fp32)
    std::vector<tensor::Tensor> weights_;  ///< 权重列表
};
}  // namespace op

#endif  // NANO_INFER_LAYER_H
