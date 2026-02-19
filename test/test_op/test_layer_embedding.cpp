#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>
#include "layer_test_utils.h"
#include "nanoinfer/op/embedding.h"

// ===========================================================================
// EmbeddingLayerTest — 测试 EmbeddingLayer:
//   check() 错误路径:
//     - 输入为空
//     - 权重维度不匹配 (vocab_size, dim)
//     - 输出维度不匹配 (token_num, dim)
//   set_weight 从裸指针加载权重表
//   forward() CPU / CUDA 数值正确性:
//     - 验证查表结果等于权重矩阵对应行
//
// 数值设计:
//   vocab_size=16, dim=8
//   权重 W[v][d] = float(v * dim + d) (每行递增)
//   token_ids = {0, 3, 7} → output 应等于 W[0], W[3], W[7]
// ===========================================================================
class EmbeddingLayerTest : public ::testing::Test {
   protected:
    static constexpr int32_t kVocabSize = 16;
    static constexpr int32_t kDim = 8;
    static constexpr int32_t kSeqLen = 3;

    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // 构建权重表: W[v][d] = v * kDim + d
        weight_data_.resize(kVocabSize * kDim);
        for (int32_t v = 0; v < kVocabSize; ++v) {
            for (int32_t d = 0; d < kDim; ++d) {
                weight_data_[v * kDim + d] = float(v * kDim + d);
            }
        }

        token_ids_ = {0, 3, 7};  // kSeqLen=3 个 token
    }

    // 创建 int32 的 CPU token_ids tensor
    tensor::Tensor make_token_ids_tensor() {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor t(base::DataType::kDataTypeInt32, kSeqLen, true, alloc);
        auto* p = t.ptr<int32_t>();
        for (int32_t i = 0; i < kSeqLen; ++i) p[i] = token_ids_[i];
        return t;
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
    std::vector<float> weight_data_;
    std::vector<int32_t> token_ids_;
};

// ---------------------------------------------------------------------------
// 元信息
TEST_F(EmbeddingLayerTest, Metadata) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    EXPECT_EQ(layer.layer_type(), op::LayerType::kLayerEmbedding);
    EXPECT_EQ(layer.input_size(), 1u);
    EXPECT_EQ(layer.output_size(), 1u);
    EXPECT_EQ(layer.weight_size(), 1u);
}

// ---------------------------------------------------------------------------
// check() 错误路径: input 为空
TEST_F(EmbeddingLayerTest, CheckFailsOnEmptyInput) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    layer.set_weight(0, {kVocabSize, kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);
    // 不设置 input
    layer.set_output(0, make_cpu_tensor_2d(kSeqLen, kDim, 0.f));
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// check() 错误路径: 输出维度不匹配 (output token 数量错误)
TEST_F(EmbeddingLayerTest, CheckFailsOnOutputDimMismatch) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    layer.set_weight(0, {kVocabSize, kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);
    layer.set_input(0, make_token_ids_tensor());
    layer.set_output(0, make_cpu_tensor_2d(kSeqLen + 1, kDim, 0.f));  // token 数量不对
    EXPECT_FALSE(layer.check());
}

// ---------------------------------------------------------------------------
// set_weight 从裸指针加载
TEST_F(EmbeddingLayerTest, SetWeightFromRawPtr) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    layer.set_weight(0, {kVocabSize, kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    const tensor::Tensor& w = layer.get_weight(0);
    EXPECT_FALSE(w.is_empty());
    EXPECT_EQ(w.get_dim(0), kVocabSize);
    EXPECT_EQ(w.get_dim(1), kDim);
    // 验证第 0 行
    for (int32_t d = 0; d < kDim; ++d) {
        EXPECT_FLOAT_EQ(w.ptr<float>()[d], float(d)) << "W[0][" << d << "]";
    }
    // 验证第 7 行
    int32_t row7_off = 7 * kDim;
    for (int32_t d = 0; d < kDim; ++d) {
        EXPECT_FLOAT_EQ(w.ptr<float>()[row7_off + d], float(7 * kDim + d));
    }
}

// ---------------------------------------------------------------------------
// forward() CPU: token_ids={0,3,7} → output rows = W[0], W[3], W[7]
TEST_F(EmbeddingLayerTest, ForwardCPU) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    layer.set_weight(0, {kVocabSize, kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    tensor::Tensor ids = make_token_ids_tensor();
    tensor::Tensor output = make_cpu_tensor_2d(kSeqLen, kDim, 0.f);
    layer.set_input(0, ids);
    layer.set_output(0, output);

    ASSERT_TRUE(layer.forward());

    float* p = output.ptr<float>();
    for (int32_t i = 0; i < kSeqLen; ++i) {
        int32_t token_id = token_ids_[i];
        for (int32_t d = 0; d < kDim; ++d) {
            float expected = float(token_id * kDim + d);
            EXPECT_FLOAT_EQ(p[i * kDim + d], expected)
                << "token[" << i << "]=" << token_id << " dim=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// forward() CUDA: 同上，验证 D2H 结果
TEST_F(EmbeddingLayerTest, ForwardCUDA) {
    op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, kDim, kSeqLen, kVocabSize);
    layer.set_weight(0, {kVocabSize, kDim}, weight_data_.data(), base::DeviceType::kDeviceCPU);

    layer.set_device_type(base::DeviceType::kDeviceCUDA);
    layer.to_cuda();
    layer.set_cuda_config(make_cuda_config());

    // token_ids 上传到 CUDA (int32)
    tensor::Tensor h_ids = make_token_ids_tensor();
    auto gpu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor d_ids(base::DataType::kDataTypeInt32, kSeqLen, true, gpu_alloc);
    cudaMemcpy(d_ids.ptr<void>(), h_ids.ptr<void>(), kSeqLen * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    tensor::Tensor d_output = make_cuda_tensor_2d(kSeqLen, kDim);
    layer.set_input(0, d_ids);
    layer.set_output(0, d_output);

    ASSERT_TRUE(layer.forward());
    cudaDeviceSynchronize();

    auto result = d2h(d_output);
    for (int32_t i = 0; i < kSeqLen; ++i) {
        int32_t token_id = token_ids_[i];
        for (int32_t d = 0; d < kDim; ++d) {
            float expected = float(token_id * kDim + d);
            EXPECT_FLOAT_EQ(result[i * kDim + d], expected)
                << "token[" << i << "]=" << token_id << " dim=" << d;
        }
    }
}
