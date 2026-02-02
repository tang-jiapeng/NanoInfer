#include <glog/logging.h>
#include <gtest/gtest.h>
#include "nanoinfer/base/buffer.h"
#include "nanoinfer/model/llama.h"
#include "nanoinfer/tensor/tensor.h"

// TEST(test_llama_model, cpu1) {
//     using namespace base;
//     std::shared_ptr<base::CPUDeviceAllocator> alloc =
//         std::make_shared<base::CPUDeviceAllocator>();

//     const char* checkpoint_path = "/home/tang/NanoInfer/tools/llama2_fp32.bin";
//     const char* tokenizer_path =
//         "/home/tang/NanoInfer/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model";
//     model::LLamaModel model(base::TokenizerType::kEncodeSpe, tokenizer_path,
//                             checkpoint_path, false);
//     auto status = model.init(base::DeviceType::kDeviceCPU);

//     if (status) {
//         std::string sentence = "Hi";
//         // 2. 获取 Tokens
//         auto tokens = model.encode(sentence);
//         ASSERT_FALSE(tokens.empty());

//         // 3. 准备 pos_tensor (参照 generate 中的 model.get_buffer 逻辑)
//         tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
//         pos_tensor.index<int32_t>(0) = 0;  // 设置初始位置为 0

//         // 4. 准备 input tensor (参照 generate 中的 embedding + fill_input 逻辑)
//         bool is_prompt = true;
//         const auto& prompt_embedding = model.embedding(tokens);
//         tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);

//         // 5. 执行推理 (调用更新后的 forward)
//         int next_token = -1;
//         auto forward_status = model.forward(input, pos_tensor, next_token);
//         ASSERT_TRUE(forward_status);

//         // 6. 验证 Logits 结果
//         const float* logits =
//             model.get_buffer(model::ModelBufferType::kForwardOutput).ptr<float>();
//         ASSERT_NEAR(logits[0], -12.7976265, 1e-3f);
//         ASSERT_NEAR(logits[32], -9.97821331, 1e-3f);
//         ASSERT_NEAR(logits[128], -12.8054199, 1e-3f);
//         ASSERT_NEAR(logits[256], -12.7876959, 1e-3f);
//         ASSERT_NEAR(logits[512], 4.75685883, 1e-3f);
//         ASSERT_NEAR(logits[613], -3.83690214, 1e-3f);
//         ASSERT_NEAR(logits[1011], -3.34461427, 1e-3f);
//         ASSERT_NEAR(logits[1022], -7.45470142, 1e-3f);
//         ASSERT_NEAR(logits[1023], -1.00463259, 1e-3f);
//     }
// }