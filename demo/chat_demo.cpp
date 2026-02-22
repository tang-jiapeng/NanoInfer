/**
 * @file chat_demo.cpp
 * @brief 交互式多轮对话 Demo — 支持 LLaMA 3.2 1B Instruct
 *
 * 使用 LLaMA 3 Chat Template 构造多轮对话上下文，每轮重建完整 prompt 提交给 Engine。
 * 启用 Prefix Caching 可复用前几轮的 KV Cache 前缀，降低重复 prefill 开销。
 *
 * 用法:
 *   ./chat_demo --model llama3                     (默认 fp32)
 *   ./chat_demo --model llama3 --dtype int8        (INT8 量化)
 *   ./chat_demo --model llama2                     (LLaMA2, 无 chat template)
 */
#include <glog/logging.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/llama.h"
#include "nanoinfer/sampler/sampling_params.h"

// ----------------------------------------------------------------------------------
// Engine 配置
// ----------------------------------------------------------------------------------
const int32_t MAX_BATCH_SIZE = 1;
const int32_t MAX_SEQUENCES = 4;
const int32_t PREFILL_CHUNK_SIZE = 512;
const int32_t BLOCK_SIZE = 16;
const int32_t NUM_CACHE_BLOCKS = 2048;

// 生成参数
const int32_t MAX_NEW_TOKENS = 512;

// ----------------------------------------------------------------------------------
// 模型预设
// ----------------------------------------------------------------------------------
struct ModelPreset {
    std::string model_path;
    std::string token_path;
    base::ModelType model_type;
    base::TokenizerType tokenizer_type;
    bool is_quant;
};

static ModelPreset get_preset(const std::string& name, bool is_quant) {
    if (name == "llama3") {
        std::string bin =
            is_quant ? "./models/llama3_instruct/llama3_instruct_int8.bin" : "./models/llama3_instruct/llama3_instruct_fp32.bin";
        return {bin, "./models/llama3_instruct/tokenizer.json", base::ModelType::kModelTypeLLaMA3,
                base::TokenizerType::kEncodeBpe, is_quant};
    }
    std::string bin =
        is_quant ? "./models/llama2/llama2_int8.bin" : "./models/llama2/llama2_fp32.bin";
    return {bin, "./models/llama2/tokenizer.model", base::ModelType::kModelTypeLLaMA2,
            base::TokenizerType::kEncodeSpe, is_quant};
}

// ----------------------------------------------------------------------------------
// 对话历史管理
// ----------------------------------------------------------------------------------
struct ChatMessage {
    std::string role;  // "system", "user", "assistant"
    std::string content;
};

/**
 * @brief 构建 LLaMA 3 Chat Template (不含 <|begin_of_text|>，由 tokenizer 自动添加 BOS)
 *
 * 格式:
 *   <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
 *   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
 *   <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
 *   ...
 *   <|start_header_id|>assistant<|end_header_id|>\n\n
 */
static std::string build_llama3_prompt(const std::vector<ChatMessage>& messages) {
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
            << msg.content << "<|eot_id|>";
    }
    // 以 assistant 头结尾，引导模型生成回复
    oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return oss.str();
}

/**
 * @brief 构建 LLaMA 2 简单对话 prompt (无特殊 chat template)
 */
static std::string build_llama2_prompt(const std::vector<ChatMessage>& messages) {
    std::ostringstream oss;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            oss << msg.content << "\n\n";
        } else if (msg.role == "user") {
            oss << "User: " << msg.content << "\n";
        } else {
            oss << "Assistant: " << msg.content << "\n";
        }
    }
    oss << "Assistant:";
    return oss.str();
}

// ----------------------------------------------------------------------------------
// 辅助函数
// ----------------------------------------------------------------------------------
static void print_separator(int width = 70) {
    std::cout << std::string(width, '=') << std::endl;
}

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// ----------------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------------
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 0;
    FLAGS_minloglevel = 1;  // 抑制 Engine/Scheduler 初始化日志

    // ------------------------------------------------------------------
    // 解析参数
    // ------------------------------------------------------------------
    std::string model_name = "llama3";
    bool is_quant = false;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--model") model_name = argv[i + 1];
        if (std::string(argv[i]) == "--dtype" && std::string(argv[i + 1]) == "int8")
            is_quant = true;
    }
    auto preset = get_preset(model_name, is_quant);
    bool is_llama3 = (preset.model_type == base::ModelType::kModelTypeLLaMA3);

    // ------------------------------------------------------------------
    // 1. 加载模型
    // ------------------------------------------------------------------
    std::cout << "\n";
    print_separator();
    std::cout << "  NanoInfer Chat Demo" << std::endl;
    print_separator();
    std::cout << "  Model: " << model_name << " (" << (is_quant ? "int8" : "fp32") << ")"
              << std::endl;

    std::cout << "  Loading model..." << std::flush;
    FLAGS_minloglevel = 0;
    auto model =
        std::make_unique<model::LLamaModel>(preset.tokenizer_type, preset.model_type,
                                            preset.token_path, preset.model_path, preset.is_quant);
    model->init(base::DeviceType::kDeviceCUDA);
    FLAGS_minloglevel = 1;
    std::cout << " done" << std::endl;
    std::cout << "  Vocab=" << model->config().vocab_size_
              << ", Layers=" << model->config().layer_num_ << ", Dim=" << model->config().dim_
              << std::endl;

    // ------------------------------------------------------------------
    // 2. 创建 Engine (启用 Prefix Caching 以复用多轮前缀)
    // ------------------------------------------------------------------
    engine::EngineConfig engine_config;
    engine_config.max_batch_size = MAX_BATCH_SIZE;
    engine_config.max_sequences = MAX_SEQUENCES;
    engine_config.prefill_chunk_size = PREFILL_CHUNK_SIZE;
    engine_config.block_size = BLOCK_SIZE;
    engine_config.num_cache_blocks = NUM_CACHE_BLOCKS;
    engine_config.enable_prefix_caching = true;

    engine::Engine eng(model.get(), engine_config);
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    auto status = eng.init(allocator);
    if (!status) {
        std::cerr << "Engine init failed: " << status.get_err_msg() << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // 3. 采样参数 (适合 chat 的平衡设置)
    // ------------------------------------------------------------------
    sampler::SamplingParams sp;
    sp.temperature = 0.7f;
    sp.top_k = 40;
    sp.top_p = 0.9f;
    sp.repetition_penalty = 1.1f;
    sp.seed = -1;  // 随机种子

    // ------------------------------------------------------------------
    // 4. 对话循环
    // ------------------------------------------------------------------
    std::vector<ChatMessage> history;

    // 系统提示词
    std::string system_prompt =
        "You are a helpful, concise assistant. Answer questions clearly and briefly.";
    history.push_back({"system", system_prompt});

    std::cout << "\n  Sampling: temp=" << sp.temperature << ", top_k=" << sp.top_k
              << ", top_p=" << sp.top_p << ", rep_penalty=" << sp.repetition_penalty << std::endl;
    std::cout << "  Max tokens per reply: " << MAX_NEW_TOKENS << std::endl;
    std::cout << "  Prefix caching: enabled" << std::endl;
    print_separator();
    std::cout << "  Type your message and press Enter. Type 'quit' or 'exit' to stop.\n"
              << "  Type '/clear' to reset conversation history.\n"
              << std::endl;

    int turn = 0;
    while (true) {
        // 读取用户输入
        std::cout << "You> " << std::flush;
        std::string user_input;
        if (!std::getline(std::cin, user_input)) break;

        user_input = trim(user_input);
        if (user_input.empty()) continue;
        if (user_input == "quit" || user_input == "exit") break;
        if (user_input == "/clear") {
            history.clear();
            history.push_back({"system", system_prompt});
            turn = 0;
            std::cout << "  [conversation cleared]\n" << std::endl;
            continue;
        }

        // 追加用户消息
        history.push_back({"user", user_input});

        // 构建完整 prompt
        std::string prompt =
            is_llama3 ? build_llama3_prompt(history) : build_llama2_prompt(history);

        // 提交请求
        auto t_start = std::chrono::high_resolution_clock::now();
        int64_t rid = eng.add_request(prompt, MAX_NEW_TOKENS, sp);
        if (rid < 0) {
            std::cerr << "  [error: failed to add request]\n";
            history.pop_back();
            continue;
        }

        // 运行推理
        std::cout << "\nAssistant> " << std::flush;
        status = eng.run();
        auto t_end = std::chrono::high_resolution_clock::now();

        if (!status) {
            std::cerr << "\n  [error: " << status.get_err_msg() << "]\n";
            history.pop_back();
            continue;
        }

        // 获取结果
        std::string reply = eng.get_request_result(rid);
        auto req = eng.get_request(rid);

        // 打印回复
        std::cout << reply << std::endl;

        // 追加 assistant 回复到历史
        history.push_back({"assistant", reply});
        ++turn;

        // 统计信息 (一行)
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        int32_t prompt_len = req ? req->prompt_len() : 0;
        int32_t gen_len = req ? req->generated_len() : 0;
        double tok_per_sec = (gen_len > 0 && total_ms > 0) ? (gen_len * 1000.0 / total_ms) : 0;
        double exec_ms = req ? req->execution_time_seconds() * 1000.0 : 0;

        std::cout << std::fixed << std::setprecision(1) << "  [turn " << turn
                  << " | prompt=" << prompt_len << " gen=" << gen_len << " tokens"
                  << " | " << exec_ms << "ms"
                  << " | " << tok_per_sec << " tok/s]\n"
                  << std::endl;
    }

    std::cout << "\n  Bye!\n" << std::endl;
    return 0;
}
