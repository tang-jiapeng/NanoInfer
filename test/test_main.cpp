#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// 测试模块表
//   key  — 命令行传入的模块名
//   value — GTest filter 中使用的 Test Suite 名称前缀
//
// 用法示例:
//   ./test_llm                         运行全部测试
//   ./test_llm --module=tensor         只运行 Tensor 相关测试
//   ./test_llm --module=tensor,buffer  运行 Tensor + Buffer 测试
//   ./test_llm --module=cuda_kernel    运行所有 CUDA Kernel 测试
//   ./test_llm --module=engine         运行 Engine 测试
//   ./test_llm --module=model          运行 Model 加载测试
//   ./test_llm --list-modules          列出所有可用模块名
// ===========================================================================
static const std::unordered_map<std::string, std::string> kModuleFilterMap = {
    // 基础类测试
    {"base", "DataTypeTest.*:StatusTest.*:DeviceTypeTest.*:NoCopyableTest.*"},
    {"alloc", "CPUAllocTest.*:CUDAAllocTest.*"},
    {"buffer", "BufferTest.*"},
    {"tensor", "TensorTest.*"},
    // CUDA Kernel 测试
    {"cuda_kernel",
     "AddKernelTest.*:ArgmaxKernelTest.*:EmbeddingKernelTest.*"
     ":MatmulKernelTest.*:PagedAttentionKernelTest.*"
     ":PagedKVWriteKernelTest.*:RMSNormKernelTest.*"
     ":RoPEKernelTest.*:SwigluKernelTest.*"},
    // Engine 测试
    {"engine",
     "EngineTest.*:BlockManagerTest.*:BlockTableTest.*"
     ":KVCacheManagerTest.*:SchedulerTest.*"
     ":BlockManagerCacheTest.*:KVCachePrefixTest.*"},
    // Layer 测试 (op/)
    {"layer",
     "LayerBaseTest.*:VecAddLayerTest.*:RmsNormLayerTest.*"
     ":MatmulLayerTest.*:SwiGLULayerTest.*:EmbeddingLayerTest.*"
     ":BpeEncodeLayerTest.*"},
    // Model 测试
    {"model", "TinyLlamaTest.*"},
    // 快捷组合
    {"fast", "DataTypeTest.*:StatusTest.*:CPUAllocTest.*:BufferTest.*:TensorTest.*"},
    {"all", "*"},
};

// 解析命令行中的 --module=xxx,yyy 参数
static std::string parse_module_arg(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--module=", 0) == 0) {
            return arg.substr(9);  // 去掉 "--module="
        }
        if (arg == "--list-modules") {
            return "__list__";
        }
    }
    return "";
}

// 将逗号分隔的模块列表转化为 GTest filter 字符串
static std::string build_gtest_filter(const std::string& modules_str) {
    std::vector<std::string> parts;
    std::stringstream ss(modules_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // 去除首尾空格
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (token.empty()) continue;

        auto it = kModuleFilterMap.find(token);
        if (it == kModuleFilterMap.end()) {
            std::cerr << "[WARNING] Unknown module: '" << token
                      << "'. Use --list-modules to see available modules.\n";
            continue;
        }
        parts.push_back(it->second);
    }
    if (parts.empty()) return "";

    std::string filter;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) filter += ":";
        filter += parts[i];
    }
    return filter;
}

int main(int argc, char* argv[]) {
    // 初始化 glog
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_v = 0;

    // 创建日志目录 (cmake build 目录下)
    // 注意: 如果目录不存在, glog 会回退到 stderr
    FLAGS_log_dir = "./log/";

    // 解析自定义参数
    std::string modules_arg = parse_module_arg(argc, argv);

    if (modules_arg == "__list__") {
        std::cout << "Available test modules:\n";
        // 有序输出
        std::vector<std::string> keys;
        for (const auto& kv : kModuleFilterMap) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        for (const auto& k : keys) {
            std::cout << "  --module=" << k << "\n";
        }
        std::cout << "\nExamples:\n"
                  << "  ./test_llm                         # run all tests\n"
                  << "  ./test_llm --module=fast           # base + alloc + buffer + tensor\n"
                  << "  ./test_llm --module=tensor,buffer  # TensorTest + BufferTest\n"
                  << "  ./test_llm --module=engine         # EngineTest + Scheduler + ...\n"
                  << "  ./test_llm --module=cuda_kernel    # all CUDA kernel tests\n";
        return 0;
    }

    // 初始化 GTest (不删除 gtest 自身的 --gtest_* 参数)
    testing::InitGoogleTest(&argc, argv);

    // 如果指定了 --module，构建并设置 filter
    if (!modules_arg.empty()) {
        std::string filter = build_gtest_filter(modules_arg);
        if (!filter.empty()) {
            LOG(INFO) << "Running modules: [" << modules_arg << "]";
            LOG(INFO) << "GTest filter: " << filter;
            testing::GTEST_FLAG(filter) = filter;
        } else {
            LOG(WARNING) << "No valid modules matched '" << modules_arg << "', running all tests.";
        }
    } else {
        // 没有 --module 参数且 GTest 未设置自己的 filter 时，运行全部
        // (TinyLlamaTest 需要真实模型文件，默认跳过)
        if (testing::GTEST_FLAG(filter) == "*") {
            std::string default_filter =
                "DataTypeTest.*:StatusTest.*:DeviceTypeTest.*:NoCopyableTest.*"
                ":CPUAllocTest.*:CUDAAllocTest.*"
                ":BufferTest.*"
                ":TensorTest.*"
                ":AddKernelTest.*:ArgmaxKernelTest.*:EmbeddingKernelTest.*"
                ":MatmulKernelTest.*:PagedAttentionKernelTest.*"
                ":PagedKVWriteKernelTest.*:RMSNormKernelTest.*"
                ":RoPEKernelTest.*:SwigluKernelTest.*"
                ":EngineTest.*:BlockManagerTest.*:BlockTableTest.*"
                ":KVCacheManagerTest.*:SchedulerTest.*"
                ":LayerBaseTest.*:VecAddLayerTest.*:RmsNormLayerTest.*"
                ":MatmulLayerTest.*:SwiGLULayerTest.*:EmbeddingLayerTest.*"
                ":TinyLlamaTest.*";
            LOG(INFO) << "No --module specified. Running default suite.";
            testing::GTEST_FLAG(filter) = default_filter;
        }
    }

    LOG(INFO) << "===== NanoInfer Test Runner Start =====";
    int result = RUN_ALL_TESTS();
    LOG(INFO) << "===== NanoInfer Test Runner End   =====";
    return result;
}
