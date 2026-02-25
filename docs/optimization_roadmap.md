# NanoInfer 优化路线图 v3

> 基于对现有代码的深度分析，以及对 **llama.cpp / vLLM / SGLang / TensorRT-LLM / MLC-LLM** 核心设计的深度参考，
> 本文档涵盖三个主方向的完整可执行方案，并附性能对比测试方法。
>
> 1. **HuggingFace safetensors 原生加载**（含 Int8 分组量化的完整处理方案，对齐 vLLM/SGLang 量化加载范式）
> 2. **零拷贝模型加载深度优化**
> 3. **统一内存池抽象 + CPU / CUDA 特化**（参考 vLLM BlockAllocator / SGLang TokenToKVPool / gpu_memory_utilization 显存预算机制）

---

## 目录

- [框架参考与设计借鉴](#框架参考与设计借鉴)
- [vLLM / SGLang 深度对比：量化与内存管理](#vllm--sglang-深度对比量化与内存管理)
- [背景与现状分析](#背景与现状分析)
- [方向一：safetensors 原生支持（含 Int8 量化）](#方向一safetensors-原生支持含-int8-量化)
- [方向二：零拷贝模型加载深度优化](#方向二零拷贝模型加载深度优化)
- [方向三：统一内存池抽象与 CPU/CUDA 特化](#方向三统一内存池抽象与-cpucuda-特化)
- [改动优先级与实施顺序](#改动优先级与实施顺序)
- [性能对比测试方案](#性能对比测试方案)

---

## 框架参考与设计借鉴

在深化方案设计前，梳理几个主流推理框架在相关方向上的核心思路：

| 框架 | 模型加载 | 内存管理 | 量化方案 |
|------|---------|---------|---------|
| **llama.cpp** | GGUF 自定义格式 + `mmap(MAP_SHARED)` + `madvise`；`llama_model_load` 按名字解析 tensor | `ggml_tallocr`（bump pointer arena）+ `ggml_gallocr`（静态计算图生命周期分析）；权重 buffer 与计算 buffer 严格隔离 | GGUF 内嵌量化类型字段（Q4_K_M / Q8_0 等），`ggml_tensor.type` 携带量化信息，dequant kernel 就地执行 |
| **vLLM** | 直接读 HF safetensors / pickle，不做格式转换；`model_loader.py` 按 HF 名字映射到 nn.Module 参数 | 通过 `BlockAllocator` 统一管理 KV Cache 物理块；推理激活值走 PyTorch 的 CUDA caching allocator（per-stream 空闲列表）；`gpu_memory_utilization` 参数控制显存预算 | 支持 AWQ / GPTQ / FP8 / INT8 W8A8；量化 tensor 和 scale tensor **分开存储**在 safetensors 中，运行时 dequant；FP8 支持在线量化 |
| **SGLang** | 兼容 HF safetensors，复用 vLLM 模型加载器或内置等效实现 | `TokenToKVPool`（flat token-level 内存池）+ `ReqToTokenPool`（请求→token 映射）；RadixAttention 前缀缓存；激活值同样走 PyTorch caching allocator | 复用 vLLM 量化基础设施，支持 AWQ / GPTQ / FP8 / Marlin |
| **TensorRT-LLM** | 统一序列化为 `.engine` 文件，离线编译；权重在 build 阶段一次性量化打包进引擎 | TensorRT 内部管理 workspace；Activation buffer 走 `IPluginV2Layer` 自定义内存策略 | FP8 / INT4 / INT8，量化 metadata（scale/zero_point）写入 tensor weight 并行存储 |
| **MLC-LLM** | NDArray 参数文件 + JSON 元数据，支持 safetensors 作为输入；`convert_weight` 阶段离线压缩 | TVM memory planner（`StorageRewrite`）静态规划激活内存；多设备走 NDArray 内存池 | 支持 AWQ / GPTQ；scale 作为独立 weight 参数按名字加载 |

**对 NanoInfer 的关键启示**：
1. vLLM 和 MLC 均直接消费 safetensors，**不要求格式转换**，但对于带量化信息的模型，**scale tensor 单独存储**而非内嵌在 weight 字节流中。
2. llama.cpp 的 `ggml_tallocr` 和 TVM 的 `StorageRewrite` 思路一致：**提前规划激活内存，推理热路径零分配**。
3. 量化 scale 的存储策略会直接决定 safetensors 支持方案的复杂度。
4. **vLLM 和 SGLang 均以离线量化（GPTQ/AWQ/FP8 checkpoint）为生产主线**，在线量化仅限 FP8 等简单类型。
5. SGLang 的 `TokenToKVPool` 展示了一种比 block-based 更简洁的 KV cache 管理模式，值得作为 NanoInfer 的备选架构方向。

---

## vLLM / SGLang 深度对比：量化与内存管理

> 本节是 v3 新增的核心章节，深入分析 vLLM 和 SGLang 在量化策略和内存池设计上的具体实现，
> 为 NanoInfer 的优化方案提供第一手参考依据。

### 量化策略：离线为主，在线为辅

#### vLLM 的量化架构

vLLM 的量化体系围绕 `QuantizationConfig` 注册机制构建：

```
用户指定 --quantization gptq/awq/fp8/...
  → QuantizationConfig.from_config()
    → 创建对应的 QuantizeMethodBase 子类
      → 每个 Linear 层调用 quant_method.create_weights() 替换标准参数
      → model_loader 读 safetensors 时，调用 weight_loader() 将原始张量转换为量化格式
```

**关键设计**：每种量化方法定义自己的 `weight_loader()` 方法，知道 safetensors 中的命名约定：

| 量化方法 | safetensors 中的 tensor 名 | 在线/离线 | 说明 |
|---------|---------------------------|---------|------|
| **GPTQ** | `qweight`(int32 packed), `scales`(fp16), `qzeros`(int32), `g_idx` | **离线** | 预量化模型，4-bit 权重 packed into int32 |
| **AWQ** | `qweight`(int32), `scales`(fp16), `qzeros`(int32) | **离线** | 与 GPTQ 类似但不同量化算法 |
| **FP8 (E4M3)** | `weight`(fp8_e4m3fn), `weight_scale`(fp32) | **离线+在线** | 唯一支持在线量化的格式 |
| **INT8 W8A8** | `weight`(int8), `weight_scale`(fp32), `input_scale`(fp32) | **离线** | 权重+激活值均预校准 |
| **Marlin** | 专用 packed 格式 | **离线** | GPTQ/AWQ 的优化 kernel 格式 |

**FP8 在线量化的特殊性**：
```python
# vLLM FP8StaticLinearMethod.process_weights_after_loading()
# 唯一的在线量化路径 —— 因为 FP8 量化极其简单
def process_weights_after_loading(self, layer):
    weight = layer.weight                        # 从 safetensors 加载的 FP16 权重
    scale = weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
    layer.weight = (weight / scale).to(torch.float8_e4m3fn)
    layer.weight_scale = scale.float()
```

**结论**：vLLM 对 GPTQ/AWQ/INT8 等复杂量化格式**完全不做在线量化**，只加载预量化好的 checkpoint。在线量化仅限于 FP8（因为它只是一个简单的 cast + 定标操作）。

#### SGLang 的量化策略

SGLang 在量化方面大量**复用 vLLM 的基础设施**：

- 早期版本直接 `from vllm.model_executor.layers.quantization import ...`
- 后续版本将核心量化逻辑移植到 SGLang 内部，但保持相同接口
- 支持的量化方法与 vLLM 高度一致：GPTQ, AWQ, FP8, Marlin, BitsAndBytes

**SGLang 的差异化**在于其 `RadixAttention` + flat token pool 架构，而非量化层面。

#### 对 NanoInfer 的量化策略决策

基于 vLLM/SGLang 的实践，NanoInfer 应遵循以下原则：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    NanoInfer 量化策略决策矩阵                             │
├──────────────┬──────────────────────┬────────────────────────────────────┤
│ 量化类型      │ 加载方式              │ 对齐的开源方案                     │
├──────────────┼──────────────────────┼────────────────────────────────────┤
│ Q8_0 (自有)   │ Route A: 离线量化     │ 类似 vLLM INT8 W8A16:             │
│              │ safetensors 分存      │ weight(int8) + weight_scale(fp32) │
│              │ weight + scale       │ 以独立 tensor 名存入 safetensors   │
├──────────────┼──────────────────────┼────────────────────────────────────┤
│ Q8_0 (自有)   │ Route B: 在线量化     │ 类似 vLLM FP8 online:             │
│              │ 从 FP16 初始化时量化  │ 加载 FP16 → cast + scale → int8   │
│              │                      │ 仅作为"免转换"体验入口             │
├──────────────┼──────────────────────┼────────────────────────────────────┤
│ GPTQ/AWQ     │ 离线加载 (未来扩展)   │ 完全对齐 vLLM:                     │
│ (可选扩展)    │ 读预量化 checkpoint  │ qweight+scales+zeros 三张量加载    │
│              │                      │ 需实现 dequant kernel              │
├──────────────┼──────────────────────┼────────────────────────────────────┤
│ FP8 (可选)    │ 离线 + 在线均可       │ 完全对齐 vLLM:                     │
│              │                      │ 需 SM89+ (Ada Lovelace)            │
└──────────────┴──────────────────────┴────────────────────────────────────┘
```

**优先级**：Q8_0 Route A > Q8_0 Route B > GPTQ/AWQ 兼容 > FP8 支持

**扩展性设计**：参考 vLLM 的 `QuantizationConfig` 注册模式，为 NanoInfer 引入类似抽象：

```cpp
// include/nanoinfer/model/quant_config.h（未来扩展用）
namespace model {

enum class QuantMethod { kNone, kQ8_0, kGPTQ, kAWQ, kFP8 };

/// @brief 量化配置抽象，定义权重加载和 dequant 行为
class QuantConfig {
public:
    virtual ~QuantConfig() = default;
    virtual QuantMethod method() const = 0;
    /// @brief 从 safetensors 中获取某个权重需要加载哪些 tensor
    ///        Q8_0: {"weight", "weight_scale"}
    ///        GPTQ: {"qweight", "scales", "qzeros"}
    virtual std::vector<std::string> required_tensor_suffixes() const = 0;
    /// @brief 将加载的原始 tensor 转换为 MatmulLayer 所需格式
    virtual base::Status process_weight(
        const std::string& name,
        const std::unordered_map<std::string, TensorView>& tensors,
        op::MatmulLayer* layer) const = 0;
};

class Q8_0QuantConfig : public QuantConfig {
public:
    QuantMethod method() const override { return QuantMethod::kQ8_0; }
    std::vector<std::string> required_tensor_suffixes() const override {
        return {"", "_scale"};  // weight + weight_scale
    }
    base::Status process_weight(...) const override {
        // 从 tensors["xxx.weight"] 获取 int8 数据
        // 从 tensors["xxx.weight_scale"] 获取 fp32 scale
        // 调用 layer->set_weight_with_scale(...)
    }
};

} // namespace model
```

### 内存管理：vLLM 与 SGLang 的两种范式

#### vLLM 的 "两池分治" 架构

vLLM 将 GPU 显存明确分为两大独立管理区域：

```
┌─────────────────────── GPU 显存 ──────────────────────────┐
│                                                           │
│  ┌──────────────────┐  ┌────────────────────────────────┐ │
│  │  Model Weights   │  │  KV Cache Pool                 │ │
│  │  (固定，加载后    │  │  (BlockAllocator 管理)          │ │
│  │   不释放)         │  │  ┌─────┬─────┬─────┬────────┐ │ │
│  │                  │  │  │Blk 0│Blk 1│Blk 2│  ...   │ │ │
│  │                  │  │  │16tok│16tok│16tok│        │ │ │
│  │                  │  │  └─────┴─────┴─────┴────────┘ │ │
│  └──────────────────┘  └────────────────────────────────┘ │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Activation / Temporary Memory                       │ │
│  │  (PyTorch CUDA Caching Allocator 自动管理)            │ │
│  │  bin-based: 512B → 20MB size classes, stream-ordered │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

**`gpu_memory_utilization` 显存预算机制**（vLLM 核心创新之一）：

```python
# vLLM 的显存预算计算（简化版）
total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
usable_memory = total_gpu_memory * gpu_memory_utilization  # 默认 0.9

# Step 1: 先做一次空跑 profiling 估算非 KV cache 显存
with torch.no_grad():
    model.forward(dummy_input)                      # 触发 PyTorch caching alloc
model_and_activation_memory = torch.cuda.memory_allocated()

# Step 2: 剩余显存全部分配给 KV Cache
kv_cache_memory = usable_memory - model_and_activation_memory
num_blocks = kv_cache_memory // per_block_size      # per_block_size = block_size × layers × 2 × heads × dim × dtype
# 一次性预分配所有 KV Cache 块
```

**NanoInfer 已有的对齐度**：
- `BlockManager` ≈ vLLM `BlockAllocator` （stack-based free list + LRU eviction）✓
- `KVCacheManager` ≈ vLLM `CacheEngine`（per-layer key/value cache tensors）✓
- `BlockTable` ≈ vLLM `BlockTable`（seq→block mapping + GPU tensor export）✓
- Prefix Caching（hash-based + ref counting + eviction）✓

**NanoInfer 尚缺的关键能力**：
1. ❌ gpu_memory_utilization 预算计算（当前 `num_blocks` 由外部硬编码传入）
2. ❌ 激活值内存管理（当前每次 forward 都走 allocator，无预分配）
3. ❌ Memory profiling（无法自动估算模型+激活占用）

#### SGLang 的 "扁平 Token 池" 架构

SGLang 提出了一种与 vLLM 截然不同的 KV Cache 管理范式：

```
┌──── SGLang TokenToKVPool ────────────────────────────────┐
│                                                           │
│  pre-allocated flat tensor:                               │
│  key_buffer:   [max_total_tokens, num_layers, num_heads, head_dim]  │
│  value_buffer: [max_total_tokens, num_layers, num_heads, head_dim]  │
│                                                           │
│  free_slots = Stack([0, 1, 2, ..., max_total_tokens-1])   │
│                                                           │
│  分配 token slot: slot_id = free_slots.pop()  → O(1)      │
│  释放 token slot: free_slots.push(slot_id)    → O(1)      │
│                                                           │
│  KV 访问: key_buffer[slot_id, layer, head, :] → 直接索引   │
│           无需 block_table 间接寻址！                       │
└───────────────────────────────────────────────────────────┘

┌──── SGLang ReqToTokenPool ───────────────────────────────┐
│  token_to_kv_pool_idx: [max_requests, max_seq_len]        │
│                                                           │
│  请求 req_id 的第 pos 个 token 的物理 slot:                │
│  slot = token_to_kv_pool_idx[req_id][pos]                 │
└───────────────────────────────────────────────────────────┘
```

**Block-based (vLLM/NanoInfer) vs Token-level (SGLang) 对比**：

| 维度 | vLLM Block-based | SGLang Token-level | NanoInfer 现状 |
|------|-----------------|-------------------|---------------|
| 分配粒度 | block（16 tokens） | 单个 token | block（可配） |
| KV 访问 | `cache[block_id][offset_in_block]` 两级间接 | `cache[slot_id]` 一级直接 | 同 vLLM |
| 内部碎片 | 有（最后一个 block 可能半空） | 无 | 有 |
| 前缀缓存 | block-level hash match（粗粒度） | token-level radix tree（细粒度） | block-level hash |
| PagedAttention kernel | 需要 block_table 参数 | 不需要（直接索引） | 需要 |
| 实现复杂度 | 中 | 低 | 中 |
| 适合场景 | 通用，大规模部署 | 长序列 + 密集前缀共享 | 通用 |

**SGLang RadixAttention 前缀缓存**（vs NanoInfer 的 hash-based）：

```
SGLang Radix Tree 示例:
          root
         / | \
       [The] [A] [In]
       /  \      |
    [cat] [dog] [the]
    ...    ...   ...

优势：
  - 共享前缀自动匹配到最长公共前缀，无需预计算 hash
  - LRU 驱逐在叶节点级别，更细粒度
  - 长前缀共享效率更高（不受 block_size 整除约束）

NanoInfer 的 hash-based:
  - 每个 block 独立计算 hash → hash_to_block_ 映射
  - 前缀匹配只到 block 粒度
  - 实现简单，性能足够好
```

#### vLLM 激活值内存管理的启示

vLLM 依赖 PyTorch 的 CUDA caching allocator 管理激活值，该 allocator 的核心设计：

```
PyTorch CUDA Caching Allocator:
  ├── Small Pool (< 1MB): 按 512B 的 2^n 分桶
  │     512B → 1KB → 2KB → ... → 512KB
  │     每个桶维护 free_blocks 链表
  │     分配: O(1) 查找匹配桶
  │
  ├── Large Pool (≥ 1MB): 按 2MB 的 2^n 分桶
  │     2MB → 4MB → 8MB → ... → 1GB
  │     分配: best-fit 扫描
  │
  ├── Stream-ordered: 每个 stream 独立跟踪
  │     同一 stream 内释放的内存立即可被后续分配复用
  │     跨 stream 需要事件同步
  │
  └── 自动 coalesce: 相邻空闲块自动合并
```

**为什么 NanoInfer 不能直接照搬 PyTorch allocator？**

NanoInfer 是纯 C++ 实现，不依赖 PyTorch 运行时。因此需要**自建激活值内存管理策略**。
两个可选方向：

1. **Arena 方式（推荐，方向三已规划的 CPUArenaPool / CUDAStreamPool）**
   - 更简洁，利用推理场景的特点：所有临时 tensor 生命周期 = 单次 forward
   - O(1) reset，O(1) 分配，零碎片
   - 完全对齐 llama.cpp 的 `ggml_tallocr` 思路

2. **Size-class 方式（PyTorch allocator 的简化版）**
   - 更通用，但对推理场景有过多功能（跨 stream 跟踪等）
   - 适合需要灵活生命周期管理的场景

#### 综合对 NanoInfer 的内存管理建议

```
┌─────────────────────────────────────────────────────────────────────────┐
│              NanoInfer 内存管理演进路线（参考 vLLM + SGLang）              │
├──────────────┬──────────────────────────────────────────────────────────┤
│ Phase 1      │ ① 引入 gpu_memory_utilization 显存预算机制               │
│ (立即可做)    │    Engine::init() 中自动计算 num_blocks                  │
│              │ ② CPUArenaPool 替代 forward_batched 中的逐次分配          │
│              │ ③ CUDAStreamPool (cudaMallocAsync) 管理 GPU 临时内存     │
├──────────────┼──────────────────────────────────────────────────────────┤
│ Phase 2      │ ④ 实现 SGLang 风格的 TokenToKVPool 作为可选后端           │
│ (可选演进)    │    用于长序列 + 密集前缀共享场景                          │
│              │ ⑤ RadixAttention tree 替代 hash_to_block_ 前缀缓存      │
│              │ ⑥ Memory profiling 自动估算模型+激活显存占用              │
├──────────────┼──────────────────────────────────────────────────────────┤
│ 保持现有      │ BlockManager / KVCacheManager / BlockTable               │
│ (已对齐 vLLM) │ 基本架构与 vLLM 一致，无需重构                           │
└──────────────┴──────────────────────────────────────────────────────────┘
```

---

## 背景与现状分析

### 当前模型加载流程

```
gen_model_from_file()
  └─ read_model_file()
       ├─ open(path, O_RDONLY)        → fd (用于 mmap)
       ├─ fopen(path, "rb")           → FILE* (用于 fread 头部，冗余双重打开！)
       ├─ fread(&ModelConfig, ...)    → 读取自定义二进制头部 (7 × int32)
       ├─ fread(&group_size_, ...)    → 量化模型额外读取 group_size
       └─ mmap(MAP_PRIVATE|PROT_READ) → 映射整个文件，weight_data = data + header_size
```

**当前自定义 bin 格式**（llama2.c 风格）：

FP32 模型：
```
[ ModelConfig (7×i32 = 28B) ][ Embedding(fp32) | AttnNorm | Wq | Wk | Wv | Wo | FFNNorm | W1 | W2 | W3 | FinalNorm | cos/sin | Cls ]
```

Int8 量化模型（`create_param_quant_layers` 中的 pos 偏移逻辑）：
```
[ ModelConfig(28B) ][ group_size(4B) ]
    [ Wq_int8 | Wq_scale(fp32) ] × L   → Wk → Wv → Wo → W1 → W2 → W3
    → Cls_int8/fp32 → Embedding(fp32) → RMSNorm(fp32) × (2L+1)
```
其中每个量化权重块的字节数 = `out_dim × in_dim`（int8），紧随其后的 scale 数量 = `get_scale_num()` = `out_dim × in_dim / group_size`。

### 现状问题总结

| 模块 | 问题 | 影响 |
|------|------|------|
| 格式支持 | 只支持自定义 bin，须用 tools/ 脚本转换 | 使用门槛高，无法直接用 HF 模型 |
| Int8 量化 | scale 与 int8 数据内嵌在同一字节流 | safetensors 原生不支持此布局，需要额外处理 |
| mmap 策略 | `MAP_PRIVATE`，无 `madvise` 提示 | 多进程不共享页，缺少预读/随机访问提示 |
| `read_model_file` | 同时打开 `fd` 和 `FILE*` | 多余一个 file descriptor |
| CPU 分配器 | 每次 `posix_memalign`，无缓存池 | 高频小分配存在系统调用开销 |
| CUDA 分配器 | 小块池线性扫描 O(n)，无 mutex | 多线程不安全，大量缓存后查找退化 |
| CUDA GC | 阈值硬编码 1 GB，大块容差固定 1 MB | 不灵活，大模型场景浪费显存 |
| 推理临时内存 | `forward_batched` 每次调用均通过分配器重新申请所有临时 tensor | 热路径存在不必要的 malloc/free 开销 |

---

## 方向一：safetensors 原生支持（含 Int8 量化）

### 1.1 关键问题：Int8 分组量化怎么办？

这是本方向最核心的设计决策。先厘清问题再给出方案。

#### 现有量化布局（自定义 bin 格式）

```
per_weight_block = [ int8_data (M×N bytes) | fp32_scale (M*N/group_size × 4 bytes) ]
```

int8 数据和 scale 在文件中**紧密内嵌**，通过 `pos` 指针顺序读取（见 `create_param_quant_layers`）。

#### HuggingFace 原生量化现状

标准 HF safetensors 存储的通常是 **FP16/BF16 原始权重**，不含量化信息。若要用量化，HF 生态有两条路：

- **BitsAndBytes (bitsandbytes)**：运行时动态量化，不修改 safetensors 文件
- **GPTQ / AWQ**：预量化后存储 `qweight (int32)` + `scales (fp16)` + `qzeros (int32)` 等多个 tensor，字段名有标准约定

**结论：NanoInfer 的 Q8_0 分组量化方案与 HF 生态中任何一种都不直接兼容。**

### 1.2 量化路线决策

> **v3 更新**：基于对 vLLM/SGLang 量化架构的深度分析（见上文深度对比章节），
> 验证并强化了以下决策。核心结论：**离线量化 + safetensors 分存是工业界共识**。

提供两条可并行推进的路线，不互斥：

#### 路线 A：safetensors 扩展存储（推荐，与 vLLM 范式完全对齐）

将 Python 导出脚本从写 `.bin` 改为写 `.safetensors`，量化 weight 和 scale 以**约定命名规则**分别存储为两个 tensor。

**与 vLLM 的对齐度**：

| 对比项 | vLLM INT8 W8A16 | NanoInfer Q8_0 Route A |
|-------|-----------------|----------------------|
| 权重 tensor | `model.layers.X.self_attn.q_proj.weight` (int8) | 相同命名 |
| Scale tensor | `model.layers.X.self_attn.q_proj.weight_scale` (fp32) | 相同命名 `_scale` 后缀 |
| 元信息 | config.json: `quantization_config` | `__metadata__`: quantization + group_size |
| 量化时机 | 离线（预量化 checkpoint） | 离线（export 脚本产出） |
| Zero Point | 可选 (asymmetric) | 无 (symmetric Q8_0) |

```
约定命名：
  weight tensor:  "model.layers.{i}.self_attn.q_proj.weight"     dtype=I8,  shape=[out, in]
  scale tensor:   "model.layers.{i}.self_attn.q_proj.weight_scale" dtype=F32, shape=[out*in/group_size]

在 __metadata__ 写入额外信息：
  "quantization": "q8_0"
  "group_size":   "128"
```

**优点**：
- 保留现有高效的 Q8_0 量化内核，不需要改 CUDA 代码
- safetensors 按 tensor 名字访问，无需顺序扫描整个文件
- 支持完整的多 shard 大模型

**工具侧修改**（`tools/export_llama2.py` / `export_llama3.py`）：

```python
# 需要引入 safetensors 库
from safetensors.numpy import save_file   # 或 safetensors.torch

def export_q8_safetensors(model, output_path, group_size=128):
    tensors = {}
    metadata = {
        "quantization": "q8_0",
        "group_size": str(group_size),
        "model_type": "llama",
    }
    # 写入 config JSON
    # ...

    for layer_idx, layer in enumerate(model.layers):
        prefix = f"model.layers.{layer_idx}"

        # Q proj
        int8_data, scale, _ = quantize_q80(layer.attention.wq.weight, group_size)
        tensors[f"{prefix}.self_attn.q_proj.weight"]       = int8_data.numpy()
        tensors[f"{prefix}.self_attn.q_proj.weight_scale"] = scale.numpy()

        # K, V, O, W1, W2, W3 同理...

    # Embedding / Norm 不量化，直接 FP32
    tensors["model.embed_tokens.weight"]   = model.tok_embeddings.weight.float().numpy()
    tensors["model.norm.weight"]            = model.norm.weight.float().numpy()

    save_file(tensors, output_path, metadata=metadata)
```

**C++ 加载侧**，加载后按约定名字分别获取 int8 weight 和 fp32 scale：

```cpp
TensorView weight_view, scale_view;
loader_->get_tensor(name + ".weight", weight_view);
loader_->get_tensor(name + ".weight_scale", scale_view);
// 传入 MatmulLayer（现有接口完全兼容，set_weight 已支持 int8）
wq->set_weight_with_scale(weight_view.data, scale_view.data, cpu_device_type);
```

#### 路线 B：加载标准 HF FP16 权重，运行时 Q8_0 量化（对齐 vLLM FP8 在线量化范式）

直接加载 HF 原生 FP16 safetensors，在 C++ 侧初始化时将 FP16 → Q8_0。
这与 **vLLM 的 FP8 在线量化**是完全相同的模式：加载高精度权重 → 初始化时一次性量化 → 推理阶段使用低精度。

```
HF safetensors (FP16)
  → SafetensorsLoader::get_tensor()   → TensorView (fp16 mmap 指针)
  → quantize_fp16_to_q8_0_cpu()      → int8 weight + fp32 scale (堆内存)
  → MatmulLayer::set_weight(...)
```

**优点**：完全无需修改 Python 工具，直接使用 HF 原始模型（vLLM 也为 FP8 提供相同体验）  
**缺点**：初始化时有一次额外的量化计算（通常几秒，可接受）；初始化后 int8 数据占用独立内存（不再 mmap 直接访问）

**建议**：路线 A 是首选生产方案（对齐 vLLM GPTQ/AWQ/INT8 的离线量化模式），路线 B 可作为"免转换快速体验"入口（对齐 vLLM FP8 的在线量化模式）。

#### 路线 C（未来扩展）：兼容 vLLM GPTQ/AWQ 预量化模型

> 此路线为中长期扩展目标，当 NanoInfer 需要直接使用 HuggingFace 上大量已有的
> GPTQ/AWQ 预量化模型时实施。

```
HF GPTQ safetensors (qweight:int32 + scales:fp16 + qzeros:int32)
  → SafetensorsLoader 读取 3 个 tensor
  → GPTQQuantConfig::process_weight()
    → 解包 int32 → int4 权重
    → 构建 dequant lookup table 或直接传入 Marlin kernel
  → MatmulLayer::set_gptq_weight(qweight, scales, zeros)
```

**实施前提**：需先实现 W4A16 dequant kernel（可移植 vLLM 的 Marlin kernel）。

### 1.3 safetensors 格式与文件结构

```
┌──────────────┬──────────────────────────────┬───────────────────────────────────┐
│  8 bytes     │  header_size bytes           │  data region                      │
│  LE uint64   │  UTF-8 JSON                  │  所有 tensor 的连续二进制数据       │
│              │  {                           │  (每个 tensor 起始 64 字节对齐)    │
│              │    "name": {                 │                                   │
│              │      "dtype": "I8"/"F32",    │  tensor_A[...bytes...]            │
│              │      "shape": [M, N],        │  tensor_B[...bytes...]            │
│              │      "data_offsets": [s, e]  │  ...                              │
│              │    }                         │                                   │
│              │  }                           │                                   │
└──────────────┴──────────────────────────────┴───────────────────────────────────┘
```

### 1.4 架构设计：WeightLoader 抽象层

新增 `WeightLoader` 统一抽象，两种格式共用加载接口：

```
include/nanoinfer/model/
├── weight_loader.h           (新增：抽象基类 + TensorView)
├── safetensors_loader.h/.cpp (新增：safetensors 实现)
├── bin_weight_loader.h/.cpp  (新增：封装现有 bin 逻辑)
└── raw_model_data.h          (保留，bin_weight_loader 内部使用)
```

#### `weight_loader.h`

```cpp
namespace model {

struct TensorView {
    const void*          data;      ///< 零拷贝指针（mmap 内部或堆内存）
    size_t               byte_size;
    base::DataType       dtype;
    std::vector<int64_t> shape;
    bool                 is_owned = false;  ///< true 时析构需 free(data)
};

class WeightLoader {
public:
    virtual ~WeightLoader() = default;
    virtual base::Status open(const std::string& path) = 0;
    virtual bool get_tensor(const std::string& name, TensorView& out) const = 0;
    virtual bool has_tensor(const std::string& name) const = 0;
    virtual std::vector<std::string> tensor_names() const = 0;
    /// 量化元信息（仅 safetensors 扩展格式填充）
    virtual int32_t quant_group_size() const { return 0; }
    virtual std::string quant_type() const { return ""; }
};

} // namespace model
```

#### `safetensors_loader.cpp` 核心实现

```cpp
base::Status SafetensorsLoader::load_single(const std::string& path) {
    // 1. 只用一个 fd，pread 读头部，不开 FILE*
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    uint64_t header_size = 0;
    pread(fd, &header_size, 8, 0);

    std::string json_str(header_size, '\0');
    pread(fd, json_str.data(), header_size, 8);

    // 2. mmap 整个文件（MAP_SHARED + PROT_READ）
    struct stat sb; fstat(fd, &sb);
    void* mapped = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    // madvise 提示（整体随机访问，metadata 区域顺序读一次）
    madvise(mapped, 8 + header_size, MADV_SEQUENTIAL);
    madvise((char*)mapped + 8 + header_size,
            sb.st_size - 8 - header_size, MADV_RANDOM);
#ifdef MADV_HUGEPAGE
    madvise(mapped, sb.st_size, MADV_HUGEPAGE);
#endif

    size_t data_off = 8 + header_size;
    shards_.push_back({fd, mapped, (size_t)sb.st_size, data_off});

    // 3. 解析 JSON header；同时识别 __metadata__ 中的量化信息
    auto root = nlohmann::json::parse(json_str);
    if (root.contains("__metadata__")) {
        auto& meta = root["__metadata__"];
        if (meta.contains("group_size"))
            quant_group_size_ = std::stoi(meta["group_size"].get<std::string>());
        if (meta.contains("quantization"))
            quant_type_       = meta["quantization"].get<std::string>();
    }
    for (auto& [name, info] : root.items()) {
        if (name == "__metadata__") continue;
        TensorMeta m;
        m.shard_idx    = (int)shards_.size() - 1;
        m.dtype        = parse_dtype(info["dtype"].get<std::string>());
        m.shape        = info["shape"].get<std::vector<int64_t>>();
        auto offs      = info["data_offsets"].get<std::vector<size_t>>();
        m.offset_start = offs[0];
        m.offset_end   = offs[1];
        meta_map_[name] = m;
    }
    return base::error::Success();
}
```

### 1.5 LLaMA 权重名映射表

HF safetensors 字段名与现有 `create_param_layers` 逻辑的对应关系：

```cpp
// include/nanoinfer/model/llama_weight_names.h
namespace model {
inline std::string llama_weight_name(const char* role, int layer = -1) {
    if (layer < 0) {
        if (!strcmp(role, "embed"))      return "model.embed_tokens.weight";
        if (!strcmp(role, "norm"))       return "model.norm.weight";
        if (!strcmp(role, "lm_head"))    return "lm_head.weight";
    }
    std::string p = "model.layers." + std::to_string(layer);
    if (!strcmp(role, "wq"))   return p + ".self_attn.q_proj.weight";
    if (!strcmp(role, "wk"))   return p + ".self_attn.k_proj.weight";
    if (!strcmp(role, "wv"))   return p + ".self_attn.v_proj.weight";
    if (!strcmp(role, "wo"))   return p + ".self_attn.o_proj.weight";
    if (!strcmp(role, "w1"))   return p + ".mlp.gate_proj.weight";
    if (!strcmp(role, "w2"))   return p + ".mlp.down_proj.weight";
    if (!strcmp(role, "w3"))   return p + ".mlp.up_proj.weight";
    if (!strcmp(role, "attn_norm")) return p + ".input_layernorm.weight";
    if (!strcmp(role, "ffn_norm"))  return p + ".post_attention_layernorm.weight";
    return "";
}
// 量化 scale 命名约定（路线 A）：直接追加 "_scale" 后缀
inline std::string llama_scale_name(const std::string& weight_name) {
    return weight_name + "_scale";
}
} // namespace model
```

### 1.6 Model::read_model_file() 路由逻辑

```cpp
base::Status Model::read_model_file() {
    namespace fs = std::filesystem;
    fs::path p(model_path_);
    std::string ext = p.extension().string();

    if (ext == ".safetensors") {
        weight_loader_ = std::make_unique<SafetensorsLoader>();
        return weight_loader_->open(model_path_);
    }
    if (p.filename() == "model.safetensors.index.json") {
        weight_loader_ = std::make_unique<SafetensorsLoader>();
        return weight_loader_->open(model_path_);  // 内部自动走 sharded 路径
    }
    // 原有 bin 格式，保持不变
    weight_loader_ = std::make_unique<BinWeightLoader>(is_quant_model_);
    return weight_loader_->open(model_path_);
}
```

### 1.7 依赖引入

```cmake
# CMakeLists.txt，通过已有的 CPM 机制添加：
CPMAddPackage("gh:nlohmann/json@3.11.3")
# 或高性能替代（大模型 shard JSON 解析更快）：
CPMAddPackage("gh:simdjson/simdjson@3.9.1")
```

---

## 方向二：零拷贝模型加载深度优化

### 2.1 现有 mmap 五个可改进点

| # | 问题 | 位置 | 修复方法 |
|---|------|------|---------|
| ① | `MAP_PRIVATE` 多进程不共享 page cache | `model.cpp:read_model_file` | → `MAP_SHARED` |
| ② | 同时打开 `fd` + `FILE*` | `model.cpp:read_model_file` | 删除 `FILE*`，用 `pread` 读头部 |
| ③ | 无 `madvise` 访问模式提示 | 同上 | 加载时 `MADV_WILLNEED`，推理时 `MADV_RANDOM` |
| ④ | 无大页提示 | 同上 | Linux 加 `MADV_HUGEPAGE`，减少 TLB miss |
| ⑤ | 无 `posix_fadvise` 文件系统预读 | 同上 | 打开时 `POSIX_FADV_SEQUENTIAL`，mmap 后改 `POSIX_FADV_RANDOM` |

### 2.2 修复后的 read_model_file 骨架

```cpp
base::Status Model::read_legacy_bin_file() {
    int fd = open(model_path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return base::error::PathNotValid(model_path_);

    // ① 文件系统预读提示（顺序加载阶段）
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    // ② pread 读头部，无需 FILE*
    ModelConfig config{};
    if (pread(fd, &config, sizeof(ModelConfig), 0) != sizeof(ModelConfig))
        return base::error::ModelParseError("read header failed");
    if (is_quant_model_) {
        if (pread(fd, &group_size_, sizeof(int32_t), sizeof(ModelConfig)) != sizeof(int32_t))
            return base::error::ModelParseError("read group_size failed");
    }

    struct stat sb; fstat(fd, &sb);
    raw_model_data_->file_size = sb.st_size;
    raw_model_data_->fd = fd;

    // ③ MAP_SHARED（只读多进程共享页）
    raw_model_data_->data = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (raw_model_data_->data == MAP_FAILED)
        return base::error::ModelParseError("mmap failed");

    // ④ 大页提示（减少 TLB miss，对 GB 级权重效果明显）
#ifdef MADV_HUGEPAGE
    madvise(raw_model_data_->data, sb.st_size, MADV_HUGEPAGE);
#endif

    // ⑤ 加载阶段预读（H2D 拷贝前把页都 fault-in）
    madvise(raw_model_data_->data, sb.st_size, MADV_WILLNEED);

    // 权重指针设置
    size_t header_size = sizeof(ModelConfig) + (is_quant_model_ ? sizeof(int32_t) : 0);
    raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + header_size;

    // ⑥ H2D 拷贝完成后切换为随机访问（CPU 推理时按需 page fault）
    // （此行在 init() 中 to_cuda() 完成后调用）
    // madvise(raw_model_data_->data, sb.st_size, MADV_RANDOM);
    // posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);

    return generate_model_infos(config);
}
```

### 2.3 GPU 推理的权重预固定（Pinned Memory）

对于 GPU 推理场景，权重在 `to_cuda()` 阶段执行 H2D 拷贝。若权重页尚未在内存中（冷启动），cudaMemcpy 会触发大量 page fault，显著延长初始化时间。

**方案**：在 H2D 拷贝前用 `cudaHostRegister` 将 mmap 区域注册为 pinned memory，DMA 可直接读取，避免中间 staging buffer：

```cpp
void LLamaModel::init_gpu_weights() {
    // 仅在有 CUDA 时执行；权重区域一次性 pin
    void* weight_start = raw_model_data_->weight_data;
    size_t weight_size  = raw_model_data_->file_size
                          - (size_t)((char*)weight_start - (char*)raw_model_data_->data);

    // cudaHostRegisterReadOnly：通知 CUDA 此内存不会被 CPU 写入，可优化 DMA 路径
    cudaHostRegister(weight_start, weight_size,
                     cudaHostRegisterReadOnly | cudaHostRegisterMapped);

    // 执行 H2D 拷贝（现有的 to_cuda() 逻辑不变）
    llama_layers_->to_cuda(cuda_config_);

    // 拷贝完成后解注册（权重已在 GPU，CPU 端 mmap 不再需要 pin）
    cudaHostUnregister(weight_start);
}
```

---

## 方向三：统一内存池抽象与 CPU/CUDA 特化

> **v3 更新**：本方向基于 vLLM 和 SGLang 的内存管理实践进行了重大扩展，
> 新增 gpu_memory_utilization 显存预算机制、可选 TokenToKVPool 后端、Memory Profiling 等内容。

### 3.1 设计目标（对齐 vLLM/SGLang 最佳实践）

现有 `DeviceAllocator` 是一个粗粒度的 `allocate/release` 接口，存在以下问题：
1. CPU 和 CUDA 的分配语义差异大（同步 vs 异步、流感知 vs 全局），但接口完全相同
2. 权重内存（只读、mmap 背衬、生命周期=模型）与推理临时内存（读写、生命周期=单次 forward）走同一接口，无法针对性优化
3. CUDA 分配器没有流 (stream) 概念，无法利用 CUDA 11.2+ 的 `cudaMallocAsync`
4. **（新增）缺少 gpu_memory_utilization 等显存预算机制**，`num_blocks` 需外部硬编码传入，不如 vLLM 的自动计算灵活
5. **（新增）缺少类似 PyTorch caching allocator 的激活值内存管理**，NanoInfer 作为纯 C++ 项目必须自建

**设计思路**：
- 增加一层 `MemoryPool` 抽象（参考 PyTorch caching allocator 的分层思想）
- `DeviceAllocator` 继续作为**最底层的物理分配接口**不变
- 新增 `MemoryPool → CPUMemoryPool / CUDAMemoryPool` 继承体系
- **新增** `MemoryBudget` 模块，实现 vLLM 风格的 gpu_memory_utilization 自动显存规划
- **新增** 可选的 `TokenToKVPool` 后端，实现 SGLang 风格的扁平 token 级 KV cache 管理

### 3.2 统一抽象：MemoryPool

```
include/nanoinfer/base/
├── alloc.h            (现有，保留 DeviceAllocator 作为底层接口)
├── memory_pool.h      (新增：MemoryPool 统一抽象)
├── cpu_memory_pool.h  (新增：CPU 特化 — Arena + Slab 二级策略)
└── cuda_memory_pool.h (新增：CUDA 特化 — StreamPool + BuddyPool)
```

#### `memory_pool.h` — 统一接口

```cpp
namespace base {

/// @brief 内存池分配结果，携带指针 + 实际分配大小 + 所属池的后向引用
struct MemBlock {
    void*  ptr       = nullptr;
    size_t byte_size = 0;
};

/**
 * @brief 统一内存池抽象
 *
 * 相比 DeviceAllocator，MemoryPool 增加了：
 *   - 流感知分配（CUDA 侧异步）
 *   - reset()：O(1) 批量释放（Arena 模式）
 *   - checkpoint / rollback：部分回滚（嵌套 forward 场景）
 *   - stat()：内存使用统计
 *
 * 仍保留 DeviceType，CPU/CUDA 特化子类分别实现。
 */
class MemoryPool {
public:
    virtual ~MemoryPool() = default;

    /// @brief 同步分配（CPU 和 CUDA 均支持）
    virtual MemBlock allocate(size_t byte_size, size_t alignment = 64) = 0;

    /// @brief 异步分配（仅 CUDA 有意义；CPU 侧退化为同步）
    virtual MemBlock allocate_async(size_t byte_size, void* stream,
                                    size_t alignment = 64) {
        (void)stream;
        return allocate(byte_size, alignment);
    }

    /// @brief 归还内存（Arena 模式下为空操作，Slab/Pool 模式下回收）
    virtual void release(MemBlock block) = 0;

    /// @brief 异步释放（仅 CUDA 有意义）
    virtual void release_async(MemBlock block, void* stream) {
        (void)stream;
        release(block);
    }

    /// @brief 重置池（Arena 模式：O(1) 全部失效；Slab 模式：空操作）
    virtual void reset() {}

    /// @brief 当前已分配字节数
    virtual size_t used_bytes() const = 0;

    /// @brief 池总容量（Arena）或已缓存字节数（Slab）
    virtual size_t capacity_bytes() const = 0;

    DeviceType device_type() const { return device_type_; }

protected:
    explicit MemoryPool(DeviceType dt) : device_type_(dt) {}
    DeviceType device_type_;
};

} // namespace base
```

### 3.3 CPU 特化：二级策略（Arena + Slab）

```cpp
namespace base {

/**
 * @brief CPU Arena 内存池（Bump Pointer，仿 ggml_tallocr）
 *
 * 用途：推理临时 Tensor（norm_out / q / k / v / attn_out / w1_out 等）
 *   - allocate() O(1)：仅移动指针
 *   - reset()    O(1)：指针归零
 *   - 线程不安全：每个推理 worker 独占一个 CPUArenaPool
 *
 * 后备内存通过 posix_memalign 预分配（64B 对齐）。
 */
class CPUArenaPool final : public MemoryPool {
public:
    explicit CPUArenaPool(size_t capacity);
    ~CPUArenaPool() override;

    MemBlock allocate(size_t byte_size, size_t alignment = 64) override;
    void     release(MemBlock block) override {}  // Arena：release 是空操作
    void     reset()                  override;   // O(1)

    size_t used_bytes()     const override { return used_; }
    size_t capacity_bytes() const override { return capacity_; }

    /// @brief 记录检查点（嵌套 forward 场景，如 beam search）
    size_t checkpoint() const { return used_; }
    void   rollback(size_t cp) { used_ = cp; }

private:
    uint8_t* base_     = nullptr;
    size_t   used_     = 0;
    size_t   capacity_ = 0;
};

/**
 * @brief CPU Slab 内存池（Size-Class 空闲列表，仿 tcmalloc per-thread cache）
 *
 * 用途：持久 Tensor（KV Cache、持久化激活缓冲，生命周期 > 单次 forward）
 *   - 按 2 的幂取整到 size class，每个 class 维护 vector<void*> 空闲列表
 *   - 超大块（> kLargeThreshold）直接走 posix_memalign，不入 slab
 *   - 线程安全：每个 size class 一把 mutex
 */
class CPUSlabPool final : public MemoryPool {
public:
    static constexpr int    kNumClasses    = 20;    // 16B ~ 512KB
    static constexpr size_t kLargeThresh   = 512 * 1024;
    static constexpr size_t kMinAlign      = 16;

    explicit CPUSlabPool();
    ~CPUSlabPool() override;

    MemBlock allocate(size_t byte_size, size_t alignment = 64) override;
    void     release(MemBlock block) override;

    size_t used_bytes()     const override { return used_bytes_.load(); }
    size_t capacity_bytes() const override { return cached_bytes_.load(); }

private:
    static int    size_class_idx(size_t size);
    static size_t size_class_size(int idx);

    struct FreeList {
        mutable std::mutex    mtx;
        std::vector<void*>    slots;
        size_t                class_size = 0;
    };
    FreeList free_lists_[kNumClasses];
    std::atomic<size_t> used_bytes_{0};
    std::atomic<size_t> cached_bytes_{0};
};

} // namespace base
```

**Size class 表**（20 级，16B → 512KB）：

```
idx  0   1   2   3   4   5    6    7    8    9
size 16  32  64  128 256 512  1K   2K   4K   8K
idx  10  11  12  13  14  15   16   17   18   19
size 16K 32K 64K 128K 256K 512K  1M(→large)
```

### 3.4 CUDA 特化：二级策略（StreamOrderedPool + BuddyPool）

```cpp
namespace base {

/**
 * @brief CUDA Stream-Ordered 内存池（CUDA 11.2+ cudaMallocAsync/cudaFreeAsync）
 *
 * 用途：推理过程中所有 CUDA 设备端临时 Tensor
 *
 * 核心优势（相比现有 CUDADeviceAllocator）：
 *   - 流内异步分配，不阻塞 cudaDeviceSynchronize
 *   - CUDA driver 在流的时间线上自动复用已完成的分配（天然类 Arena 行为）
 *   - 无需手动 GC，driver 自动管理空闲池上限（可通过 cudaMemPoolSetAttribute 配置）
 *   - 线程安全：cudaMemPool_t 本身是线程安全的
 */
class CUDAStreamPool final : public MemoryPool {
public:
    explicit CUDAStreamPool(int device_id, size_t initial_size = 256 * 1024 * 1024);
    ~CUDAStreamPool() override;

    MemBlock allocate(size_t byte_size, size_t alignment = 256) override;
    MemBlock allocate_async(size_t byte_size, void* stream,
                            size_t alignment = 256) override;
    void release(MemBlock block)                override;
    void release_async(MemBlock block, void* stream) override;

    /// @brief 修改 pool 允许保留的最大空闲显存
    void set_release_threshold(size_t max_idle_bytes);

    size_t used_bytes()     const override;
    size_t capacity_bytes() const override;

private:
    int             device_id_;
    cudaMemPool_t   mem_pool_ = nullptr;
};

/**
 * @brief CUDA Buddy 内存池（大块分配，向后兼容现有 cudaDeviceAllocator 语义）
 *
 * 用途：KV Cache 物理块、持久中间 buffer（生命周期 > forward pass）
 *   - Buddy System：将显存按 2 的幂拆分，合并空闲的 buddy 对
 *   - 相比 size-class 池，碎片率更低（大块工作负载）
 *   - 每个 device 一把 mutex
 *
 * 注意：CUDA 11.2 以下退化为现有的 size-class 池实现。
 */
class CUDABuddyPool final : public MemoryPool {
public:
    explicit CUDABuddyPool(int device_id, size_t total_size);
    ~CUDABuddyPool() override;

    MemBlock allocate(size_t byte_size, size_t alignment = 256) override;
    void     release(MemBlock block) override;

    size_t used_bytes()     const override;
    size_t capacity_bytes() const override;

private:
    struct BuddyNode { ... };  // 省略实现细节
    int                  device_id_;
    mutable std::mutex   mtx_;
    void*                base_ptr_ = nullptr;
    size_t               total_size_;
};

} // namespace base
```

### 3.5 两类内存的生命周期对照

```
┌────────────────┬──────────────────────┬──────────────────────────────────────┐
│ 内存类型       │ 生命周期              │ 推荐池                               │
├────────────────┼──────────────────────┼──────────────────────────────────────┤
│ 权重 (CPU)     │ 模型生命周期          │ mmap 映射，use_external=true，不走池  │
│ 权重 (GPU)     │ 模型生命周期          │ CUDABuddyPool 一次性分配后不释放      │
│ KV Cache (GPU) │ 请求生命周期          │ Engine 持有的 CUDABuddyPool 块管理    │
│ 推理临时 (CPU) │ 单次 forward_batched │ CPUArenaPool（forward 完成后 reset）  │
│ 推理临时 (GPU) │ 单次 forward_batched │ CUDAStreamPool（stream 内自动复用）   │
│ 元数据/对象    │ 不定                 │ CPUSlabPool（size-class 分桶）        │
└────────────────┴──────────────────────┴──────────────────────────────────────┘
```

### 3.6 vLLM 风格 gpu_memory_utilization 显存预算机制（v3 新增）

> **背景**：当前 NanoInfer 的 `EngineConfig::num_blocks` 由**用户外部硬编码**传入，
> 用户需要手动估算显存余量，容易设错（过大 → OOM，过小 → 显存浪费）。
> vLLM 的 `gpu_memory_utilization` 参数（默认 0.9）实现了**自动显存预算计算**，
> 是其被广泛采用的关键易用性特征之一。

#### 实现方案

在 `Engine::init()` 中新增 memory profiling 步骤，自动计算最优 `num_blocks`：

```cpp
// include/nanoinfer/engine/memory_budget.h （新增）
namespace engine {

struct MemoryBudget {
    size_t total_gpu_memory;       ///< GPU 总显存
    size_t usable_memory;          ///< total * gpu_memory_utilization
    size_t model_weight_memory;    ///< 模型权重显存占用
    size_t activation_memory;      ///< forward pass 激活值峰值显存（通过 profiling 或公式估算）
    size_t kv_cache_memory;        ///< usable - weight - activation
    int32_t num_kv_blocks;         ///< kv_cache_memory / per_block_size
    float   gpu_memory_utilization; ///< 用户配置（默认 0.9）
};

/**
 * @brief 计算 KV Cache 可用显存预算（对齐 vLLM 实现）
 *
 * 计算流程：
 *   1. 查询 GPU 总显存: cudaMemGetInfo
 *   2. 统计模型权重占用: sum(weight_tensor.byte_size())
 *   3. 估算激活值显存: 基于 max_batch_size × max_seq_len × hidden_dim 的公式
 *      或执行一次空跑 profiling（更精确但需要额外启动时间）
 *   4. 计算 KV cache 预算: total × utilization - weight - activation
 *   5. 换算为 block 数量
 *
 * @param config     Engine 配置
 * @param model      已加载的模型（用于获取权重大小和结构参数）
 * @param budget     输出的内存预算
 */
base::Status compute_memory_budget(const EngineConfig& config,
                                   const model::Model& model,
                                   MemoryBudget& budget);

} // namespace engine
```

**核心计算逻辑**：

```cpp
base::Status compute_memory_budget(const EngineConfig& config,
                                   const model::Model& model,
                                   MemoryBudget& budget) {
    // Step 1: GPU 总显存
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    budget.total_gpu_memory = total_mem;
    budget.gpu_memory_utilization = config.gpu_memory_utilization;  // 默认 0.9
    budget.usable_memory = static_cast<size_t>(total_mem * budget.gpu_memory_utilization);

    // Step 2: 模型权重显存（已加载到 GPU 后的实际占用）
    budget.model_weight_memory = model.total_weight_bytes_on_device();

    // Step 3: 激活值显存估算（公式法，无需空跑）
    // 参考 vLLM 的 _get_max_num_batched_tokens 思路
    auto& tc = model.config();
    int32_t max_tokens = config.max_batch_size * config.prefill_chunk_size;
    // 每个 token 在一次 forward pass 中的峰值激活显存：
    //   hidden_states + norm_out + q + k + v + attn_out + ffn_up + ffn_gate + ffn_out
    //   ≈ 9 × hidden_dim × sizeof(float)  (FP32 推理)
    //   对于 Int8 推理，matmul 输出仍为 FP32，所以激活同样是 FP32
    size_t per_token_activation = 9ULL * tc.dim_ * sizeof(float);
    budget.activation_memory = max_tokens * per_token_activation;

    // 加上固定开销（block_table tensor, sampled_ids 等）
    budget.activation_memory += 64 * 1024 * 1024;  // 64MB headroom

    // Step 4: KV Cache 预算
    if (budget.usable_memory <= budget.model_weight_memory + budget.activation_memory) {
        return base::error::InternalError("GPU memory insufficient for model + activations");
    }
    budget.kv_cache_memory = budget.usable_memory
                             - budget.model_weight_memory
                             - budget.activation_memory;

    // Step 5: 换算为 block 数
    // per_block_size = block_size × num_layers × 2(K+V) × num_kv_heads × head_dim × sizeof(dtype)
    int32_t head_dim = tc.dim_ / tc.num_heads_;
    size_t per_block_bytes = static_cast<size_t>(config.block_size)
                           * tc.num_layers_ * 2
                           * tc.num_kv_heads_ * head_dim
                           * sizeof(float);  // FP32 KV cache
    budget.num_kv_blocks = static_cast<int32_t>(budget.kv_cache_memory / per_block_bytes);

    LOG(INFO) << "Memory Budget: total=" << (total_mem >> 20) << "MB"
              << " usable=" << (budget.usable_memory >> 20) << "MB"
              << " weights=" << (budget.model_weight_memory >> 20) << "MB"
              << " activation=" << (budget.activation_memory >> 20) << "MB"
              << " kv_cache=" << (budget.kv_cache_memory >> 20) << "MB"
              << " num_blocks=" << budget.num_kv_blocks;

    return base::error::Success();
}
```

**集成到 Engine::init()**：

```cpp
base::Status Engine::init(const model::Model& model) {
    // ... 现有初始化逻辑 ...

    // v3 新增：自动计算 num_blocks（如果用户未显式指定）
    if (config_.num_blocks <= 0 && device_type_ == base::DeviceType::kDeviceCUDA) {
        engine::MemoryBudget budget;
        auto status = compute_memory_budget(config_, model, budget);
        if (!status) return status;
        config_.num_blocks = budget.num_kv_blocks;
        // 同时可输出 max_concurrent_sequences 估算
        LOG(INFO) << "Auto-computed num_blocks=" << config_.num_blocks
                  << " (supports ~" << budget.num_kv_blocks * config_.block_size
                  << " total KV tokens)";
    }

    // 继续 KV cache 初始化...
}
```

### 3.7 SGLang 风格 TokenToKVPool 可选后端（v3 新增）

> **背景**：NanoInfer 当前的 `BlockManager` + `KVCacheManager` 已对齐 vLLM 的 block-based 模式。
> 但 SGLang 的 token-level flat pool 在某些场景下有优势（无内部碎片、实现更简洁、
> 对精确前缀缓存友好），可作为**可选后端**提供。

#### KVCacheBackend 抽象

```cpp
// include/nanoinfer/engine/kv_cache_backend.h（新增）
namespace engine {

/// @brief KV Cache 管理后端抽象
/// 统一 block-based (vLLM 风格) 和 token-level (SGLang 风格) 两种实现
class KVCacheBackend {
public:
    virtual ~KVCacheBackend() = default;

    /// @brief 初始化后端，分配 KV cache 张量
    virtual base::Status init(int32_t num_layers, int32_t num_kv_heads,
                             int32_t head_dim, int32_t max_capacity,
                             base::DataType dtype,
                             std::shared_ptr<base::DeviceAllocator> allocator) = 0;

    /// @brief 为序列分配 KV cache 空间
    virtual base::Status allocate_sequence(int32_t seq_id, int32_t num_tokens) = 0;

    /// @brief 扩展序列（追加 token）
    virtual base::Status extend_sequence(int32_t seq_id, int32_t additional_tokens) = 0;

    /// @brief 释放序列占用的 KV cache
    virtual base::Status free_sequence(int32_t seq_id) = 0;

    /// @brief 获取 KV cache 张量（attention kernel 使用）
    virtual tensor::Tensor& get_key_cache(int32_t layer_idx) = 0;
    virtual tensor::Tensor& get_value_cache(int32_t layer_idx) = 0;

    /// @brief 获取 attention kernel 需要的位置映射信息
    /// Block-based: 返回 block_table [num_seqs, max_blocks]
    /// Token-level: 返回 token_indices [num_seqs, max_seq_len]
    virtual base::Status get_position_mapping(const std::vector<int32_t>& seq_ids,
                                             tensor::Tensor& mapping) = 0;

    /// @brief 查询可用容量
    virtual int32_t get_available_tokens() const = 0;
};

/// @brief Block-based 后端（封装现有 BlockManager + KVCacheManager）
class BlockBasedBackend : public KVCacheBackend { /* 现有逻辑的包装 */ };

/// @brief Token-level 后端（SGLang TokenToKVPool 风格）
class TokenLevelBackend : public KVCacheBackend { /* 新实现 */ };

} // namespace engine
```

#### TokenLevelBackend 核心实现

```cpp
// src/engine/token_level_backend.cpp（新增）

/**
 * @brief SGLang 风格的 Token-Level KV Cache 池
 *
 * 与 SGLang TokenToKVPool 的对应关系：
 *   NanoInfer                    SGLang
 *   ────────                    ──────
 *   key_caches_                  TokenToKVPool.k_buffer
 *   value_caches_                TokenToKVPool.v_buffer
 *   free_slots_ (stack)          TokenToKVPool.free_slots
 *   seq_token_indices_           ReqToTokenPool.req_to_token
 *
 * KV Cache 张量布局（与 SGLang 一致）：
 *   key_cache[layer]:   [max_total_tokens, num_kv_heads, head_dim]
 *   value_cache[layer]: [max_total_tokens, num_kv_heads, head_dim]
 *
 * vs Block-based 布局（vLLM/NanoInfer 现有）：
 *   key_cache[layer]:   [num_blocks, num_kv_heads, block_size, head_dim]
 *   value_cache[layer]: [num_blocks, num_kv_heads, block_size, head_dim]
 */
class TokenLevelBackend : public KVCacheBackend {
    // KV Cache 张量 [max_total_tokens, heads, dim] per layer
    std::vector<tensor::Tensor> key_caches_;
    std::vector<tensor::Tensor> value_caches_;

    // 空闲 slot 栈（O(1) 分配/释放）
    std::vector<int32_t> free_slots_;

    // 每个序列的 token→slot 映射 (类似 SGLang ReqToTokenPool)
    std::unordered_map<int32_t, std::vector<int32_t>> seq_token_indices_;

    int32_t max_total_tokens_;

public:
    base::Status init(int32_t num_layers, int32_t num_kv_heads,
                     int32_t head_dim, int32_t max_total_tokens,
                     base::DataType dtype,
                     std::shared_ptr<base::DeviceAllocator> allocator) override {
        max_total_tokens_ = max_total_tokens;

        // 初始化空闲栈
        free_slots_.reserve(max_total_tokens);
        for (int32_t i = max_total_tokens - 1; i >= 0; --i) {
            free_slots_.push_back(i);
        }

        // 预分配 KV cache 张量
        for (int32_t layer = 0; layer < num_layers; ++layer) {
            // Shape: [max_total_tokens, num_kv_heads, head_dim]
            key_caches_.emplace_back(dtype, max_total_tokens, num_kv_heads,
                                     head_dim, true, allocator);
            value_caches_.emplace_back(dtype, max_total_tokens, num_kv_heads,
                                      head_dim, true, allocator);
        }

        return base::error::Success();
    }

    base::Status allocate_sequence(int32_t seq_id, int32_t num_tokens) override {
        if (free_slots_.size() < num_tokens) {
            return base::error::InternalError("TokenLevelBackend OOM");
        }
        auto& indices = seq_token_indices_[seq_id];
        indices.reserve(num_tokens);
        for (int32_t i = 0; i < num_tokens; ++i) {
            indices.push_back(free_slots_.back());
            free_slots_.pop_back();
        }
        return base::error::Success();
    }

    base::Status extend_sequence(int32_t seq_id, int32_t additional_tokens) override {
        if (free_slots_.size() < additional_tokens) {
            return base::error::InternalError("TokenLevelBackend OOM");
        }
        auto& indices = seq_token_indices_[seq_id];
        for (int32_t i = 0; i < additional_tokens; ++i) {
            indices.push_back(free_slots_.back());
            free_slots_.pop_back();
        }
        return base::error::Success();
    }

    base::Status free_sequence(int32_t seq_id) override {
        auto it = seq_token_indices_.find(seq_id);
        if (it == seq_token_indices_.end()) {
            return base::error::InvalidArgument("Sequence not found");
        }
        for (int32_t slot : it->second) {
            free_slots_.push_back(slot);
        }
        seq_token_indices_.erase(it);
        return base::error::Success();
    }

    /// @brief 返回 token indices 张量（attention kernel 直接用 slot_id 索引 KV cache）
    base::Status get_position_mapping(const std::vector<int32_t>& seq_ids,
                                     tensor::Tensor& mapping) override {
        // 构建 [num_seqs, max_seq_len] 的 token indices 矩阵
        // attention kernel 使用 mapping[seq][pos] 作为 KV cache 的行索引
        // ... (类似 BlockTable::to_gpu_format 的逻辑)
    }
};
```

**使用场景建议**：

| 场景 | 推荐后端 | 理由 |
|------|---------|------|
| 通用推理 | BlockBased（默认） | 成熟稳定，与现有 PagedAttention kernel 兼容 |
| 长序列 + 密集前缀共享 | TokenLevel | 无内部碎片，前缀缓存更精确 |
| 内存受限 | TokenLevel | 无 block 粒度浪费 |
| Beam Search / 并行采样 | BlockBased | CoW + ref counting 更高效 |

**注意**：TokenLevel 后端需要配套的 attention kernel 修改（索引方式从 `block_table + offset` 变为直接 `slot_index`），这是一个非平凡的工作量。建议作为 Phase 3+ 的可选演进方向。

### 3.8 与现有代码的集成策略（更新版）

**最小侵入式**：不改变 `DeviceAllocator` 接口，在 `LLamaModel` 和 `Engine` 中以**组合方式**引入池。v3 额外新增 `MemoryBudget` 自动计算和可选 `KVCacheBackend` 切换：

```cpp
class Engine {
    // v3 新增：KV Cache 后端抽象（默认 BlockBased，可选 TokenLevel）
    std::unique_ptr<KVCacheBackend> kv_backend_;

    base::Status init(const model::Model& model) {
        // 自动计算 num_blocks（如果未指定）
        if (config_.num_blocks <= 0) {
            MemoryBudget budget;
            compute_memory_budget(config_, model, budget);
            config_.num_blocks = budget.num_kv_blocks;
        }

        // 选择 KV Cache 后端
        if (config_.kv_cache_backend == "token_level") {
            kv_backend_ = std::make_unique<TokenLevelBackend>();
        } else {
            kv_backend_ = std::make_unique<BlockBasedBackend>();  // 默认
        }
        // ...
    }
};

class LLamaModel : public Model {
    // 新增：推理临时内存池（通过 init() 初始化，forward_batched() 使用）
    std::unique_ptr<base::CPUArenaPool>    cpu_arena_;   // CPU 推理
    std::unique_ptr<base::CUDAStreamPool>  cuda_stream_pool_;  // GPU 推理

    base::Status forward_batched(...) override {
        cpu_arena_->reset();  // O(1) 清除上次的临时 tensor
        // 所有临时 tensor 用 cpu_arena_->allocate() 代替 allocator->allocate()
        // hidden_states / norm_out / q / k / v / attn_out / w1_out / w3_out / ffn_out
        // 均从 arena 分配，共计约 (7 * total_tokens * dim * sizeof(float)) 字节
        ...
    }
};
```

**Arena 容量预估**（以 TinyLlama-1.1B，batch=4，seq=512 为例）：

```
total_tokens = 4 × 512 = 2048
dim = 2048, hidden_dim = 8192, kv_dim = 256

临时 tensor：
  hidden_states : 2048 × 2048 × 4B = 16 MB
  norm_out      : 16 MB
  q             : 16 MB
  k / v         : 2048 × 256 × 4B × 2 = 4 MB
  attn_out      : 16 MB
  w1_out        : 2048 × 8192 × 4B = 64 MB
  w3_out        : 64 MB
  ffn_out       : 16 MB
  ─────────────────────────────────────
  总计约 212 MB → arena 预分配 256 MB
```

---

## 改动优先级与实施顺序

### Phase 1：高收益低风险修复（1-2 天）

| # | 改动 | 文件 | 预期收益 |
|---|------|------|---------|
| 1 | `MAP_PRIVATE` → `MAP_SHARED` | `src/model/model.cpp` | 多进程共享 page，冷启动加速 |
| 2 | 删除冗余 `FILE*`，改用 `pread` | `src/model/model.cpp` | 减少 1 个 fd，代码简洁 |
| 3 | `madvise(MADV_WILLNEED)` + `posix_fadvise` | `src/model/model.cpp` | 加速初始化时 page fault |
| 4 | `madvise(MADV_HUGEPAGE)` | `src/model/model.cpp` | 减少 TLB miss，大模型效果明显 |
| 5 | CUDA 分配器加 `std::mutex` | `src/base/alloc_cuda.cpp` | 修复线程安全问题 |
| 6 | CUDA 大块容差 1MB → 4MB | `src/base/alloc_cuda.cpp` | 减少不必要的 cudaMalloc |

### Phase 2：safetensors 支持（5-7 天）

| # | 改动 | 文件 |
|---|------|------|
| 7  | 引入 `nlohmann/json` (CPM) | `CMakeLists.txt` |
| 8  | 实现 `WeightLoader` 抽象 + `TensorView` | `include/nanoinfer/model/weight_loader.h` |
| 9  | 实现 `SafetensorsLoader`（单文件 + sharded） | `src/model/safetensors_loader.cpp` |
| 10 | 实现 `BinWeightLoader`（封装原有 bin 逻辑） | `src/model/bin_weight_loader.cpp` |
| 11 | `Model` 按扩展名路由加载器 | `src/model/model.cpp` |
| 12 | `llama_weight_names.h` 权重名映射表 | `include/nanoinfer/model/llama_weight_names.h` |
| 13 | `LLamaModel` 按名字绑定权重（FP32 路径） | `src/model/llama.cpp` |
| 14 | `export_llama2/3.py` 新增 safetensors 导出 | `tools/export_llama2.py`, `tools/export_llama3.py` |
| 15 | Int8 量化：`export_q8_safetensors()` + C++ scale 绑定 | 同上 + `src/model/llama.cpp` |

### Phase 3：统一内存池 + 显存预算（5-7 天）

| # | 改动 | 文件 |
|---|------|------|
| 16 | 实现 `MemoryPool` 统一抽象基类 | `include/nanoinfer/base/memory_pool.h` |
| 17 | 实现 `CPUArenaPool`（bump pointer） | `src/base/cpu_memory_pool.cpp` |
| 18 | 实现 `CPUSlabPool`（size-class） | 同上 |
| 19 | 实现 `CUDAStreamPool`（`cudaMallocAsync`） | `src/base/cuda_memory_pool.cpp` |
| 20 | 实现 `CUDABuddyPool`（大块替代现有 CUDA 池） | 同上 |
| 21 | `LLamaModel` 推理临时 tensor 改用 `CPUArenaPool` | `src/model/llama.cpp` |
| 22 | CUDA 推理临时 tensor 改用 `CUDAStreamPool` | `src/model/llama.cpp` |
| 23 | **（v3 新增）** 实现 `MemoryBudget` 自动 num_blocks 计算 | `src/engine/memory_budget.cpp` |
| 24 | **（v3 新增）** `EngineConfig` 新增 `gpu_memory_utilization` 参数 | `include/nanoinfer/engine/engine.h` |
| 25 | **（v3 新增）** `Engine::init()` 集成自动显存预算 | `src/engine/engine.cpp` |

### Phase 4：可选 TokenToKVPool 后端 + 扩展量化（7-10 天，可选）

| # | 改动 | 文件 |
|---|------|------|
| 26 | **（v3 新增）** `KVCacheBackend` 抽象接口 | `include/nanoinfer/engine/kv_cache_backend.h` |
| 27 | **（v3 新增）** `BlockBasedBackend` 封装现有逻辑 | `src/engine/block_based_backend.cpp` |
| 28 | **（v3 新增）** `TokenLevelBackend` (SGLang 风格) | `src/engine/token_level_backend.cpp` |
| 29 | **（v3 新增）** Token-level attention kernel 适配 | `src/op/kernels/` |
| 30 | **（v3 新增）** `QuantConfig` 量化配置抽象 (vLLM 风格) | `include/nanoinfer/model/quant_config.h` |
| 31 | **（v3 新增）** GPTQ/AWQ 兼容加载（如需求） | `src/model/gptq_loader.cpp` |

---

## 性能对比测试方案

### T1：内存分配器微基准测试

**目标**：量化 `CPUArenaPool`、`CPUSlabPool` 相对现有 `CPUDeviceAllocator` 的吞吐提升。

新增测试文件 `test/test_base/test_allocator_bench.cpp`：

```cpp
// 基准测试结构：对每种分配器重复执行 N 次分配+释放，统计耗时
struct BenchConfig {
    size_t alloc_size;    // 单次分配大小
    int    num_allocs;    // 循环次数
    int    warmup;        // 预热轮次
};

// 预期对比项
// ┌─────────────────────────┬─────────┬──────────────────────────────────────┐
// │ 分配器                  │ 操作    │ 预期结果                              │
// ├─────────────────────────┼─────────┼──────────────────────────────────────┤
// │ CPUDeviceAllocator      │ alloc   │ baseline（posix_memalign 系统调用）   │
// │ CPUArenaPool            │ alloc   │ 约 10-50x 加速（纯指针移动，无 syscall）│
// │ CPUArenaPool            │ reset   │ O(1)，纳秒级                          │
// │ CPUSlabPool（命中）     │ alloc   │ 约 3-5x 加速（空闲列表 pop）           │
// │ CPUSlabPool（未命中）   │ alloc   │ 与 baseline 相当（走 posix_memalign） │
// └─────────────────────────┴─────────┴──────────────────────────────────────┘

void bench_cpu_allocators(BenchConfig cfg) {
    using namespace std::chrono;
    auto run = [&](auto& alloc, const char* name) {
        std::vector<base::MemBlock> blocks;
        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < cfg.num_allocs; ++i)
            blocks.push_back(alloc.allocate(cfg.alloc_size));
        for (auto& b : blocks) alloc.release(b);
        auto t1 = high_resolution_clock::now();
        double ns = duration_cast<nanoseconds>(t1 - t0).count();
        printf("%-28s  alloc+free × %d = %.1f ns/op\n",
               name, cfg.num_allocs, ns / cfg.num_allocs);
    };
    // baseline
    auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    // ...run with wrapper

    base::CPUArenaPool arena(64 * 1024 * 1024);  // 64MB
    run(arena, "CPUArenaPool");

    base::CPUSlabPool slab;
    run(slab, "CPUSlabPool");
}
```

**运行命令（集成到现有 CMake 测试框架）**：

```bash
cd build && ctest -R test_allocator_bench -V
# 或直接运行：
./test/test_nanoinfer --gtest_filter="AllocBench*"
```

---

### T2：模型加载时间对比

**目标**：验证 mmap 优化（`MAP_SHARED` + `madvise`）对冷/热启动的影响。

新增测试 `test/test_model/test_load_time.cpp`：

```cpp
// 测试矩阵（每个配置运行 5 次，取中位数）：
// ┌──────────────────────────────────┬───────────────────────────────────┐
// │ 配置                              │ 度量指标                          │
// ├──────────────────────────────────┼───────────────────────────────────┤
// │ 原始 MAP_PRIVATE，无 madvise       │ cold: 第一次加载耗时（ms）        │
// │ MAP_SHARED，无 madvise             │ cold: 同上；hot: 二次加载耗时（ms）│
// │ MAP_SHARED + MADV_WILLNEED        │ 同上                              │
// │ MAP_SHARED + MADV_WILLNEED + HUGEPAGE │ 同上                          │
// │ safetensors（等效权重）            │ 同上                              │
// └──────────────────────────────────┴───────────────────────────────────┘

auto measure_load = [](const std::string& model_path) {
    // 冷启动：先 echo 3 > /proc/sys/vm/drop_caches 清 page cache
    auto t0 = std::chrono::steady_clock::now();
    auto model = model::create_model(model_path);
    model->init(base::DeviceType::kDeviceCPU);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
};
```

---

### T3：推理热路径内存分配开销隔离

**目标**：测量 `forward_batched` 中因临时 tensor 分配/释放产生的时间占比，验证 Arena 化后的提升。

使用 `base::tick.h`（项目已有）插桩：

```cpp
// 在 forward_batched 中已有的临时 tensor 分配前后打点
// 对比：
//   方案 A：所有临时 tensor 走 allocator（现有方式）
//   方案 B：所有临时 tensor 走 CPUArenaPool（优化后）

// 测量维度：
//   1. 单次 forward pass 总耗时（TTFT for single token）
//   2. 其中"纯分配时间"占比（通过打点隔离）
//   3. 连续推理 100 tokens 的平均 TPOT（time per output token）
```

---

### T4：CUDA 分配器吞吐对比

**目标**：验证 size-class 分桶相对线性扫描的查找加速，以及 `CUDAStreamPool` 相对手动池的优势。

```cpp
// test/test_cuda_kernel/test_cuda_alloc_bench.cu
void bench_cuda_allocators() {
    const int N = 1000;   // 分配次数
    const size_t sizes[] = { 1024, 4096, 65536, 1 << 20, 64 << 20 };

    // 对比：
    //   1. 现有 CUDADeviceAllocator（线性扫描）
    //   2. 新 Size-Class Pool（O(1) 查找）
    //   3. CUDAStreamPool（cudaMallocAsync，流内零等待）
    //
    // 预期结果（alloc+free 吞吐，单位 GB/s）：
    //   大量小块（<1MB）：size-class 应 2-5x 快于线性扫描
    //   大块（>1MB）：BuddyPool 应优于 best-fit 扫描
    //   Stream Pool：在流内连续场景下几乎消除同步开销
}
```

---

### T5：端到端 TTFT / TPOT 对比

**目标**：综合评估全部优化后的端到端推理性能提升。

使用现有的 `demo/batched_infer_multi_prompts.cpp` 作为基础，新增计时和统计输出：

```bash
# 测试脚本 tools/bench_e2e.sh
#!/bin/bash
MODEL=$1
BATCH_SIZES=(1 4 8 16)
PROMPT_LEN=512
OUTPUT_LEN=128

for BS in "${BATCH_SIZES[@]}"; do
    echo "=== batch_size=$BS ==="
    ./build/demo/batched_infer_multi_prompts \
        --model $MODEL \
        --batch_size $BS \
        --prompt_len $PROMPT_LEN \
        --output_len $OUTPUT_LEN \
        --warmup 3 \
        --repeat 10 \
        --report ttft,tpot,memory_peak
done
```

**对比矩阵**：

| 优化组合 | 模型加载(s) | TTFT(ms) | TPOT(ms/tok) | RSS峰值(MB) |
|---------|------------|---------|-------------|------------|
| 基线（当前） | — | — | — | — |
| +MAP_SHARED+madvise | — | — | — | — |
| +safetensors | — | — | — | — |
| +CPUArenaPool | — | — | — | — |
| +CUDAStreamPool | — | — | — | — |
| 全部优化 | — | — | — | — |

（数字留空，实测填写；建议使用 TinyLlama-1.1B 作为基准模型以便复现）

---

### T6：内存占用对比（RSS / GPU Mem）

```bash
# CPU 内存
/usr/bin/time -v ./build/demo/chat_demo --model model.bin 2>&1 | grep "Maximum resident"

# GPU 显存（每秒采样）
nvidia-smi dmon -s mu -d 1 -o T > gpu_mem.log &
./build/demo/chat_demo --model model.safetensors
kill %1
```

**验证点**：
- `MAP_SHARED` 下多进程共享同一模型，RSS 理论上仅增加 1 个模型大小（而非 N×）
- Arena 化消除了临时 Tensor 的堆外碎片，RSS 应更稳定（无锯齿）
- `CUDAStreamPool` 的 idle 显存阈值可配置，观察峰值与均值的差距

