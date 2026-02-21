# eval — NanoInfer 精度验证工具

使用 HuggingFace transformers 对与 C++ demo 完全相同的 prompt 执行 **greedy (argmax) decoding**，
输出逐 token 的 ID 序列，与 C++ 推理结果对比以验证计算精度。

## 依赖安装

```bash
pip install -r eval/requirements.txt
```

## 快速开始

```bash
# 1. 准备 HF 格式模型（从你已有的 .bin 文件重新导出，或直接用原始 HF 权重目录）
#    假设 HF 权重在 ./models/tinyllama_hf/

# 2. 运行所有验证（单条 + 多条）
python eval/hf_verify.py \
    --model_dir ./models/tinyllama_hf \
    --mode all \
    --verbose

# 3. 若 model_dir 内没有 tokenizer，单独指定
python eval/hf_verify.py \
    --model_dir ./models/tinyllama_hf \
    --tokenizer_path ./models/llama2/tokenizer.model \
    --mode all

# 4. 直接从 HuggingFace Hub 加载（需网络）
python eval/hf_verify.py \
    --model_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --mode all
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `--model_dir` | HF 模型目录或 Hub model-id（必填）|
| `--tokenizer_path` | SentencePiece `.model` 文件路径（可选）|
| `--mode` | `single` / `batched` / `all`（默认 `all`）|
| `--device` | `cuda` / `cpu`（默认自动检测）|
| `--verbose` | 打印每解码步的 token ID，便于逐步对比 |

## 对应关系

| Python 脚本 mode | C++ demo |
|---|---|
| `--mode single` | `demo/llama2.cpp` |
| `--mode batched` | `demo/batched_infer_multi_prompts.cpp` |

## 验证方法

1. 在 C++ demo 中添加打印语句，输出每步生成的 token ID（`next_token_id`）。
2. 运行 `hf_verify.py --verbose`，查看 `[decode step N] id=XXXX` 行。
3. 逐步对比两侧 token ID：
   - **完全一致** → 精度验证通过 ✓  
   - **某步开始不一致** → 说明该步之前的累积误差导致了分叉，需定位具体算子。

> **注意**：C++ 使用 `float32`，HF 脚本默认也用 `torch.float32` 以消除 dtype 差异带来的误差。
> 若使用 `bfloat16` 权重，微小的数值差异可能导致极少数 token 分叉，属正常现象。
