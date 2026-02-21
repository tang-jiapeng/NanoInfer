#!/usr/bin/env python3
"""
NanoInfer HuggingFace 精度验证 — Greedy (argmax) 解码
用法：
  python eval/hf_verify.py --model_dir ./models/tinyllama_hf
  python eval/hf_verify.py --model_dir ./models/tinyllama_hf --tokenizer ./models/llama2/tokenizer.model
  python eval/hf_verify.py --model_dir ./models/tinyllama_hf --mode single
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# ── Prompts（与 C++ demo 完全一致）─────────────────────────────────────────
# demo/llama2.cpp — MAX_NEW_TOKENS = 128
PROMPT_SINGLE = (
    "Once upon a time, there was a little girl named Lily who lived in a small village near "
    "the mountains. One day, she found a mysterious key in the forest and",
    128,
)

# demo/batched_infer_multi_prompts.cpp — MAX_NEW_TOKENS = 64
PROMPTS_BATCHED = [
    (
        "Once upon a time, there was a little girl named Lily who lived in a small village.",
        64,
    ),
    ("The meaning of life is", 64),
    ("In a galaxy far far away, there was a powerful wizard who could", 64),
    (
        "The quick brown fox jumps over the lazy dog. This sentence is famous because",
        64,
    ),
]


def load(model_dir: str, tokenizer_path: str | None, device: str):
    print(f"Loading model : {model_dir}")
    if tokenizer_path:
        p = Path(tokenizer_path)
        tok = (
            LlamaTokenizer(vocab_file=str(p), legacy=True)
            if p.suffix == ".model"
            else LlamaTokenizer.from_pretrained(str(p), legacy=True)
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, legacy=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, device_map=device
    ).eval()

    print(
        f"bos={tok.bos_token_id}  eos={tok.eos_token_id}  "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n"
    )
    return model, tok


def infer(
    model, tok, prompt: str, max_new_tokens: int, device: str
) -> tuple[str, int, float]:
    """Greedy decode，返回 (生成文本, 生成 token 数, 耗时 ms)。"""
    ids = tok.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    prompt_len = ids.shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = argmax，与 C++ ArgmaxSampler 完全一致
            temperature=1.0,
            repetition_penalty=1.0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    new_ids = out[0][prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids), elapsed_ms


def run(model, tok, prompts: list, device: str, title: str):
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)
    for i, (prompt, max_new) in enumerate(prompts):
        text, gen_tokens, ms = infer(model, tok, prompt, max_new, device)
        print(f"\n[{i}] Prompt   : {prompt}")
        print(f"    Output   : {text}")
        print(
            f"    Tokens   : {gen_tokens} / {max_new}  |  "
            f"Time: {ms:.0f} ms  |  {gen_tokens * 1000 / ms:.1f} tok/s"
        )
    print()


def main():
    ap = argparse.ArgumentParser(description="NanoInfer HuggingFace 精度验证")
    ap.add_argument("--model_dir", required=True, help="HF 模型目录或 Hub model-id")
    ap.add_argument(
        "--tokenizer", default=None, help=".model 文件或含 tokenizer 的目录"
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["single", "batched", "all"], default="all")
    args = ap.parse_args()

    model, tok = load(args.model_dir, args.tokenizer, args.device)

    if args.mode in ("single", "all"):
        run(
            model,
            tok,
            [PROMPT_SINGLE],
            args.device,
            "demo/llama2.cpp  [single, max_new=128]",
        )

    if args.mode in ("batched", "all"):
        run(
            model,
            tok,
            PROMPTS_BATCHED,
            args.device,
            "demo/batched_infer_multi_prompts.cpp  [4 prompts, max_new=64]",
        )


if __name__ == "__main__":
    main()
