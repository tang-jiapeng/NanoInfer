#!/usr/bin/env bash
# =============================================================================
# export_models.sh — NanoInfer 模型导出脚本
#
# 用法：
#   bash export_models.sh [命令]
#
# 可用命令：
#   download-llama2             从 HuggingFace 下载 TinyLlama 权重
#   download-llama3             从 HuggingFace 下载 LLaMA3.2-1B 权重
#   download-llama3-instruct    从 HuggingFace 下载 LLaMA3.2-1B-Instruct 权重 (新增)
#   download-qwen3              从 HuggingFace 下载 Qwen3-0.6B 权重
#   export-llama2-fp32          导出 TinyLlama FP32 格式
#   export-llama2-int8          导出 TinyLlama W8A32 量化格式
#   export-llama3-fp32          导出 LLaMA3.2-1B FP32 格式
#   export-llama3-int8          导出 LLaMA3.2-1B W8A32 量化格式
#   export-llama3-instruct-fp32 导出 LLaMA3.2-1B-Instruct FP32 格式 (新增)
#   export-qwen3-fp32           导出 Qwen3-0.6B FP32 格式
#   all                         依次执行 download + export（FP32 + INT8）所有步骤
#   help                        显示此帮助信息
#
# 路径约定（可在下方 CONFIG 区修改）：
#   MODELS_DIR                  存放权重的根目录，默认 ../models
#   TOOLS_DIR                   本脚本所在的 tools/ 目录
# =============================================================================

set -euo pipefail

# ---- 路径配置 ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}"
MODELS_DIR="${SCRIPT_DIR}/../models"

LLAMA2_HF_DIR="${MODELS_DIR}/llama2"
LLAMA3_HF_DIR="${MODELS_DIR}/llama3"
LLAMA3_INSTRUCT_HF_DIR="${MODELS_DIR}/llama3_instruct" # 新增 Instruct 目录
QWEN3_HF_DIR="${MODELS_DIR}/qwen3" # 新增 Qwen3 目录

LLAMA2_FP32_BIN="${MODELS_DIR}/llama2/llama2_fp32.bin"
LLAMA2_INT8_BIN="${MODELS_DIR}/llama2/llama2_int8.bin"
LLAMA3_FP32_BIN="${MODELS_DIR}/llama3/llama3_fp32.bin"
LLAMA3_INT8_BIN="${MODELS_DIR}/llama3/llama3_int8.bin"
LLAMA3_INSTRUCT_FP32_BIN="${MODELS_DIR}/llama3_instruct/llama3_instruct_fp32.bin" # 新增 Instruct bin
QWEN3_FP32_BIN="${MODELS_DIR}/qwen3/qwen3_fp32.bin" # 新增 Qwen3 bin

# HuggingFace 模型 ID
LLAMA2_HF_ID="TinyLlama/TinyLlama_v1.1"
LLAMA3_HF_ID="meta-llama/Llama-3.2-1B"
LLAMA3_INSTRUCT_HF_ID="meta-llama/Llama-3.2-1B-Instruct" # 新增 Instruct ID
QWEN3_HF_ID="Qwen/Qwen3-0.6B"
# =============================================================================
# 辅助函数
# =============================================================================
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

check_python() {
    python3 -c "import torch" 2>/dev/null || error "未找到 PyTorch，请先安装：pip install torch"
    python3 -c "import transformers" 2>/dev/null || error "未找到 transformers，请先安装：pip install transformers"
}

# =============================================================================
# 各步骤实现
# =============================================================================

do_download_llama2() {
    info "下载 TinyLlama 权重 → ${LLAMA2_HF_DIR}"
    mkdir -p "${LLAMA2_HF_DIR}"
    hf download "${LLAMA2_HF_ID}" --local-dir "${LLAMA2_HF_DIR}"
    info "TinyLlama 下载完成"
}

do_download_llama3() {
    info "下载 LLaMA3.2-1B 权重 → ${LLAMA3_HF_DIR}"
    mkdir -p "${LLAMA3_HF_DIR}"
    hf download "${LLAMA3_HF_ID}" --local-dir "${LLAMA3_HF_DIR}"
    info "LLaMA3.2-1B 下载完成"
}

do_download_llama3_instruct() {
    info "下载 LLaMA3.2-1B-Instruct 权重 → ${LLAMA3_INSTRUCT_HF_DIR}"
    mkdir -p "${LLAMA3_INSTRUCT_HF_DIR}"
    hf download "${LLAMA3_INSTRUCT_HF_ID}" --local-dir "${LLAMA3_INSTRUCT_HF_DIR}"
    info "LLaMA3.2-1B-Instruct 下载完成"
}

do_download_qwen3() {
    info "下载 Qwen3-0.6B 权重 → ${QWEN3_HF_DIR}"
    mkdir -p "${QWEN3_HF_DIR}"
    hf download "${QWEN3_HF_ID}" --local-dir "${QWEN3_HF_DIR}"
    info "Qwen3-0.6B 下载完成"
}

do_export_qwen3_fp32() {
    check_python
    info "导出 Qwen3-0.6B FP32 → ${QWEN3_FP32_BIN}"
    mkdir -p "$(dirname "${QWEN3_FP32_BIN}")"

    # Step 1: 从 HuggingFace 权重导出 PyTorch .pth 文件
    local pth_file="${MODELS_DIR}/qwen3/qwen3_0.6b_weights.pth"
    if [[ ! -f "${pth_file}" ]]; then
        info "正在导出 HuggingFace 权重为 .pth 格式..."
        pushd "${TOOLS_DIR}/export_qwen3" > /dev/null
        python3 load.py --model_name "${QWEN3_HF_DIR}" --output_file "${pth_file}"
        popd > /dev/null
    else
        info "已存在 .pth 文件，跳过导出: ${pth_file}"
    fi

    # Step 2: 将 .pth 转换为 NanoInfer 自定义 .bin 格式
    pushd "${TOOLS_DIR}/export_qwen3" > /dev/null
    python3 write_bin.py \
        -n "${QWEN3_HF_DIR}" \
        -p "${pth_file}" \
        -d cpu \
        -o "${QWEN3_FP32_BIN}"
    popd > /dev/null

    info "Qwen3-0.6B FP32 导出完成：${QWEN3_FP32_BIN}"
}

do_export_llama2_fp32() {
    check_python
    info "导出 TinyLlama FP32 → ${LLAMA2_FP32_BIN}"
    mkdir -p "$(dirname "${LLAMA2_FP32_BIN}")"
    python3 "${TOOLS_DIR}/export_llama2.py" "${LLAMA2_FP32_BIN}" \
        --hf "${LLAMA2_HF_DIR}" \
        --version 0 \
        --dtype fp32
    info "TinyLlama FP32 导出完成：${LLAMA2_FP32_BIN}"
}

do_export_llama2_int8() {
    check_python
    info "导出 TinyLlama W8A32 INT8 → ${LLAMA2_INT8_BIN}"
    mkdir -p "$(dirname "${LLAMA2_INT8_BIN}")"
    # version 3 = legacy_export_quant (NanoInfer INT8 格式，W8A32 分组量化)
    python3 "${TOOLS_DIR}/export_llama2.py" "${LLAMA2_INT8_BIN}" \
        --hf "${LLAMA2_HF_DIR}" \
        --version 3
    info "TinyLlama INT8 导出完成：${LLAMA2_INT8_BIN}"
}

do_export_llama3_fp32() {
    check_python
    info "导出 LLaMA3.2-1B FP32 → ${LLAMA3_FP32_BIN}"
    mkdir -p "$(dirname "${LLAMA3_FP32_BIN}")"
    python3 "${TOOLS_DIR}/export_llama3.py" "${LLAMA3_FP32_BIN}" \
        --hf "${LLAMA3_HF_DIR}" \
        --version 0 \
        --dtype fp32
    info "LLaMA3.2-1B FP32 导出完成：${LLAMA3_FP32_BIN}"
}

do_export_llama3_int8() {
    check_python
    info "导出 LLaMA3.2-1B W8A32 INT8 → ${LLAMA3_INT8_BIN}"
    mkdir -p "$(dirname "${LLAMA3_INT8_BIN}")"
    # version 3 = legacy_export_quant (NanoInfer INT8 格式，W8A32 分组量化)
    python3 "${TOOLS_DIR}/export_llama3.py" "${LLAMA3_INT8_BIN}" \
        --hf "${LLAMA3_HF_DIR}" \
        --version 3
    info "LLaMA3.2-1B INT8 导出完成：${LLAMA3_INT8_BIN}"
}

do_export_llama3_instruct_fp32() {
    check_python
    info "导出 LLaMA3.2-1B-Instruct FP32 → ${LLAMA3_INSTRUCT_FP32_BIN}"
    mkdir -p "$(dirname "${LLAMA3_INSTRUCT_FP32_BIN}")"
    # 直接复用 export_llama3.py，结构相同
    python3 "${TOOLS_DIR}/export_llama3.py" "${LLAMA3_INSTRUCT_FP32_BIN}" \
        --hf "${LLAMA3_INSTRUCT_HF_DIR}" \
        --version 0 \
        --dtype fp32
    info "LLaMA3.2-1B-Instruct FP32 导出完成：${LLAMA3_INSTRUCT_FP32_BIN}"
}

do_all() {
    do_download_llama2
    do_download_llama3
    do_download_llama3_instruct
    do_download_qwen3
    do_export_llama2_fp32
    do_export_llama2_int8
    do_export_llama3_fp32
    do_export_llama3_int8
    do_export_llama3_instruct_fp32
    do_export_qwen3_fp32
    info "所有步骤完成"
}

print_help() {
    sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,2\}//'
}

# =============================================================================
# 入口
# =============================================================================
CMD="${1:-help}"

case "${CMD}" in
    download-llama2)             do_download_llama2             ;;
    download-llama3)             do_download_llama3             ;;
    download-llama3-instruct)    do_download_llama3_instruct    ;;
    download-qwen3)              do_download_qwen3              ;;
    export-llama2-fp32)          do_export_llama2_fp32          ;;
    export-llama2-int8)          do_export_llama2_int8          ;;
    export-llama3-fp32)          do_export_llama3_fp32          ;;
    export-llama3-int8)          do_export_llama3_int8          ;;
    export-llama3-instruct-fp32) do_export_llama3_instruct_fp32 ;;
    export-qwen3-fp32)           do_export_qwen3_fp32           ;;
    all)                         do_all                         ;;
    help|--help|-h)              print_help                     ;;
    *)
        warn "未知命令：${CMD}"
        print_help
        exit 1
        ;;
esac