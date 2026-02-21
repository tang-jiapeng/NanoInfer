/**
 * @file encode.cpp
 * @brief Tokenizer 编解码层实现
 *
 * SpeEncodeLayer: 封装 Google SentencePiece，适用于 LLaMA-2 等 .model 格式模型。
 *   - encode()  : 字符串 → Token ID 序列（可选添加 BOS/EOS）
 *   - decode()  : Token ID → 字符串
 *
 * BpeEncodeLayer: 封装 tiktoken BPE，适用于 LLaMA-3 等 tiktoken.json 格式模型。
 *   - 词表和特殊 token 从 JSON 加载
 *   - 支持 <|end_of_text|> 和 <|eot_id|> 两种停止 token
 */
#include "nanoinfer/op/encode.h"
#include <fstream>
#include "glog/logging.h"
#include "unicode.h"

namespace op {

/** @brief 构造并加载 SentencePiece 模型文件 */
SpeEncodeLayer::SpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using namespace sentencepiece::util;
    spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto rc = spe->Load(token_model_path_);
    if (rc.code() != StatusCode::kOk) {
        LOG(FATAL) << "The token model path is not valid, please check the path and type "
                      "of token model.";
    }
}

std::string SpeEncodeLayer::decode(int32_t token_id) const {
    CHECK(spe != nullptr);
    std::vector<int32_t> token_ids{token_id};
    return this->spe->DecodeIds(token_ids);
}

std::string SpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(spe != nullptr);
    return this->spe->DecodeIds(token_ids);
}

/** @brief 编码：字符串 → Token ID 序列（可选添加 BOS/EOS） */
std::vector<int32_t> SpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(spe != nullptr);
    // sentencepiece
    std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
    if (has_bos_) {
        input_ids.insert(input_ids.begin(), spe->bos_id());
    }
    if (has_eos_) {
        input_ids.push_back(spe->eos_id());
    }
    return input_ids;
}

bool SpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    CHECK(this->spe != nullptr);
    return token_id == this->spe->eos_id();
}

int32_t SpeEncodeLayer::vocab_size() const {
    CHECK(spe != nullptr);
    return spe->GetPieceSize();
}

int32_t SpeEncodeLayer::bos_id() const {
    CHECK(spe != nullptr);
    return spe->bos_id();
}

int32_t SpeEncodeLayer::eos_id() const {
    CHECK(spe != nullptr);
    return spe->eos_id();
}

static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

/**
 * @brief 构造并加载 tiktoken JSON 词表文件
 *
 * 流程：
 *   1. 解析 JSON 中的 added_tokens 为 special_tokens（<|bos|>、<|eos|>等）
 *   2. 解析 model.vocab 为常规 BPE encoder，将 GPT-2 字节表示（如 Ġ）还原为原始字节
 *   3. 初始化 tiktoken 实例
 * @param token_model_path  tiktoken.json 路径
 */
BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);
    CHECK(f.is_open())
        << "The token model path is not valid, please check the path and type of token model.";
    json data;
    try {
        data = json::parse(f);
    } catch (json::parse_error&) {
        LOG(FATAL)
            << "The token model path is not valid, please check the path and type of token model.";
    }

    // 加载特殊 token（BOS、EOS、模板符 etc.）
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& item : data["added_tokens"]) {
        special_tokens.insert({item["content"].get<std::string>(), item["id"].get<int>()});
    }

    // 加载常规 BPE 词表：将 GPT-2 字节表示（如 Ġ 代表空格）还原为原始字节
    ankerl::unordered_dense::map<std::string, int> encoder;
    for (const auto& v : data["model"]["vocab"].items()) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            key += unicode_utf8_to_byte(unicode_cpt_to_utf8(cpt));
        }
        encoder[key] = v.value().get<int32_t>();
    }

    // 读取 BOS/EOS 和所有停止 token
    auto find_special = [&](const std::string& key) -> int32_t {
        auto it = special_tokens.find(key);
        return (it != special_tokens.end()) ? static_cast<int32_t>(it->second) : -1;
    };
    bos_id_ = find_special("<|begin_of_text|>");
    eos_id_ = find_special("<|end_of_text|>");
    stop_token1_ = eos_id_;
    stop_token2_ = find_special("<|eot_id|>");

    CHECK(bos_id_ != -1) << "BOS token <|begin_of_text|> not found in tokenizer JSON";
    CHECK(eos_id_ != -1) << "EOS token <|end_of_text|> not found in tokenizer JSON";

    num_token_ = static_cast<int32_t>(encoder.size() + special_tokens.size());
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(std::move(encoder), std::move(special_tokens),
                                                     PAT_STR);
}

/**
 * @brief 编码：字符串 → Token ID 序列
 *
 * 直接将原始 UTF-8 文本传给 tiktoken 进行 BPE 分词。
 * encoder_ 的键已经是原始字节（由 unicode_utf8_to_byte 还原），
 * 因此不需要对空格做任何预处理。
 * 可选在序列首尾添加 BOS/EOS token。
 */
std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
    CHECK(tiktoken_ != nullptr);
    const std::vector<int> ids = tiktoken_->encode(sentence);

    std::vector<int32_t> input_ids;
    input_ids.reserve(ids.size() + 2);
    if (has_bos_) {
        input_ids.push_back(bos_id_);
    }
    for (int id : ids) {
        input_ids.push_back(static_cast<int32_t>(id));
    }
    if (has_eos_) {
        input_ids.push_back(eos_id_);
    }
    return input_ids;
}

/** @brief 解码：单个 Token ID → 字符串 */
std::string BpeEncodeLayer::decode(int32_t token_id) const {
    return decode(std::vector<int32_t>{token_id});
}

/**
 * @brief 解码：Token ID 序列 → 完整字符串
 *
 * tiktoken._decode_native() 直接返回原始字节（decoder_ 键已是原始字节），
 * 即空格以 0x20 呈现，无需任何后处理。
 */
std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(tiktoken_ != nullptr);
    std::vector<int> ids(token_ids.begin(), token_ids.end());
    return tiktoken_->decode(ids);
}

/** @brief 判断 token 是否为停止符（<|end_of_text|> 或 <|eot_id|>） */
bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
    return token_id == stop_token1_ || token_id == stop_token2_;
}

/** @brief 返回词表大小（常规 token + 特殊 token） */
int32_t BpeEncodeLayer::vocab_size() const {
    return num_token_;
}

/** @brief 返回 BOS token ID */
int32_t BpeEncodeLayer::bos_id() const {
    return bos_id_;
}

/** @brief 返回 EOS token ID */
int32_t BpeEncodeLayer::eos_id() const {
    return eos_id_;
}

}  // namespace op