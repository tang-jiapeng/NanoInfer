/**
 * @file encode.cpp
 * @brief SentencePiece 编解码层实现（SpeEncodeLayer）
 *
 * 封装 Google SentencePiece 库，提供：
 *   - encode()  : 字符串 → Token ID 序列（可选添加 BOS/EOS）
 *   - decode()  : Token ID → 字符串
 *   - vocab_size() / bos_id() / eos_id() : 词表信息查询
 */
#include "nanoinfer/op/encode.h"

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

}  // namespace op