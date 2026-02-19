/**
 * @file model.cpp
 * @brief Model 基类实现（权重加载、Tokenizer 初始化、模型配置解析）
 *
 * Model 是所有具体模型（如 LLaMA）的基类，提供通用的初始化流水线：
 *
 *   gen_model_from_file()
 *     │
 *     ├─ read_model_file()       : 打开文件 → fread ModelConfig → mmap 整个文件
 *     ├─ generate_model_infos()  : 从 ModelConfig 推导 kv_dim / kv_mul / head_size 等
 *     ├─ create_encode_layer()   : 创建 SentencePiece Tokenizer，获取 vocab_size
 *     └─ create_layers()         : 纯虚函数，由子类实现各网络层的创建
 *
 * 权重文件通过 mmap 映射到内存，避免一次性加载全部参数。
 * 支持 FP32 和 Int8 量化模型格式（通过 is_quant_model_ 区分）。
 */
#include "nanoinfer/model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace model {

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {
}

base::ModelType Model::model_type() const {
    return model_type_;
}

const std::string& Model::token_path() const {
    return token_path_;
}

const std::string& Model::model_path() const {
    return model_path_;
}

/**
 * @brief 读取并 mmap 模型权重文件
 *
 * 流程：
 *   1. open 文件 → fread 读取头部 ModelConfig 结构体
 *   2. 量化模型额外读取 group_size
 *   3. mmap 整个文件到内存，设置 weight_data 指向头部之后的偏移
 */
base::Status Model::read_model_file() {
    using namespace base;
    if (model_path_.empty()) {
        return error::PathNotValid("Failed to open the weight file, the model path is empty!");
    }
    int32_t fd = open(model_path_.data(), O_RDONLY);
    if (fd == -1) {
        return error::PathNotValid("Failed to open the weight file " + model_path_ +
                                   " may be the path does not exist!");
    }

    FILE* file = fopen(model_path_.data(), "rb");
    if (!file) {
        return error::PathNotValid("Failed to open the file. The path may be invalid.");
    }

    auto config = ModelConfig{};
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
        return error::ModelParseError(
            "Failed to retrieve the configuration information from the model "
            "file.");
    }
    if (is_quant_model_) {
        if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
            return error::ModelParseError(
                "Failed to retrieve the group size information from the model "
                "file.");
        }
    }

    auto gen_status = generate_model_infos(config);
    if (!gen_status) {
        return gen_status;
    }

    if (!is_quant_model_) {
        raw_model_data_ = std::make_shared<RawModelDataFp32>();
    } else {
        raw_model_data_ = std::make_shared<RawModelDataInt8>();
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        return error::ModelParseError(
            "Failed to retrieve the file size information from the model "
            "file.");
    }
    raw_model_data_->file_size = sb.st_size;

    raw_model_data_->fd = fd;
    raw_model_data_->data =
        mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

    if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
        return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                      " into memory.");
    }
    if (!is_quant_model_) {
        raw_model_data_->weight_data =
            static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
    } else {
        raw_model_data_->weight_data =
            static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
    }
    if (raw_model_data_ == nullptr) {
        LOG(ERROR);
        return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                      " into memory, the pointer to weight start address is null");
    }
    return error::Success();
}

/**
 * @brief 从 ModelConfig 推导 Transformer 架构参数
 *
 * 派生参数：kv_dim = dim * kv_head_num / head_num，
 * kv_mul = head_num / kv_head_num（GQA 倍数），head_size = dim / head_num。
 * vocab_size 为负值表示不共享 Embedding 与 Classifier 权重。
 */
base::Status Model::generate_model_infos(const ModelConfig& config) const {
    config_->dim_ = config.dim;
    config_->hidden_dim_ = config.hidden_dim;
    config_->layer_num_ = config.layer_num;
    config_->head_num_ = config.head_num;
    config_->kv_head_num_ = config.kv_head_num;
    config_->seq_len_ = config.seq_len;

    switch (model_type_) {
        case base::ModelType::kModelTypeLLaMA2:
            config_->norm_eps_ = 1e-5f;
            config_->rope_theta_ = 10000.0f;
            config_->bos_token_id_ = 1;
            config_->eos_token_id_ = 2;
            break;
        case base::ModelType::kModelTypeLLaMA3:
            config_->norm_eps_ = 1e-6f;
            config_->rope_theta_ = 500000.0f;
            config_->bos_token_id_ = 128000;
            config_->eos_token_id_ = 128001;
            break;
        default:
            // 其它模型由 tokenizer 提供 bos/eos
            config_->bos_token_id_ = -1;
            config_->eos_token_id_ = -1;
            break;
    }

    config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
    config_->kv_mul_ = config.head_num / config.kv_head_num;
    config_->head_size_ = config.dim / config.head_num;

    if (config.vocab_size > 0) {
        config_->is_shared_weight_ = true;
    } else {
        config_->is_shared_weight_ = false;
    }

    config_->vocab_size_ = std::abs(config.vocab_size);
    return base::error::Success();
}

/** @brief 创建 Tokenizer（SentencePiece），并从中获取 vocab_size / bos_id / eos_id */
base::Status Model::create_encode_layer() {
    using namespace base;

    // create token encode decode layer
    if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
        encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
    } else if (tokenizer_type_ == TokenizerType::kEncodeBpe) {
        encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
    }

    if (!encode_layer_) {
        return error::InternalError("Create the encode layer failed.");
    }

    config_->vocab_size_ = encode_layer_->vocab_size();
    if (config_->vocab_size_ <= 0) {
        return error::InternalError("The vocab size param read error from the model file!");
    }

    if (config_->bos_token_id_ == -1) {
        config_->bos_token_id_ = encode_layer_->bos_id();
    }
    if (config_->eos_token_id_ == -1) {
        config_->eos_token_id_ = encode_layer_->eos_id();
    }

    return error::Success();
}

/**
 * @brief 模型初始化主流程：mmap + Tokenizer + 网络层创建
 *
 * 调用顺序：read_model_file() → create_encode_layer() → create_layers()
 */
base::Status Model::gen_model_from_file() {
    using namespace base;
    config_ = std::make_unique<TransformerConfig>();

    // mmap
    auto mmap_status = read_model_file();
    if (!mmap_status) {
        LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
        return mmap_status;
    }

    auto create_encode_status = create_encode_layer();
    if (!create_encode_status) {
        LOG(ERROR) << "Create the encode layer failed!";
        return create_encode_status;
    }

    auto layer_create_status = create_layers();
    if (!layer_create_status) {
        LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
        return layer_create_status;
    }

    return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->decode(token_idxs);
}

}  // namespace model