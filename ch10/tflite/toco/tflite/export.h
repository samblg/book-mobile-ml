#ifndef TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_

#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tflite/operator.h"
#include "tensorflow/lite/util.h"

namespace toco {

namespace tflite {

struct ExportParams {
  bool allow_custom_ops = false;
  bool enable_select_tf_ops = false;
  bool quantize_weights = false;
};

tensorflow::Status Export(const Model& model, string* output_file_contents,
                          const ExportParams& params);

tensorflow::Status Export(
    const Model& model, string* output_file_contents,
    const ExportParams& params,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type);

inline void Export(const Model& model, bool allow_custom_ops,
                   bool quantize_weights, string* output_file_contents) {
  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights = quantize_weights;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

inline void Export(
    const Model& model, bool allow_custom_ops, bool quantize_weights,
    string* output_file_contents,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type) {
  ExportParams params;
  params.allow_custom_ops = allow_custom_ops;
  params.quantize_weights = quantize_weights;
  auto status = Export(model, output_file_contents, params, ops_by_type);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

inline void Export(const Model& model, string* output_file_contents) {
  ExportParams params;
  params.allow_custom_ops = true;
  auto status = Export(model, output_file_contents, params);
  if (!status.ok()) LOG(QFATAL) << status.error_message();
}

namespace details {

using TensorsMap = std::unordered_map<string, int>;

class OperatorKey {
 public:
  OperatorKey() {}

  OperatorKey(
      const ::toco::OperatorSignature& op_signature,
      const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
      bool enable_select_tf_ops);

  OperatorKey(::tflite::BuiltinOperator type, const std::string& custom_code,
              int version)
      : type_(type), custom_code_(custom_code), version_(version) {}

  ::tflite::BuiltinOperator type() const { return type_; }
  const std::string& custom_code() const { return custom_code_; }
  int version() const { return version_; }

  bool is_custom_op() const { return is_custom_op_; }
  bool is_flex_op() const { return is_flex_op_; }
  bool is_unsupported_flex_op() const { return is_unsupported_flex_op_; }
  const std::string& flex_tensorflow_op() const { return flex_tensorflow_op_; }

  bool operator<(const OperatorKey& other) const {
    if (type_ < other.type_)
      return true;
    else if (type_ > other.type_)
      return false;
    else if (custom_code_ < other.custom_code_)
      return true;
    else if (custom_code_ > other.custom_code_)
      return false;
    else
      return version_ < other.version_;
  }

  bool operator==(const OperatorKey& other) const {
    return type_ == other.type_ && custom_code_ == other.custom_code_ &&
           version_ == other.version_;
  }

  struct Hash {
    size_t operator()(const OperatorKey& key) const {
      return ::tflite::CombineHashes(
          {std::hash<size_t>()(static_cast<size_t>(key.type())),
           std::hash<std::string>()(key.custom_code()),
           std::hash<int>()(key.version())});
    }
  };

 private:
  ::tflite::BuiltinOperator type_ = ::tflite::BuiltinOperator_CUSTOM;
  std::string custom_code_;
  int version_ = 1;

  bool is_custom_op_ = false;
  bool is_flex_op_ = false;
  bool is_unsupported_flex_op_ = false;
  std::string flex_tensorflow_op_;
};

using OperatorsMap = std::unordered_map<OperatorKey, int, OperatorKey::Hash>;

void LoadTensorsMap(const Model& model, TensorsMap* tensors_map);
void LoadOperatorsMap(
    const Model& model, OperatorsMap* operators_map,
    const std::map<OperatorType, std::unique_ptr<BaseOperator>>& ops_by_type,
    bool enable_select_tf_ops);

}  // namespace details
}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_EXPORT_H_
