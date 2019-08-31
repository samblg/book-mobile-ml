#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {

const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  auto it = builtins_.find(std::make_pair(op, version));
  return it != builtins_.end() ? &it->second : nullptr;
}

const TfLiteRegistration* MutableOpResolver::FindOp(const char* op,
                                                    int version) const {
  auto it = custom_ops_.find(std::make_pair(op, version));
  return it != custom_ops_.end() ? &it->second : nullptr;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   const TfLiteRegistration* registration,
                                   int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    TfLiteRegistration new_registration = *registration;
    new_registration.custom_name = nullptr;
    new_registration.builtin_code = op;
    new_registration.version = version;
    auto op_key = std::make_pair(op, version);
    builtins_[op_key] = new_registration;
  }
}

void MutableOpResolver::AddCustom(const char* name,
                                  const TfLiteRegistration* registration,
                                  int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    TfLiteRegistration new_registration = *registration;
    new_registration.builtin_code = BuiltinOperator_CUSTOM;
    new_registration.custom_name = name;
    new_registration.version = version;
    auto op_key = std::make_pair(name, version);
    custom_ops_[op_key] = new_registration;
  }
}

void MutableOpResolver::AddAll(const MutableOpResolver& other) {
  for (const auto& other_builtin : other.builtins_) {
    builtins_[other_builtin.first] = other_builtin.second;
  }
  for (const auto& other_custom_op : other.custom_ops_) {
    custom_ops_[other_custom_op.first] = other_custom_op.second;
  }
}

}  // namespace tflite
