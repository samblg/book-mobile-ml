#ifndef TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_

#include <unordered_map>
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace op_resolver_hasher {
template <typename V>
struct ValueHasher {
  size_t operator()(const V& v) const { return std::hash<V>()(v); }
};

template <>
struct ValueHasher<tflite::BuiltinOperator> {
  size_t operator()(const tflite::BuiltinOperator& v) const {
    return std::hash<int>()(static_cast<int>(v));
  }
};

template <typename T>
struct OperatorKeyHasher {
  size_t operator()(const T& x) const {
    size_t a = ValueHasher<typename T::first_type>()(x.first);
    size_t b = ValueHasher<typename T::second_type>()(x.second);
    return CombineHashes({a, b});
  }
};
}  // namespace op_resolver_hasher

class MutableOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                   int version) const override;
  const TfLiteRegistration* FindOp(const char* op, int version) const override;
  void AddBuiltin(tflite::BuiltinOperator op,
                  const TfLiteRegistration* registration, int min_version = 1,
                  int max_version = 1);
  void AddCustom(const char* name, const TfLiteRegistration* registration,
                 int min_version = 1, int max_version = 1);
  void AddAll(const MutableOpResolver& other);

 private:
  typedef std::pair<tflite::BuiltinOperator, int> BuiltinOperatorKey;
  typedef std::pair<std::string, int> CustomOperatorKey;

  std::unordered_map<BuiltinOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<BuiltinOperatorKey> >
      builtins_;
  std::unordered_map<CustomOperatorKey, TfLiteRegistration,
                     op_resolver_hasher::OperatorKeyHasher<CustomOperatorKey> >
      custom_ops_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MUTABLE_OP_RESOLVER_H_
