#ifndef TENSORFLOW_LITE_INTERPRETER_H_
#define TENSORFLOW_LITE_INTERPRETER_H_

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {

template <class T>
constexpr TfLiteType typeToTfLiteType() {
  return kTfLiteNoType;
}
template <>
constexpr TfLiteType typeToTfLiteType<int>() {
  return kTfLiteInt32;
}
template <>
constexpr TfLiteType typeToTfLiteType<int16_t>() {
  return kTfLiteInt16;
}
template <>
constexpr TfLiteType typeToTfLiteType<int64_t>() {
  return kTfLiteInt64;
}
template <>
constexpr TfLiteType typeToTfLiteType<float>() {
  return kTfLiteFloat32;
}
template <>
constexpr TfLiteType typeToTfLiteType<unsigned char>() {
  return kTfLiteUInt8;
}
template <>
constexpr TfLiteType typeToTfLiteType<int8_t>() {
  return kTfLiteInt8;
}
template <>
constexpr TfLiteType typeToTfLiteType<bool>() {
  return kTfLiteBool;
}
template <>
constexpr TfLiteType typeToTfLiteType<std::complex<float>>() {
  return kTfLiteComplex64;
}
template <>
constexpr TfLiteType typeToTfLiteType<string>() {
  return kTfLiteString;
}

class Interpreter {
 public:
  explicit Interpreter(ErrorReporter* error_reporter = DefaultErrorReporter());

  ~Interpreter();

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  TfLiteStatus SetInputs(std::vector<int> inputs);

  TfLiteStatus SetOutputs(std::vector<int> outputs);

  TfLiteStatus SetVariables(std::vector<int> variables);

  void ReserveNodes(int count);

  TfLiteStatus AddNodeWithParameters(const std::vector<int>& inputs,
                                     const std::vector<int>& outputs,
                                     const char* init_data,
                                     size_t init_data_size, void* builtin_data,
                                     const TfLiteRegistration* registration,
                                     int* node_index = nullptr);

  TfLiteStatus AddTensors(int tensors_to_add,
                          int* first_new_tensor_index = nullptr);

  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  inline TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes,
      const Allocation* allocation = nullptr) {
    return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
                                       dims.data(), quantization, buffer, bytes,
                                       allocation);
  }

  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  TfLiteStatus SetTensorParametersReadWrite(int tensor_index, TfLiteType type,
                                            const char* name,
                                            const std::vector<int>& dims,
                                            TfLiteQuantization quantization,
                                            bool is_variable = false);

  TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      bool is_variable = false);

  const std::vector<int>& inputs() const { return primary_subgraph().inputs(); }

  const char* GetInputName(int index) const {
    return context_->tensors[inputs()[index]].name;
  }

  const std::vector<int>& outputs() const {
    return primary_subgraph().outputs();
  }

  const std::vector<int>& variables() const {
    return primary_subgraph().variables();
  }

  const char* GetOutputName(int index) const {
    return context_->tensors[outputs()[index]].name;
  }

  size_t tensors_size() const { return context_->tensors_size; }

  size_t nodes_size() const { return primary_subgraph().nodes_size(); }

  const std::vector<int>& execution_plan() const {
    return primary_subgraph().execution_plan();
  }

  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

  TfLiteTensor* tensor(int tensor_index) {
    return primary_subgraph().tensor(tensor_index);
  }

  const TfLiteTensor* tensor(int tensor_index) const {
    return primary_subgraph().tensor(tensor_index);
  }

  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    return primary_subgraph().node_and_registration(node_index);
  }

  template <class T>
  T* typed_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  template <class T>
  const T* typed_tensor(int tensor_index) const {
    if (const TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<const T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  template <class T>
  T* typed_input_tensor(int index) {
    return typed_tensor<T>(inputs()[index]);
  }

  template <class T>
  const T* typed_input_tensor(int index) const {
    return typed_tensor<T>(inputs()[index]);
  }

  template <class T>
  T* typed_output_tensor(int index) {
    return typed_tensor<T>(outputs()[index]);
  }

  template <class T>
  const T* typed_output_tensor(int index) const {
    return typed_tensor<T>(outputs()[index]);
  }

  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  TfLiteStatus AllocateTensors();

  TfLiteStatus Invoke();

  void UseNNAPI(bool enable);

  void SetNumThreads(int num_threads);

  void SetAllowFp16PrecisionForFp32(bool allow);

  bool GetAllowFp16PrecisionForFp32() const {
    return context_->allow_fp32_relax_to_fp16;
  }

  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
    return primary_subgraph().EnsureTensorDataIsReadable(tensor_index);
  }

  TfLiteStatus SetBufferHandle(int tensor_index,
                               TfLiteBufferHandle buffer_handle,
                               TfLiteDelegate* delegate);

  TfLiteStatus GetBufferHandle(int tensor_index,
                               TfLiteBufferHandle* buffer_handle,
                               TfLiteDelegate** delegate);

  void SetProfiler(profiling::Profiler* profiler);

  profiling::Profiler* GetProfiler();

  static constexpr int kTensorsReservedCapacity = 128;
  static constexpr int kTensorsCapacityHeadroom = 16;

  void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
    allow_buffer_handle_output_ = allow_buffer_handle_output;
  }

  TfLiteStatus ResetVariableTensors();

  const char* OpProfilingString(const TfLiteRegistration& op_reg,
                                const TfLiteNode* node) const {
    if (op_reg.profiling_string == nullptr) return nullptr;
    return op_reg.profiling_string(context_, node);
  }

  void SetExternalContext(TfLiteExternalContextType type,
                          TfLiteExternalContext* ctx);

  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  size_t subgraphs_size() const { return subgraphs_.size(); }

  Subgraph* subgraph(int subgraph_index) {
    if (subgraph_index < 0 ||
        static_cast<size_t>(subgraph_index) >= subgraphs_size())
      return nullptr;
    return &*subgraphs_[subgraph_index];
  }

  Subgraph& primary_subgraph() {
    return *subgraphs_.front();  // Safe as subgraphs_ always has 1 entry.
  }

  const Subgraph& primary_subgraph() const {
    return *subgraphs_.front();  // Safe as subgraphs_ always has 1 entry.
  }

 private:
  friend class InterpreterBuilder;
  friend class InterpreterTest;

  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegatePtr delegate) {
    owned_delegates_.push_back(std::move(delegate));
    return ModifyGraphWithDelegate(owned_delegates_.back().get());
  }

  TfLiteContext* context_;

  ErrorReporter* error_reporter_;

  std::vector<TfLiteDelegatePtr> owned_delegates_;

  bool allow_buffer_handle_output_ = false;

  TfLiteExternalContext* external_contexts_[kTfLiteMaxExternalContexts];

  std::vector<std::unique_ptr<Subgraph>> subgraphs_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERPRETER_H_
