#ifndef TENSORFLOW_LITE_CORE_SUBGRAPH_H_
#define TENSORFLOW_LITE_CORE_SUBGRAPH_H_

#include <cstdlib>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/util.h"

namespace tflite {

class NNAPIDelegate;

class Subgraph {
 public:
  friend class Interpreter;

  Subgraph(ErrorReporter* error_reporter,
           TfLiteExternalContext** external_contexts,
           std::vector<std::unique_ptr<Subgraph>>* subgraphs);

  Subgraph(const Subgraph&) = delete;

  Subgraph(Subgraph&&) = default;
  Subgraph& operator=(const Subgraph&) = delete;
  virtual ~Subgraph();

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

  inline TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      const char* buffer, size_t bytes,
      const Allocation* allocation = nullptr) {
    return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
                                       dims.data(), quantization, buffer, bytes,
                                       allocation);
  }
  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantization quantization, const char* buffer,
      size_t bytes, const Allocation* allocation = nullptr);

  inline TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      bool is_variable = false) {
    return SetTensorParametersReadWrite(tensor_index, type, name, dims.size(),
                                        dims.data(), quantization, is_variable);
  }
  TfLiteStatus SetTensorParametersReadWrite(int tensor_index, TfLiteType type,
                                            const char* name, const size_t rank,
                                            const int* dims,
                                            TfLiteQuantization quantization,
                                            bool is_variable = false);

  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

  TfLiteTensor* tensor(int tensor_index) {
    if (tensor_index < 0 ||
        static_cast<size_t>(tensor_index) >= context_->tensors_size) {
      return nullptr;
    }
    return &context_->tensors[tensor_index];
  }

  const TfLiteTensor* tensor(int tensor_index) const {
    if (tensor_index < 0 ||
        static_cast<size_t>(tensor_index) >= context_->tensors_size) {
      return nullptr;
    }
    return &context_->tensors[tensor_index];
  }

  std::vector<int>& inputs() { return inputs_; }

  const std::vector<int>& inputs() const { return inputs_; }

  std::vector<int>& outputs() { return outputs_; }

  const std::vector<int>& outputs() const { return outputs_; }

  std::vector<int>& variables() { return variables_; }

  const std::vector<int>& variables() const { return variables_; }

  size_t tensors_size() const { return tensors_.size(); }

  size_t nodes_size() const { return nodes_and_registration_.size(); }

  std::vector<int>& execution_plan() { return execution_plan_; }

  const std::vector<int>& execution_plan() const { return execution_plan_; }

  std::vector<TfLiteTensor>& tensors() { return tensors_; }
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>>&
  nodes_and_registration() {
    return nodes_and_registration_;
  }

  const std::vector<std::pair<TfLiteNode, TfLiteRegistration>>&
  nodes_and_registration() const {
    return nodes_and_registration_;
  }

  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    if (node_index < 0 || static_cast<size_t>(node_index) >= nodes_size())
      return nullptr;
    return &nodes_and_registration_[node_index];
  }

  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  TfLiteStatus AllocateTensors();

  TfLiteStatus Invoke();

  void ReportError(const char* format, ...);

  void UseNNAPI(bool enable);

  TfLiteContext* context() { return context_; }

  void SetExternalContext(TfLiteExternalContextType type,
                          TfLiteExternalContext* ctx);
  bool GetAllowFp16PrecisionForFp32() const {
    return context_->allow_fp32_relax_to_fp16;
  }

  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
    TfLiteTensor* t = &tensors_[tensor_index];
    TF_LITE_ENSURE(context_, t != nullptr);
    if (t->data_is_stale) {
      TF_LITE_ENSURE(context_, t->delegate != nullptr);
      TF_LITE_ENSURE(context_, t->buffer_handle != kTfLiteNullBufferHandle);
      TF_LITE_ENSURE(context_, t->delegate->CopyFromBufferHandle != nullptr);
      // TODO(b/120420546): we must add a test that exercise this code.
      TF_LITE_ENSURE_STATUS(t->delegate->CopyFromBufferHandle(
          context_, t->delegate, t->buffer_handle, t));
      t->data_is_stale = false;
    }
    return kTfLiteOk;
  }

  static constexpr int kTensorsReservedCapacity = 128;
  static constexpr int kTensorsCapacityHeadroom = 16;

  TfLiteStatus ResetVariableTensors();

  void SetProfiler(profiling::Profiler* profiler) {
    profiler_ = profiler;
    context_->profiler = profiler;
  }

  profiling::Profiler* GetProfiler() { return profiler_; }

  std::vector<std::unique_ptr<Subgraph>>* GetSubgraphs() { return subgraphs_; }

  bool HasDynamicTensors() { return has_dynamic_tensors_; }

 private:
  void SwitchToKernelContext();

  void SwitchToDelegateContext();

  void* OpInit(const TfLiteRegistration& op_reg, const char* buffer,
               size_t length) {
    if (op_reg.init == nullptr) return nullptr;
    return op_reg.init(context_, buffer, length);
  }

  void OpFree(const TfLiteRegistration& op_reg, void* buffer) {
    if (op_reg.free == nullptr) return;
    if (buffer) {
      op_reg.free(context_, buffer);
    }
  }
  i
  TfLiteStatus OpPrepare(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    if (op_reg.prepare == nullptr) return kTfLiteOk;
    return op_reg.prepare(context_, node);
  }

  TfLiteStatus OpInvoke(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    if (op_reg.invoke == nullptr) return kTfLiteError;
    return op_reg.invoke(context_, node);
  }

  TfLiteStatus PrepareOpsAndTensors();

  TfLiteStatus PrepareOpsStartingAt(int first_execution_plan_index,
                                    int* last_execution_plan_index_prepared);

  std::vector<TfLiteTensor> tensors_;

  TfLiteStatus CheckTensorIndices(const char* label, const int* indices,
                                  int length);

  TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                             size_t* bytes);

  TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor, TfLiteIntArray* new_size);

  void ReportErrorImpl(const char* format, va_list args);

  static TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
                                   TfLiteIntArray* new_size);
  static void ReportErrorC(TfLiteContext* context, const char* format, ...);

  static TfLiteStatus AddTensors(TfLiteContext* context, int tensors_to_add,
                                 int* first_new_tensor_index);

  static TfLiteStatus ReplaceNodeSubsetsWithDelegateKernels(
      TfLiteContext* context, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  TfLiteStatus ReplaceNodeSubsetsWithDelegateKernels(
      TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegate* delegate);

  TfLiteStatus GetNodeAndRegistration(int node_index, TfLiteNode** node,
                                      TfLiteRegistration** registration);

  static TfLiteStatus GetNodeAndRegistration(struct TfLiteContext*,
                                             int node_index, TfLiteNode** node,
                                             TfLiteRegistration** registration);

  TfLiteStatus GetExecutionPlan(TfLiteIntArray** execution_plan);

  static TfLiteStatus GetExecutionPlan(struct TfLiteContext* context,
                                       TfLiteIntArray** execution_plan);

  TfLiteExternalContext* GetExternalContext(TfLiteExternalContextType type);
  static TfLiteExternalContext* GetExternalContext(
      struct TfLiteContext* context, TfLiteExternalContextType type);

  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  void EnsureTensorsVectorCapacity() {
    const size_t required_capacity = tensors_.size() + kTensorsCapacityHeadroom;
    if (required_capacity > tensors_.capacity()) {
      tensors_.reserve(required_capacity);
      context_->tensors = tensors_.data();
    }
  }

  enum State {
    kStateUninvokable = 0,
    kStateInvokable,
    kStateInvokableAndImmutable,
  };
  State state_ = kStateUninvokable;

  TfLiteContext owned_context_;
  TfLiteContext* context_;

  std::vector<std::pair<TfLiteNode, TfLiteRegistration>>
      nodes_and_registration_;

  bool consistent_ = true;

  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<int> variables_;
  ErrorReporter* error_reporter_;

  int next_execution_plan_index_to_prepare_;
  std::vector<int> execution_plan_;
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> plan_cache_;
  std::unique_ptr<NNAPIDelegate> nnapi_delegate_;
  std::unique_ptr<MemoryPlanner> memory_planner_;
  bool tensor_resized_since_op_invoke_ = false;
  TfLiteExternalContext** external_contexts_;
  profiling::Profiler* profiler_ = nullptr;

  std::vector<std::unique_ptr<Subgraph>>* subgraphs_ = nullptr;

  bool has_dynamic_tensors_ = true;

  bool (*check_cancelled_func_)(void*) = nullptr;
  void* cancellation_data_ = nullptr;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_CORE_SUBGRAPH_H_
