#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include "tensorflow/lite/c/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

float Clip(float f, float abs_limit);

bool IsZeroVector(const float* vector, int v_size);

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor);

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result,
                                         int result_stride);

void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const float* vector, int n_batch, float* result, int result_stride);

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride);

void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride);

void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result);

void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result);

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size);

void BatchVectorBatchVectorDotProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      int n_batch, float* result,
                                      int result_stride);

void VectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                   const float* batch_vector, int n_batch,
                                   float* result);

void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result);

void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector);

void VectorBatchVectorAssign(const float* vector, int v_size, int n_batch,
                             float* batch_vector);

void ApplySigmoidToVector(const float* vector, int v_size, float* result);

void ApplyActivationToVector(const float* vector, int v_size,
                             TfLiteFusedActivation activation, float* result);

void CopyVector(const float* vector, int v_size, float* result);

void Sub1Vector(const float* vector, int v_size, float* result);

void ZeroVector(float* vector, int v_size);

void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result);

void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result);

void VectorShiftLeft(float* vector, int v_size, float shift_value);

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size);

void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch,
                             float normalization_epsilon);
}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
