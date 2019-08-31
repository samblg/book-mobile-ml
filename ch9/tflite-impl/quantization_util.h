#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_

#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
template <typename T>
QuantizationParams ChooseQuantizationParams(double rmin, double rmax,
                                            bool narrow_range) {
  const T qmin = std::numeric_limits<T>::min() + (narrow_range ? 1 : 0);
  const T qmax = std::numeric_limits<T>::max();
  const double qmin_double = qmin;
  const double qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  TFLITE_CHECK_LE(rmin, 0.);
  TFLITE_CHECK_GE(rmax, 0.);
  if (rmin == rmax) {
    // Special case where the min,max range is a point. Should be {0}.
    TFLITE_CHECK_EQ(rmin, 0.);
    TFLITE_CHECK_EQ(rmax, 0.);
    QuantizationParams quantization_params;
    quantization_params.zero_point = 0;
    quantization_params.scale = 0.;
    return quantization_params;
  }

  // General case.
  const double scale = (rmax - rmin) / (qmax_double - qmin_double);

  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);

  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(round(zero_point_double));
  }
  // The zero point should always be in the range of quantized value,
  // [qmin, qmax].
  TFLITE_CHECK_GE(nudged_zero_point, qmin);
  TFLITE_CHECK_LE(nudged_zero_point, qmax);

  // Finally, store the result nudged quantization params.
  QuantizationParams quantization_params;
  quantization_params.zero_point = nudged_zero_point;
  quantization_params.scale = scale;
  return quantization_params;
}

template <typename T>
QuantizationParams ChooseQuantizationParams(double rmin, double rmax) {
  return ChooseQuantizationParams<T>(rmin, rmax, false);
}

template <class IntOut, class FloatIn>
IntOut SafeCast(FloatIn x) {
  static_assert(!std::numeric_limits<FloatIn>::is_integer,
                "FloatIn is integer");
  static_assert(std::numeric_limits<IntOut>::is_integer,
                "IntOut is not integer");
  static_assert(std::numeric_limits<IntOut>::radix == 2, "IntOut is base 2");

  // Special case NaN, for which the logic below doesn't work.
  if (std::isnan(x)) {
    return 0;
  }

  // Negative values all clip to zero for unsigned results.
  if (!std::numeric_limits<IntOut>::is_signed && x < 0) {
    return 0;
  }

  // Handle infinities.
  if (std::isinf(x)) {
    return x < 0 ? std::numeric_limits<IntOut>::min()
                 : std::numeric_limits<IntOut>::max();
  }

  int exp = 0;
  std::frexp(x, &exp);

  if (exp <= std::numeric_limits<IntOut>::digits) {
    return x;
  }

  // Handle numbers with magnitude >= 2^N.
  return x < 0 ? std::numeric_limits<IntOut>::min()
               : std::numeric_limits<IntOut>::max();
}

// Restricted to the case where the multiplier < 1 (and non-negative).
void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift);

// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift);

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);

int64_t IntegerFrExp(double input, int* shift);

double DoubleFromFractionAndShift(int64_t fraction, int shift);

double IntegerDoubleMultiply(double a, double b);

int IntegerDoubleCompare(double a, double b);

void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift);
void PreprocessLogSoftmaxScalingExp(double beta, double input_scale,
                                    int input_integer_bits,
                                    int32_t* quantized_multiplier,
                                    int* left_shift,
                                    int32_t* reverse_scaling_divisor,
                                    int* reverse_scaling_left_shift);
int CalculateInputRadius(int input_integer_bits, int input_left_shift);

void NudgeQuantizationRange(const float min, const float max,
                            const int quant_min, const int quant_max,
                            float* nudged_min, float* nudged_max,
                            float* nudged_scale);

void FakeQuantizeArray(const float nudged_scale, const float nudged_min,
                       const float nudged_max, const float* input_data,
                       float* output_data, const float size);

bool CheckedLog2(const float x, int* log2_result);

void QuantizeMultiplierArray(const double* effective_scales, size_t size,
                             int32_t* effective_scale_significand,
                             int* effective_shift);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_QUANTIZATION_UTIL_H_
