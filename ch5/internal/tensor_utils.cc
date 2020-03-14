#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/common.h"

#ifndef USE_NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#endif  //  defined(__ARM_NEON__) || defined(__ARM_NEON)
#endif  //  USE_NEON

#ifdef USE_NEON
#include "tensorflow/lite/kernels/internal/optimized/neon_tensor_utils.h"
#else
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils.h"
#endif  // USE_NEON
