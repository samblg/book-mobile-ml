#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

TfLiteDelegate* NnApiDelegate();
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
