#ifndef TENSORFLOW_LITE_TOCO_TOCO_CONVERT_H_
#define TENSORFLOW_LITE_TOCO_TOCO_CONVERT_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/toco/args.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

namespace toco {

tensorflow::Status Convert(const string& graph_def_contents,
                           const TocoFlags& toco_flags,
                           const ModelFlags& model_flags,
                           string* output_file_contents);

tensorflow::Status Convert(const ParsedTocoFlags& parsed_toco_flags,
                           const ParsedModelFlags& parsed_model_flags);
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TOCO_CONVERT_H_
