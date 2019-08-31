#include <cstdio>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_cmdline_flags.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_tooling.h"
#include "tensorflow/lite/toco/toco_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {

void CheckOutputFilePermissions(const Arg<string>& output_file) {
  QCHECK(output_file.specified()) << "Missing required flag --output_file.\n";
  QCHECK(port::file::Writable(output_file.value()).ok())
      << "Specified output_file is not writable: " << output_file.value()
      << ".\n";
}

void CheckFrozenModelPermissions(const Arg<string>& input_file) {
  QCHECK(input_file.specified()) << "Missing required flag --input_file.\n";
  QCHECK(port::file::Exists(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file does not exist: " << input_file.value() << ".\n";
  QCHECK(port::file::Readable(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file exists, but is not readable: "
      << input_file.value() << ".\n";
}

void ReadInputData(const ParsedTocoFlags& parsed_toco_flags,
                   const ParsedModelFlags& parsed_model_flags,
                   TocoFlags* toco_flags, ModelFlags* model_flags,
                   string* graph_def_contents) {
  port::CheckInitGoogleIsDone("InitGoogle is not done yet.\n");

  QCHECK(!parsed_toco_flags.savedmodel_directory.specified())
      << "Use `tensorflow/lite/python/tflite_convert` script with "
      << "SavedModel directories.\n";

  CheckFrozenModelPermissions(parsed_toco_flags.input_file);
  CHECK(port::file::GetContents(parsed_toco_flags.input_file.value(),
                                graph_def_contents, port::file::Defaults())
            .ok());
}
}  // namespace

tensorflow::Status Convert(const string& graph_def_contents,
                           const TocoFlags& toco_flags,
                           const ModelFlags& model_flags,
                           string* output_file_contents) {
  std::unique_ptr<Model> model =
      Import(toco_flags, model_flags, graph_def_contents);
  Transform(toco_flags, model.get());
  return Export(toco_flags, *model, toco_flags.allow_custom_ops(),
                output_file_contents);
}

tensorflow::Status Convert(const ParsedTocoFlags& parsed_toco_flags,
                           const ParsedModelFlags& parsed_model_flags) {
  ModelFlags model_flags;
  ReadModelFlagsFromCommandLineFlags(parsed_model_flags, &model_flags);

  TocoFlags toco_flags;
  ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags, &toco_flags);

  string graph_def_contents;
  ReadInputData(parsed_toco_flags, parsed_model_flags, &toco_flags,
                &model_flags, &graph_def_contents);
  CheckOutputFilePermissions(parsed_toco_flags.output_file);

  string output_file_contents;
  TF_RETURN_IF_ERROR(Convert(graph_def_contents, toco_flags, model_flags,
                             &output_file_contents));

  TF_RETURN_IF_ERROR(
      port::file::SetContents(parsed_toco_flags.output_file.value(),
                              output_file_contents, port::file::Defaults()));
  return tensorflow::Status();
}

}  // namespace toco
