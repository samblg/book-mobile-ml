#ifndef TENSORFLOW_LITE_TOCO_TFLITE_IMPORT_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_IMPORT_H_

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/toco/model.h"

namespace toco {

namespace tflite {

std::unique_ptr<Model> Import(const ModelFlags &model_flags,
                              const string &input_file_contents);

namespace details {

using TensorsTable = std::vector<string>;

using OperatorsTable = std::vector<string>;

void LoadTensorsTable(const ::tflite::Model &input_model,
                      TensorsTable *tensors_table);
void LoadOperatorsTable(const ::tflite::Model &input_model,
                        OperatorsTable *operators_table);

}  // namespace details
}  // namespace tflite

}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_IMPORT_H_
