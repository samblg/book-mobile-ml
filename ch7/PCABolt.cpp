#include "PCABolt.h"
#include "hurricane/util/StringUtil.h"

#include <iostream>
#include <sstream>
#include <cstring>

const int MAX_SIZE = 1024;
const std::string PCA_MODEL_FILE = "sample"

void PCABolt::Prepare(std::shared_ptr<hurricane::collector::OutputCollector> outputCollector) {
    _outputCollector = outputCollector;
    _size = 0;
}

void PCABolt::Cleanup() {
}

std::vector<std::string> PCABolt::DeclareFields() {
    return { "modelFile" };
}

void PCABolt::Execute(const hurricane::base::Tuple& tuple) {
    std::vector<double> record = tuple[0].GetValue<std::vector<double>>();
    _pca.add_record(record);
    _size ++;

    if (_size == MAX_SIZE) {
        _outputCollector->Emit({ newUrl });
        _size = 0;
        _pca.save(PCA_MODEL_FILE);
        _pca = pca();
    }
}

