#ifndef EXTRACTOR_PROCESSOR_HPP_
#define EXTRACTOR_PROCESSOR_HPP_

#include "common/caffe-fast-rcnn/caffe/caffe.hpp"
#include "minicv/minicv.h"

#include <string>
#include <vector>
#include <memory>

using std::string;

class KTimer;

namespace authen {
namespace resunpack {
    class Package;
}

namespace core {
namespace extractor {

class ExtractorProcessor
{
public:
    ExtractorProcessor(){};
    ExtractorProcessor(
        const std::string& packagePath, const std::string&  modelPath, const std::string&  weightsPath,
        float mean_value, float scale, int deviceId);
    ~ExtractorProcessor();

    void extractFeature(std::vector<minicv::Mat> images, const std::string& LayerName,
        std::vector<std::string>& features, int featureSize);

private:
    void readMeanData(resunpack::Package& package, const std::string& meanPath);

private:
    std::shared_ptr<caffe::Net<float>> _net;
    std::shared_ptr<caffe::Blob<float>> data_mean_;
    float mean_value;
    float scale;
    int _deviceId;
};

}
}
}

#endif
