#ifndef LM_PROCESSOR_HPP_
#define LM_PROCESSOR_HPP_

#include "caffe/caffe.hpp"

#include <string>
#include <vector>

class KTimer;

namespace authen {
namespace core {
namespace landmarker {

class LmProcessor
{
public:
    LmProcessor(const std::string& packagePath, 
        const std::string& modelPath, const std::string& weightsPath, const std::string& meanPath,
        int deviceId);
    ~LmProcessor();

    void setDeviceId(int deviceId);
    int deviceId() const;

    std::vector<float> feaExtractionMat(float* imageData, int numImage, int imageWidth, int imageHeight, float *fea, KTimer* timer = nullptr);
    std::vector<float> getPTScore(float* imageData, int numImage, int imageWidth, int imageHeight);
    std::vector<float> getPoseScore(float* imageData, int numImage, int imageWidth, int imageHeight);

private:
    std::shared_ptr<caffe::Net<float>> _caffeNet;
    std::shared_ptr<caffe::Net<float>> _ptscoreNet;
    std::shared_ptr<caffe::Net<float>> _poseNet;
    std::shared_ptr<caffe::Blob<float>> _dataMean;
    int _deviceId;
};

}
}
}

#endif
