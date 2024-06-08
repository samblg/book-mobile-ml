#ifndef LM_PROCESSOR_HPP_
#define LM_PROCESSOR_HPP_

#include "caffe/caffe.hpp"

#include <string>
#include <vector>
#include <memory>

class KTimer;

namespace authen {

class AntiProcessor
{
public:
    AntiProcessor(std::string packagePath,
                  const std::string& modelPath,
                  const std::string& weightsPath,
                  const std::string& meanPath);
    ~AntiProcessor();

    void setDeviceId(int deviceId);
    int deviceId() const;

    float getAntiScore(float* imageData, int imageWidth, int imageHeight, int channels);
    float getAntiScoreVideo(float* imageData, int imageWidth, int imageHeight, int channels);

private:
    std::shared_ptr<caffe::Net<float>> anti_net;
    int _deviceId;
    std::shared_ptr<caffe::Blob<float>> data_mean_;
};

} // namespace caffe

#endif
