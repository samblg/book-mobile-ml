#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include "common/caffe-fast-rcnn/caffe/caffe.hpp"
#include "common/caffe-fast-rcnn/caffe/detect_util.h"
#include <memory>

#include <string>
#include <vector>

//typedef struct feaData
//{
//	float fea[256];
//	string name;
//}feaData;

class KTimer;

namespace authen {
namespace core {
namespace detector {

class SSDProcessor
{
public:
    SSDProcessor(const std::string& packagePath, 
        const std::string& modelPath, const std::string& weightsPath, 
        int deviceId);
    ~SSDProcessor();

    void setDeviceId(int deviceId);
    int deviceId() const;

    std::vector<caffe::bbox_fu> feaExtractionMat(
        float* imageData, int imageWidth, int imageHeight, float *fea, int minObjSize, KTimer* timer = nullptr);
    std::vector<caffe::bbox_fu_nms> feaExtractionMatIngroup(
        float* imageData_group, int numImage, int imageWidth, int imageHeight, int minObjSize);

private:
    std::shared_ptr<caffe::Net<float>> _net;
    int _deviceId;
};
}
}
}

#endif
