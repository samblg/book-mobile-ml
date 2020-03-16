#ifndef CAFFE_LANDMARKER_H
#define CAFFE_LANDMARKER_H

#include "common/AmType.h"
#include "landmarker/Landmarker.h"

#include <string>
#include <memory>

namespace authen {
namespace core {
namespace landmarker {

class LmProcessor;

class CaffeLandmarker : public Landmarker {
public:
    CaffeLandmarker(const std::string& packagePath, int deviceId);
    ~CaffeLandmarker();
    virtual void setParam(AmLandmarkerParam* param) override;
    virtual void getLandmark(unsigned char* bgr, int width, int height,
        int pitch, AmFaceRect* rect, AmLandmarkResult* result) override;
    virtual void getLandmarksInGroup(std::vector<AmImage> images, int faceCount, AmFaceRect* rect, AmLandmarkResult* result) override;

private:
    std::shared_ptr<LmProcessor> _lmProcessor;
};

}
}
}

#endif
