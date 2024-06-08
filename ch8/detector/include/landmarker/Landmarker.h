#pragma once

#include "common/AmType.h"
#include <vector>

namespace authen {
namespace core {
namespace landmarker {

class Landmarker {
public:
    virtual ~Landmarker() {}
    virtual void setParam(AmLandmarkerParam* param) = 0;
    virtual void getLandmark(unsigned char* bgr, int width, int height,
        int pitch, AmFaceRect* rect, AmLandmarkResult* result) = 0;

    virtual void getLandmarksInGroup(std::vector<AmImage> images, int faceCount, AmFaceRect* rect, AmLandmarkResult* result) = 0;

    void cropLandmark(unsigned char* bgr, int width, int height, int pitch, AmFaceRect* rect, AmCroppedResult* result);
    void cropLandmark(unsigned char* imageData, int imageWidth, int imageHeight, AmLandmarkResult* landmarkResult, AmCroppedResult* croppedResult);
};

}
}
}