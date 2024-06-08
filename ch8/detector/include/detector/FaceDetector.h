#pragma once

#include "common/AmType.h"
#include <vector>

namespace authen {
namespace core {
namespace detector {

class FaceDetector {
public:
    static void Resize(unsigned char* src, unsigned char* dst, int srcW, int srcH, int dstW, int dstH, int bpp);

    virtual ~FaceDetector() {}

    virtual void setParam(AmDetectorParam* param) = 0;
    virtual int detectFaces(unsigned char* bgr, int width, int height,
        int pitch, float threshold, int maxCount, AmFaceRect* results) = 0;

    virtual int detectFacesInGroup(std::vector<AmImage> images, float threshold, int maxCount, AmFaceRect* results) = 0;
};

}
}
}
