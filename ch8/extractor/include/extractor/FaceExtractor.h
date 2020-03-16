#pragma once

#include "common/AmType.h"
#include <string>

namespace authen {
namespace core {
namespace extractor {

class FaceExtractor {
public:
    virtual ~FaceExtractor() {}
    virtual int extractFeature(const unsigned char* image, int width, int height, int pitch,
        const AmLandmarkResult* mark, unsigned char* feature) {
        return 0;
    }
    virtual int extractFeature(const AmCroppedResult* result, int fdSize, unsigned char* feature) = 0;
};

}
}
}