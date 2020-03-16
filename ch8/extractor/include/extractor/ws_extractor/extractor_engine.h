#pragma once

#include "cropImage.h"
#include <vector>

#define PATCH_FEATURE_LENGTH 256  //v1.3

namespace authen {
namespace core {
namespace extractor {

class ExtractorProcessor;

class ExtractorEngine
{
public:
    ExtractorEngine(const std::string& packagePath, int deviceid);
    ~ExtractorEngine();

    void extractFeature(unsigned char* imageData, int imageWidth, int imageHeight, int imageChannel, unsigned char* feature, const TLandmarks1* lms);
    void extractFeature(
        std::vector<const unsigned char*> images,
        std::vector<TLandmarks1> landmarks, unsigned char* features);

private:
    std::vector<ExtractorProcessor> _processors;
    int _deviceId;
    int _isMirror;
    std::vector<int> _partIds;
};

}
}
}