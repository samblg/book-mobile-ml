#pragma once

#include "extractor/FaceExtractor.h"
#include "common/externc.h"
#include "common/AmType.h"
#include <memory>

namespace authen {
namespace core {
namespace extractor {

class ExtractorEngine;

class WSFaceExtractor : public FaceExtractor {
public:
    WSFaceExtractor();
    WSFaceExtractor(const std::string& packagePath, int deviceId);

    virtual ~WSFaceExtractor() override;

    virtual int extractFeature(const unsigned char* image, int width, int height, int pitch,
        const AmLandmarkResult* mark, unsigned char* feature) override;
    virtual int extractFeature(const AmCroppedResult* result, int fdSize, unsigned char* feature);

private:
    // Disable copy and assignment
    WSFaceExtractor(const WSFaceExtractor&) = delete;
    const WSFaceExtractor& operator=(const WSFaceExtractor&) = delete;

    int _deviceId;
    std::shared_ptr<ExtractorEngine> _extractorEngine;
};

}
}
}