#include "detector/FaceDetector.h"
#include <memory>
#include <string>

namespace authen {
namespace core {
namespace detector {

class SSDProcessor;

class CaffeFastRCNNFaceDetector : public FaceDetector {
public:
    CaffeFastRCNNFaceDetector(const std::string& packagePath, int deviceId);
    virtual ~CaffeFastRCNNFaceDetector();

    virtual void setParam(AmDetectorParam* param);
    virtual int detectFaces(unsigned char* bgr, int width, int height,
        int pitch, float threshold, int maxCount, AmFaceRect* results);

    virtual int detectFacesInGroup(std::vector<AmImage> images, float threshold, int maxCount, AmFaceRect* results);

private:
    std::shared_ptr<SSDProcessor> _processor;
    float _minObjectSize;
    float _scaleFactor;
    int _deviceId;
};
}

}
}
