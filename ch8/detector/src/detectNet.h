#include "MlType.h"
#include "net.h"

namespace learnml {

class DetectNet
{
public:
    DetectNet();
    DetectNet(const char* modelPath);

	int SetParam(MlDetectorParam* param);
    int DetectObjectRects(unsigned char* imageData, int width, int height, int pitch, int bpp,
        float threshold, int maxCount, MlObjectRect* objectRects);
    int DetectObjectByRect(unsigned char* imageData, int width, int height, int pitch, int bpp,
        MlObjectRect* objectRect, MlLandmarkResult* objectLandMarkResult);

    int DetectObjects(unsigned char* imageData, int width, int height, int pitch, int bpp,
        float threshold, int maxCount, MlLandmarkResult* objectLandMarkResults);

private:
    Net* detectNet;
    Net* landmarkNet;
    Mat landmark_mean;

	float _scale;
};

}
