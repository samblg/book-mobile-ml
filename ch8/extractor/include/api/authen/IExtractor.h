#pragma once

#ifdef WIN32
#ifdef IEXTRACTOR_EXPORTS
#define IEXTRACTOR_API __declspec(dllexport)
#else
#define IEXTRACTOR_API __declspec(dllimport)
#endif
#else
#define IEXTRACTOR_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>
#include <memory>

class FaceExtractor;
class JBProjector;

namespace authen {
namespace core {
namespace api {

class IEXTRACTOR_API IExtractor {
public:
    static int checkEnvironment(std::string modelPath, std::string product_code);

    IExtractor(std::string modelPath, std::string product_code, int extractDeviceId = 0);

    ~IExtractor();

    std::string extractFeatures(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, const AmLandmarkResult& landmarkResult);

    std::string extractFeatures_Inner(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, const AmLandmarkResult& landmarkResult);

	//ONLY for GPU
    std::vector<std::string> extractFeatures(const std::vector<AmCroppedResult>& croppedResults);

    //ONLY for GPU
    std::vector<std::string> extractFeatures_Inner(const std::vector<AmCroppedResult>& croppedResults);

    //Merge Features
    std::string mergeFeatures(std::vector<std::string> innerFeatures);

    static float normalMatch(const std::string& feature1, const std::string& feature2);

private:
    void _projectByJB(float* dfea, float* jfea, int faceCount = 0);

private:
    FaceExtractor* _wsExtractor;
    std::shared_ptr<JBProjector> _projector;
    int _extractDeviceId;
};


}
}
}
