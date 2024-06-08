#pragma once

#ifdef WIN32
#ifdef AMWT_API_EXPORTS
#define AMWT_API __declspec(dllexport)
#else
#define AMWT_API __declspec(dllimport)
#endif
#else
#define AMWT_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

namespace authen {
namespace core {
namespace api {

class AMWT_API AmWTProcessor {
public:
    AmWTProcessor();
    AmWTProcessor(std::string modelPath);
    ~AmWTProcessor();

    std::vector<AmLandmarkResult> detectFaces(const unsigned char* imageData,
                                                   int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    std::vector<AmCroppedResult> detectAndCropFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);

private:
	std::vector<AmLandmarkResult> detectByJQW(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
	std::vector<AmCroppedResult> detectCropByJQW(const unsigned char* imageData, int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

private:
    std::set<Task> _tasks;
	AmDetectorHandle _jqwDetector;
	AmLandmarkerHandle _jqwLandmarker;
};


}
}
}
