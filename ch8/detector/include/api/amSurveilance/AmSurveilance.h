#pragma once

#ifdef WIN32
#ifdef AMSURVEILANCE_API_EXPORTS
#define AMSURVEILANCE_API __declspec(dllexport)
#else
#define AMSURVEILANCE_API __declspec(dllimport)
#endif
#else
#define AMSURVEILANCE_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class FaceDetector;
class Landmarker;
class FaceExtractor;
class FaceFeatureMatcher;

namespace authen {
namespace core {
namespace api {

class AMSURVEILANCE_API AmSurveilance {
public:
    enum class Task {
        RemoveWater,
        Detect,
        Extract,
        Match
    };

    AmSurveilance(int extractDeviceId = 0);
    AmSurveilance(std::set<Task> tasks, int extractDeviceId = 0);
    ~AmSurveilance();

    std::vector<AmLandmarkResult> detectFaces(const unsigned char* imageData,
                                                   int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    std::vector<AmCroppedResult> detectAndCropFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

    std::string extractFeatures(const unsigned char* imageData,
                                int imageWidth, int imageHeight, int imagePitch, const AmLandmarkResult& landmarkResult);
    std::vector<std::string> extractFeatures(const std::vector<AmCroppedResult>& croppedResults);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);
    float normalMatch(const std::string& feature1, const std::string& feature2);

    bool hasTask(Task task);

private:
	std::vector<AmLandmarkResult> detectByJQW(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
	std::vector<AmCroppedResult> detectCropByJQW(const unsigned char* imageData, int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

private:
    std::set<Task> _tasks;
    FaceDetector* _jqwDetector;

    Landmarker* _jqwLandmarker;
    FaceExtractor* _wsExtractor;
    FaceFeatureMatcher* _normalMatcher;
};


}
}
}
