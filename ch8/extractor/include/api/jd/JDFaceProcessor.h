#pragma once

#ifdef WIN32
#ifdef AMJDAPI_EXPORTS
#define AMJDAPI_API __declspec(dllexport)
#else
#define AMJDAPI_API __declspec(dllimport)
#endif
#else
#define AMJDAPI_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class WaterRemover;
class FaceDetector;
class DirectionDetector;
class Landmarker;
class FaceExtractor;
class FaceFeatureMatcher;

namespace authen {
namespace core {
namespace api {

class JDWaterRemovalResult;
class JDLandmarkResult;

class AMJDAPI_API JDFaceProcessor {
public:
    enum class Task {
        RemoveWater,
        Detect,
        Extract,
        Match
    };

    JDFaceProcessor(int extractDeviceId = 0);
    JDFaceProcessor(std::set<Task> tasks, int extractDeviceId = 0);
    ~JDFaceProcessor();

    JDWaterRemovalResult removeWater(unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch);
    std::vector<AmLandmarkResult> detectIdFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    JDLandmarkResult detectSceneFaces(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

    std::vector<AmCroppedResult> detectAndCropIdFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmCroppedResult> detectAndCropSceneFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

    std::string extractFeatures(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, const AmLandmarkResult& landmarkResult);
    std::vector<std::string> extractFeatures(const std::vector<AmCroppedResult>& croppedResults);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);

    float slowMatch(const std::string& feature1, const std::string& feature2);
    float normalMatch(const std::string& feature1, const std::string& feature2);
    float idCaptureMatch(const std::string& id_feature, const std::string& capture_feature);

    float normalMatchDebug(unsigned char* feature1, unsigned char* feature2);
    float idCaptureMatchDebug(unsigned char* id_feature, unsigned char* capture_feature);
    float getScoreDebug(unsigned char* capture_feature);

    bool hasTask(Task task) {
        return _tasks.find(task) != _tasks.end();
    }

private:
    AmFaceDirection detectDirection(const unsigned char* imageData,
                                    int imageWidth, int imageHeight, int imagePitch);

    std::vector<AmLandmarkResult> detectByFR(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmCroppedResult> detectCropByFR(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
	std::vector<AmLandmarkResult> detectByJQW(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
	std::vector<AmCroppedResult> detectCropByJQW(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    
	std::vector<AmLandmarkResult> detectAndFaces_Inner(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);
    std::vector<AmCroppedResult> detectAndCropFaces_Inner(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

private:
    std::set<Task> _tasks;
    WaterRemover* _caffeWaterRemover;
    WaterRemover* _oldWaterRemover;
    FaceDetector* _frDetector;
    FaceDetector* _jqwDetector;
    DirectionDetector* _caffeDirectionDetector;
    Landmarker* _sdmFRLandmarker;
    Landmarker* _esrJQWLandmarker;
    FaceExtractor* _wsExtractor;
    FaceFeatureMatcher* _mlpMatcher;
    FaceFeatureMatcher* _normalMatcher;
};

class JDImage {
public:
    JDImage() :
            _width(0), _height(0) {
    }

    JDImage(const unsigned char* data, int width, int height, int channels) :
            _width(width), _height(height), _channels(channels),
            _data(reinterpret_cast<const char*>(data), width * height * channels){
    }

    int channels() const {
        return _channels;
    }

    int width() const {
        return _width;
    }

    int height() const {
        return _height;
    }

    int pitch() const {
        return _width * _channels;
    }

    int size() const {
        return _data.size();
    }

    const std::string& buffer() const {
        return _data;
    }

    const unsigned char* data() const {
        return reinterpret_cast<const unsigned char*>(_data.c_str());
    }

    unsigned char* data() {
        return reinterpret_cast<unsigned char*>(const_cast<char*>(_data.c_str()));
    }

private:
    int _channels;
    int _width;
    int _height;
    std::string _data;
};

class JDLandmarkResult {
public:
    enum class Direction {
        Up,
        Right,
        Down,
        Left
    };

    JDLandmarkResult() : _direction(Direction::Up) {}

    Direction getDirection() const {
        return _direction;
    }

    void setDirection(Direction direction) {
        _direction = direction;
    }

    const std::vector<AmLandmarkResult>& getLandmarkResults() const {
        return _landmarkResults;
    }

    void setLandmarkResults(const std::vector<AmLandmarkResult>& landmarkResults) {
        _landmarkResults = landmarkResults;
    }

    const JDImage& getFace() const {
        return _face;
    }

    void setFace(const JDImage& face) {
        _face = face;
    }

    void freeLandmarkResults() {
        for (AmLandmarkResult& landmarkResult : _landmarkResults) {
            if (landmarkResult.landmarks) {
                delete[] landmarkResult.landmarks;
                landmarkResult.landmarks = nullptr;
            }
        }
    }

private:
    Direction _direction;
    std::vector<AmLandmarkResult> _landmarkResults;
    JDImage _face;
};

class JDWaterRemovalResult {
public:
    JDWaterRemovalResult() :
            _width(0), _height(0) {
    }

    JDWaterRemovalResult(const AmWaterRemovalResult& result) :
            _width(result.width), _height(result.height), _channels(result.channels),
            _data(reinterpret_cast<char*>(result.data), result.width * result.height * result.channels){
    }

    JDWaterRemovalResult(const unsigned char* data, int width, int height, int channels) :
            _width(width), _height(height), _channels(channels),
            _data(reinterpret_cast<const char*>(data), width * height * channels){
    }

    int channels() const {
        return _channels;
    }

    int width() const {
        return _width;
    }

    int height() const {
        return _height;
    }

    int pitch() const {
        return _width * _channels;
    }

    int size() const {
        return _data.size();
    }

    const std::string& buffer() const {
        return _data;
    }

    const unsigned char* data() const {
        return reinterpret_cast<const unsigned char*>(_data.c_str());
    }

    unsigned char* data() {
        return reinterpret_cast<unsigned char*>(const_cast<char*>(_data.c_str()));
    }

private:
    int _channels;
    int _width;
    int _height;
    std::string _data;
};

}
}
}
