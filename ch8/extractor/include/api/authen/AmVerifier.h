#pragma once

#ifdef WIN32
#ifdef AMVERIFIERAPI_EXPORTS
#define AMVERIFIERAPI_API __declspec(dllexport)
#else
#define AMVERIFIERAPI_API __declspec(dllimport)
#endif
#else
#define AMVERIFIERAPI_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>
#include <memory>

class WaterRemover;
class FaceDetector;
class DirectionDetector;
class Landmarker;
class FaceExtractor;
class JBProjector;

namespace authen {
namespace core {
namespace api {

class AmVerifyWaterRemovalResult;
class AmVerifyDetectResult;
class AmVerifyLandmarkResult;

//#define AM_VERIFIER_FEATURE_LENGTH 256 * 2 * 3 * 4

class AMVERIFIERAPI_API AmVerifier {
public:
    enum class Task {
        RemoveWater,
        Detect,
        Extract,
        Match
    };

    static int checkEnvironment(std::string modelPath);

    AmVerifier(std::string modelPath, int extractDeviceId = 0);
    AmVerifier(std::string modelPath, std::set<Task> tasks, int extractDeviceId = 0);

	void setDetectParams(AmDetectorParam params);

    ~AmVerifier();

	AmVerifyWaterRemovalResult removeWater(unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch);

    AmVerifyDetectResult detectFaceRects(
            const unsigned char* imageData, int imageWidth, int imageHeight, int imagePitch,
            float threshold, int maxFaceCount, bool enableRotate = true);
	std::vector<AmLandmarkResult> detectFaceLandMarksByRect(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, std::vector<AmFaceRect> faceRects);

    AmVerifyLandmarkResult detectFaces(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);

    //Only for Surveilance
    AmVerifyDetectResult detectFaceRects_Surveilance(
        const unsigned char* imageData, int imageWidth, int imageHeight, int imagePitch,
        float threshold, int maxFaceCount);
    std::vector<AmLandmarkResult> detectFaceLandMarksByRect_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, std::vector<AmFaceRect> faceRects);
    AmVerifyLandmarkResult detectFaces_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

	//ONLY for GPU
    std::vector<AmCroppedResult> detectAndCropFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);
    std::vector<AmCroppedResult> detectAndCropFaces_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);


    std::string extractFeatures(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, const AmLandmarkResult& landmarkResult);

	//ONLY for GPU
    std::vector<std::string> extractFeatures(const std::vector<AmCroppedResult>& croppedResults);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);

    float normalMatch(const std::string& feature1, const std::string& feature2);

    bool hasTask(Task task);

private:
    AmFaceDirection detectDirection(const unsigned char* imageData,
                                    int imageWidth, int imageHeight, int imagePitch);

	std::vector<AmFaceRect> detectFaceRectByJQW(const unsigned char* imageData,
		int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmLandmarkResult> detectByJQW(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmCroppedResult> detectCropByJQW(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    std::vector<AmFaceRect> detectFaceRectByFR(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmLandmarkResult> detectByFR(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmCroppedResult> detectCropByFR(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    void projectByJB(float* dfea, float* jfea, int faceCount = 0);
    
private:
    std::set<Task> _tasks;
    WaterRemover* _caffeWaterRemover;
    WaterRemover* _oldWaterRemover;
    FaceDetector* _jqwDetector;
    DirectionDetector* _caffeDirectionDetector;
    Landmarker* _jqwLandmarker;

    FaceDetector* _frDetector;
    Landmarker* _zhuLandmarker;

    FaceExtractor* _wsExtractor;

    std::shared_ptr<JBProjector> _projector;
};

class AmVerifyImage {
public:
    AmVerifyImage() :
            _width(0), _height(0) {
    }

    AmVerifyImage(const unsigned char* data, int width, int height, int channels) :
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

class AmVerifyDetectResult {
public:
    enum class Direction {
        Up,
        Right,
        Down,
        Left
    };

    AmVerifyDetectResult() : _direction(Direction::Up) {}

    Direction getDirection() const {
        return _direction;
    }

    void setDirection(Direction direction) {
        _direction = direction;
    }

    const std::vector<AmFaceRect>& getFaceRects() const {
        return _faceRects;
    }

    void setFaceRects(const std::vector<AmFaceRect>& faceRects) {
        _faceRects = faceRects;
    }

    const AmVerifyImage& getFace() const {
        return _face;
    }

    void setFace(const AmVerifyImage& face) {
        _face = face;
    }

private:
    Direction _direction;
    std::vector<AmFaceRect> _faceRects;
    AmVerifyImage _face;
};

class AmVerifyLandmarkResult {
public:
    enum class Direction {
        Up,
        Right,
        Down,
        Left
    };

    AmVerifyLandmarkResult() : _direction(Direction::Up) {}

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

    const AmVerifyImage& getFace() const {
        return _face;
    }

    void setFace(const AmVerifyImage& face) {
        _face = face;
    }

    void freeLandmarkResults()
    {
    }

private:
    Direction _direction;
    std::vector<AmLandmarkResult> _landmarkResults;
    AmVerifyImage _face;
};

class AmVerifyWaterRemovalResult {
public:
	AmVerifyWaterRemovalResult() :
            _width(0), _height(0) {
    }

	AmVerifyWaterRemovalResult(const AmWaterRemovalResult& result) :
            _width(result.width), _height(result.height), _channels(result.channels),
            _data(reinterpret_cast<char*>(result.data), result.width * result.height * result.channels){
    }

	AmVerifyWaterRemovalResult(const unsigned char* data, int width, int height, int channels) :
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
