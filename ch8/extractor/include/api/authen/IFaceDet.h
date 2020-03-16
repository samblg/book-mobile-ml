#pragma once

#ifdef WIN32
#ifdef IDETECTOR_EXPORTS
#define IDETECTOR_API __declspec(dllexport)
#else
#define IDETECTOR_API __declspec(dllimport)
#endif
#else
#define IDETECTOR_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class FaceDetector;
class DirectionDetector;
class Landmarker;

namespace authen {
namespace core {
namespace api {

class AmVerifyDetectResult;
class AmVerifyLandmarkResult;

class IDETECTOR_API IFaceDet {
public:
    static int checkEnvironment(std::string modelPath, std::string product_code);

    IFaceDet(std::string modelPath, std::string product_code, int extractDeviceId = 0);

	void setDetectParams(AmDetectorParam params);

    ~IFaceDet();

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

    std::vector<AmCroppedResult> detectAndCropFacesByRect_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, std::vector<AmFaceRect> faceRects);

    AmVerifyLandmarkResult detectFaces_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    //ONLY for GPU
    std::vector<AmCroppedResult> detectAndCropFaces(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount, bool enableRotate = true);
    std::vector<AmCroppedResult> detectAndCropFaces_Surveilance(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    std::vector<AmCroppedResult> detectAndCropFaces_Surveilance_group(std::vector<AmImage>, float threshold, int maxFaceCount);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);

private:
    static bool existsFile(const std::string& path);

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

    //v3.3 new interface
    std::vector<AmCroppedResult> detectCropByFR_group(std::vector<AmImage> images, float threshold, int maxFaceCount);
    
private:
    FaceDetector* _jqwDetector;
    DirectionDetector* _caffeDirectionDetector;
    Landmarker* _jqwLandmarker;

    FaceDetector* _frDetector;
    Landmarker* _zhuLandmarker;

	std::string modelPathGlobal;
	std::string productCodeGlobal;
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
        //for (AmLandmarkResult& landmarkResult : _landmarkResults) {
        //    if (landmarkResult.landmarks) {
        //        delete[] landmarkResult.landmarks;
        //        landmarkResult.landmarks = nullptr;
        //    }
        //}
    }

private:
    Direction _direction;
    std::vector<AmLandmarkResult> _landmarkResults;
    AmVerifyImage _face;
};

}
}
}
