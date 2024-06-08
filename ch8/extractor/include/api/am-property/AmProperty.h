#pragma once

#ifdef WIN32
#ifdef AMPROPERTYAPI_EXPORTS
#define AMPROPERTYAPI_API __declspec(dllexport)
#else
#define AMPROPERTYAPI_API __declspec(dllimport)
#endif
#else
#define AMPROPERTYAPI_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class FaceDetector;
class Landmarker;
class FacePropHandler;

namespace authen {
namespace core {
namespace api {

class AmPropertyWaterRemovalResult;

class AMPROPERTYAPI_API AmProperty {
public:
    enum class Task {
        Detect,
        Prop
    };

    AmProperty(std::string modelPath, int extractDeviceId = 0);
    AmProperty(std::string modelPath, std::set<Task> tasks, int extractDeviceId = 0);

    void setDetectParams(AmDetectorParam params);

    ~AmProperty();

    std::vector<AmFaceRect> detectFaceRects(const unsigned char* imageData,
                                            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmLandmarkResult> detectFaceLandMarksByRect(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, std::vector<AmFaceRect> faceRects);

    std::vector<AmLandmarkResult> detectFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    //ONLY for GPU
    std::vector<AmCroppedResult> detectAndCropFaces(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);


    //Property
    void getPropResult(unsigned char* imageData, int imageWidth, int imageHeight,
                       int imagePitch, const AmLandmarkResult& landmarkResult, AmPropResult* amPropResult);

    void freeLandmarkResults(std::vector<AmLandmarkResult>& landmarkResults);

    bool hasTask(Task task);

private:
    std::vector<AmFaceRect> detectFaceRectByJQW(const unsigned char* imageData,
                                               int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);

    std::vector<AmLandmarkResult> detectByJQW(const unsigned char* imageData,
                                             int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    std::vector<AmCroppedResult> detectCropByJQW(const unsigned char* imageData,
                                                int imageWidth, int imageHeight, int imagePitch, float threshold, int maxFaceCount);
    

private:
    std::set<Task> _tasks;
    FaceDetector* _jqwDetector;
    Landmarker* _jqwLandmarker;
    FacePropHandler* _propHandler;
};

class AmPropertyWaterRemovalResult {
public:
    AmPropertyWaterRemovalResult() :
        _width(0), _height(0) {
    }

    AmPropertyWaterRemovalResult(const AmWaterRemovalResult& result) :
        _width(result.width), _height(result.height), _channels(result.channels),
        _data(reinterpret_cast<char*>(result.data), result.width * result.height * result.channels){
    }

    AmPropertyWaterRemovalResult(const unsigned char* data, int width, int height, int channels) :
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
