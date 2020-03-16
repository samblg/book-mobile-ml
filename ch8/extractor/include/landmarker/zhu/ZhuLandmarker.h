#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include "common/AmType.h"
#include "landmarker/Landmarker.h"
#include "landmarker/zhu/model.h"

#include <string>

class ZhuLandmarker : public Landmarker {
public:
    ZhuLandmarker(const std::string& path);
    //~ZhuLandmarker() override; //???
    ~ZhuLandmarker();
    virtual void setParam(AmLandmarkerParam* param) override;
    virtual void getLandmark(unsigned char* bgr, int width, int height,
        int pitch, AmFaceRect* rect, AmLandmarkResult* result) override;
private:
    Model _model;
};

ZhuLandmarker CreateZHULandMarker(const char* LandmarkModelPath);

class ZhuCropImage {
public:
    ZhuCropImage() :
        _width(0), _height(0) {
    }

    ZhuCropImage(const unsigned char* data, int width, int height, int channels) :
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

#endif
