#pragma once

#include <string>

namespace authen {
namespace core {
namespace api {

class Rect {
public:
    Rect() :
        left(0), top(0), right(0), bottom(0) {
    }

    Rect(float left, float top, float right, float bottom) {
        this->left = left;
        this->top = top;
        this->right = right;
        this->bottom = bottom;
    }

    float left;
    float top;
    float right;
    float bottom;
};

class Image {
public:
    Image() :
            _width(0), _height(0) {
    }

    Image(const unsigned char* data, int width, int height, int channels) :
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

static const std::string AMCORE_CONFIG_FILE = "amcore.config.json";

}
}
}
