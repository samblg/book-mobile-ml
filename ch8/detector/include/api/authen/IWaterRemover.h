#pragma once

#ifdef WIN32
#ifdef IWATERREMOVER_EXPORTS
#define IWATERREMOVER_API __declspec(dllexport)
#else
#define IWATERREMOVER_API __declspec(dllimport)
#endif
#else
#define IWATERREMOVER_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class WaterRemover;

namespace authen {
namespace core {
namespace api {

class AmVerifyWaterRemovalResult;

class IWATERREMOVER_API IWaterRemover {
public:
    static int checkEnvironment(std::string modelPath, std::string product_code);

    IWaterRemover(std::string modelPath, std::string product_code, int extractDeviceId = 0); 
    ~IWaterRemover();

	AmVerifyWaterRemovalResult removeWater(unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch);

private:
    WaterRemover* _caffeWaterRemover;
    WaterRemover* _oldWaterRemover;
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
