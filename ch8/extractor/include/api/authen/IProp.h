#pragma once

#ifdef WIN32
#ifdef IPROP_EXPORTS
#define PROPAPI_API __declspec(dllexport)
#else
#define PROPAPI_API __declspec(dllimport)
#endif
#else
#define PROPAPI_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

class FacePropHandler;

namespace authen {
namespace core {
namespace api {

class PROPAPI_API IProp {
public:
    static int checkEnvironment(std::string modelPath, std::string product_code);

	IProp(std::string modelPath, std::string product_code, int DeviceId = 0);

    ~IProp();

    //Property
    void getPropResult(unsigned char* imageData, int imageWidth, int imageHeight,
                       int imagePitch, const AmLandmarkResult& landmarkResult, AmPropResult* amPropResult);

private:
    FacePropHandler* _propHandler;
};

}
}
}
