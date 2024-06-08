#pragma once

#ifdef WIN32
#ifdef IQUALITY_EXPORTS
#define QUALITYAPI_API __declspec(dllexport)
#else
#define QUALITYAPI_API __declspec(dllimport)
#endif
#else
#define QUALITYAPI_API
#endif

#include "common/AmType.h"
#include <string>
#include <vector>
#include <set>

//class QualityDetector;
class FacePropHandler;

namespace authen {
namespace core {
namespace api {

class QUALITYAPI_API IQuality {
public:
    static int checkEnvironment(std::string modelPath, std::string product_code);

    IQuality(std::string modelPath, std::string product_code, int DeviceId = 0);

    ~IQuality();

    //Qualityerty
    void getQualityResult(unsigned char* imageData, int imageWidth, int imageHeight,
                       int imagePitch, const AmLandmarkResult& landmarkResult, AmQualityResult* amQualityResult);

private:
    //QualityDetector* _qualityHandler;
	FacePropHandler* _qualityHandler;

};

}
}
}
