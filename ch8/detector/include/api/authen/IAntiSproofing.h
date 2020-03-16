#pragma once

#ifdef WIN32
#ifdef IANTISPROOFING_EXPORTS
#define IANTISPROOFING_API __declspec(dllexport)
#else
#define IANTISPROOFING_API __declspec(dllimport)
#endif
#else
#define IANTISPROOFING_API
#endif

#include <string>
#include <vector>
#include <set>

#define AntiHandle void*

namespace authen {
namespace core {
namespace api {

class IANTISPROOFING_API IAntiSproofing {
public:
    static int checkEnvironment(std::string modelPath);

    IAntiSproofing(const std::string& modelPath, int type = 0);

    ~IAntiSproofing();

    float getScore(const unsigned char* imageData,
            int imageWidth, int imageHeight, int imagePitch, int bpp, const float* marks);

    float getScoreVideo(const unsigned char* imageData_list,
        int imageWidth, int imageHeight, int imagePitch, int bpp, const float* marks_list, int num);

    //just support infraid, not support color
    float getEyeScore(const unsigned char* imageData,
        int imageWidth, int imageHeight, int imagePitch, int bpp, const float* marks);

private:
    AntiHandle _antiHandle;
};


}
}
}
