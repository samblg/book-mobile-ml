#ifndef CAFFE_LANDMARKER_H
#define CAFFE_LANDMARKER_H

#include "common/AmType.h"
#include "antisproof/antiProcessor.hpp"

#include <string>
#include <memory>

class CaffeAntiSproofor{
public:
   // CaffeAntiSproofor();
    CaffeAntiSproofor(const std::string& path, int type); //type = 0: infrared  type = 1: color
	//~ZhuLandmarker() override; //¸Ä¶¯
    ~CaffeAntiSproofor();
	// setParam(AmLandmarkerParam* param);
	float getAntiScore(unsigned char* bgr, int width, int height,
		int pitch, int bpp, float* marks);
    float getAntiScoreVideo(unsigned char* bgr_list, int width, int height,
        int pitch, int bpp, float* marks_list, int num);
    float getEyeScore(unsigned char* bgr, int width, int height,
        int pitch, int bpp, float* marks);
private:
    std::shared_ptr<authen::AntiProcessor> _antiProcessor;
    std::shared_ptr<authen::AntiProcessor> _eyeAntiProcessor;
};

//CaffeAntiSproofor CreateCaffeAntiSproofor(const char* LandmarkModelPath);

#endif
