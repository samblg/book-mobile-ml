#pragma once

#include "landmarker/esr/AuthenFAFun.h"
#include "common/AmType.h"
#include "landmarker/Landmarker.h"
#include <string>

class EsrLandmarker : public Landmarker {
public:
    EsrLandmarker(const std::string& dataDirPath);
    ~EsrLandmarker();
	virtual void setParam(AmLandmarkerParam* param) override;
    virtual void getLandmark(unsigned char* bgr, int width, int height,
        int pitch, AmFaceRect* rect, AmLandmarkResult* result) override;
    virtual void getLandmark_group(std::vector<AmImage> images, int faceCount, AmFaceRect* rect, AmLandmarkResult* result) override;

private:
    std::string _dataDirPath;
    ESRCLASS _esrObject;
};

