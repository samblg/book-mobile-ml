#pragma once

#include "common/AmType.h"

void CropLandmark(unsigned char* imageData, int imageWidth, int imageHeight,
    AmLandmarkResult* landmarkResult, AmCroppedResult* croppedResult);
