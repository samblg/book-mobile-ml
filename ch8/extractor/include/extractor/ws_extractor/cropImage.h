#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thread>
#include "common/minicv/minicv.h"
#include "common/AmType.h"

namespace authen {
namespace core {
namespace extractor {

class CropImageUtil
{
public:
    CropImageUtil();
    ~CropImageUtil();
    static CropImageUtil& GetInstance();
    static std::vector<minicv::Mat> PreparePatches(minicv::Mat color, const TLandmarks1* lms, int isMirror);
    static std::vector<minicv::Mat> PrepareSinglePatch(minicv::Mat color, const TLandmarks1* lms, int isMirror);
};

}
}
}
