#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include "common/minicv/minicv.h"
#include "MlType.h"

using namespace std;

namespace learnml {

class CropImageUtil
{
public:
    CropImageUtil();
    ~CropImageUtil();
    static CropImageUtil& GetInstance();
    static vector<minicv::Mat> PreparePatches(minicv::Mat color, const TLandmarks1* lms, int isMirror);
    static vector<minicv::Mat> PrepareSinglePatch(minicv::Mat color, const TLandmarks1* lms, int isMirror);

};

}
