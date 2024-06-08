#pragma once

#include "common/AmType.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>

namespace authen {
namespace core {
namespace landmarker {

class CropPartUtil
{
public:
    CropPartUtil();
    ~CropPartUtil();
    static CropPartUtil& GetInstance();
    static void PreparePart2ForPTScore(unsigned char* src, int width, int height, unsigned char* dst, float* lms);
};

}
}
}