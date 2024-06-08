#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include "common/AmType.h"

using namespace std;

class CropPartUtil
{
public:
	CropPartUtil();
    ~CropPartUtil();
    static CropPartUtil& GetInstance();
    static void PreparePart2ForPTScore(unsigned char* src, int width, int height, unsigned char* dst, float* lms);

};
