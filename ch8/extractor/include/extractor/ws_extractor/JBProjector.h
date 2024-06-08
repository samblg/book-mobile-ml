#pragma once

#include "common/AmType.h"
#include <string>

class JBProjector {
public:
    enum {
        DFEATURE_DIM = 512 * 1,
        JFEATURE_DIM = 257 //temp set this value
    };

    JBProjector(const std::string& packagePath);
    ~JBProjector() {}

    void project(float* dfea, float* jfea);
    void project(float* dfea, float* jfea, int faceCount);

private:
    float _modelM[DFEATURE_DIM];
    float _modelA[DFEATURE_DIM * JFEATURE_DIM];
    float _modelG[DFEATURE_DIM * JFEATURE_DIM];
};
