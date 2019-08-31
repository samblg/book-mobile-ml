#ifndef LAYER_POOLING_H
#define LAYER_POOLING_H

#include "Layer.h"

class Pooling : public Layer
{
public:
    Pooling();
    virtual ~Pooling();

    virtual int32_t LoadParam(const ParamDict& pd);
    virtual int32_t Forward(const Mat& bottomBlob, Mat& topBlob, const Option& opt) const;
    enum class { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };

public:
    // param
    int32_t _poolingType {0};
    int32_t _kernelW {0};
    int32_t _kernelH {0};
    int32_t _strideW {0};
    int32_t _strideH {0};
    int32_t _padLeft {0};
    int32_t _padRight {0};
    int32_t _padTop {0};
    int32_t _padBottom {0};
    int32_t _globalPooling {0};
};

#endif

