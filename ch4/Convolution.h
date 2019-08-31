#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "Layer.h"

class Convolution : public Layer
{
public:
    Convolution();
    virtual ~Convolution();

    virtual int32_t LoadParam(const ParamDict& pd);
    virtual int32_t LoadModel(const ModelBin& mb);
    virtual int32_t Forward(const Mat& bottomBlob, Mat& topBlob, const Option& opt) const;

protected:
    int32_t _numOutput {0};
    int32_t _kernelW {0};
    int32_t _kernelH {0};
    int32_t _dilationW {0};
    int32_t _dilationH {0};
    int32_t _strideW {0};
    int32_t _strideH {0};
    int32_t _padW {0};
    int32_t _padH {0};
    int32_t _biasTerm {0};
    int32_t _weightDataSize {0};

    Mat _weightData;
    Mat _biasData;
};

#endif

