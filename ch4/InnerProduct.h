#ifndef LAYER_INNERPRODUCT_H
#define LAYER_INNERPRODUCT_H

#include "Layer.h"

class InnerProduct : public Layer
{
public:
    InnerProduct();
    virtual ~InnerProduct();

    virtual int32_t LoadParam(const ParamDict& pd);
    virtual int32_t LoadModel(const ModelBin& mb);
    virtual int32_t Forward(const Mat& bottomBlob, Mat& topBlob) const;

protected:
    int32_t _numOutput {0};
    int32_t _biasTerm {0};
    int32_t _weightDataSize {0};

    Mat _weightData;
    Mat _biasData;
};

#endif

