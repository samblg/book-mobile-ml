#include "InnerProduct.h"
#include "LayerType.h"

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct() : oneBlobOnly(true)
{
}

InnerProduct::~InnerProduct()
{
}

int32_t InnerProduct::LoadParam(const ParamDict& pd)
{
    _numOutput = pd.get(0, 0);
    _biasTerm = pd.get(1, 0);
    _weightDataSize = pd.get(2, 0);

    return 0;
}

int32_t InnerProduct::LoadModel(const ModelBin& mb)
{
    _weightData = mb.load(_weightDataSize, 0);
    if (_weightData.empty())
        return -100;

    if (_biasTerm)
    {
        _biasData = mb.load(_numOutput, 1);
        if (_biasData.empty())
            return -100;
    }

    return 0;
}

int32_t InnerProduct::Forward(const Mat& bottomBlob, Mat& topBlob, const Option& opt) const
{
    int32_t w = bottomBlob.w;
    int32_t h = bottomBlob.h;
    int32_t channels = bottomBlob.c;
    size_t elemsize = bottomBlob.elemsize;
    int32_t size = w * h;

    topBlob.create(_numOutput, elemsize, opt.blob_allocator);
    if (topBlob.empty())
        return -100;

    for (int32_t p=0; p<_numOutput; p++)
    {
        float sum = 0.f;

        if (_biasTerm)
            sum = _biasData[p];

        for (int32_t q=0; q<channels; q++)
        {
            const float* w = (const float*)_weightData + size * channels * p + size * q;
            const float* m = bottomBlob.channel(q);

            for (int32_t i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        topBlob[p] = sum;
    }

    return 0;
}
