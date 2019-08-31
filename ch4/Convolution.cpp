#include "Convolution.h"
#include "LayerType.h"

DEFINE_LAYER_CREATOR(Convolution)

Convolution::Convolution() : _oneBlobOnly(true)
{
}

Convolution::~Convolution()
{
}

int32_t Convolution::LoadParam(const ParamDict& pd)
{
    _numOutput = pd.get(0, 0);
    _kernelW = pd.get(1, 0);
    _kernelH = pd.get(11, _kernelW);
    _dilationW = pd.get(2, 1);
    _dilationH = pd.get(12, _dilationW);
    _strideW = pd.get(3, 1);
    _strideH = pd.get(13, _strideW);
    _padW = pd.get(4, 0);
    _padH = pd.get(14, _padW);
    _biasTerm = pd.get(5, 0);
    _weightDataSize = pd.get(6, 0);

    return 0;
}

int32_t Convolution::LoadModel(const ModelBin& mb)
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

int32_t Convolution::Forward(const Mat& bottomBlob, Mat& topBlob, const Option& opt) const
{
    if (bottomBlob.dims == 1 && _kernelW == 1 && _kernelH == 1)
    {
        int32_t num_input = _weightDataSize / _numOutput;
        if (bottomBlob.w == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::CreateLayer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, _numOutput);
            pd.set(1, _biasTerm);
            pd.set(2, _weightDataSize);

            op->LoadParam(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = _weightData;
            weights[1] = _biasData;

            op->LoadModel(ModelBinFromMatArray(weights));
            op->Forward(bottomBlob, topBlob, opt);

            delete op;

            return 0;
        }
    }

    int32_t w = bottomBlob.w;
    int32_t h = bottomBlob.h;
    int32_t channels = bottomBlob.c;
    size_t elemsize = bottomBlob.elemsize;

    const int32_t kernel_extent_w = _dilationW * (_kernelW - 1) + 1;
    const int32_t kernel_extent_h = _dilationH * (_kernelH - 1) + 1;

    Mat bottomBlob_bordered = bottomBlob;
    if (_padW > 0 || _padH > 0)
    {
        copy_make_border(bottomBlob, bottomBlob_bordered, _padH, _padH, _padW, _padW, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
        if (bottomBlob_bordered.empty())
            return -100;

        w = bottomBlob_bordered.w;
        h = bottomBlob_bordered.h;
    }
    else if (_padW == -233 && _padH == -233)
    {
        int32_t wpad = kernel_extent_w + (w - 1) / _strideW * _strideW - w;
        int32_t hpad = kernel_extent_h + (h - 1) / _strideH * _strideH - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottomBlob, bottomBlob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottomBlob_bordered.empty())
                return -100;
        }

        w = bottomBlob_bordered.w;
        h = bottomBlob_bordered.h;
    }

    int32_t outw = (w - kernel_extent_w) / _strideW + 1;
    int32_t outh = (h - kernel_extent_h) / _strideH + 1;

    topBlob.create(outw, outh, _numOutput, elemsize, opt.blob_allocator);
    if (topBlob.empty())
        return -100;

    const int32_t maxk = _kernelW * _kernelH;

    // kernel offsets
    std::vector<int32_t> _space_ofs(maxk);
    int32_t* space_ofs = &_space_ofs[0];
    {
        int32_t p1 = 0;
        int32_t p2 = 0;
        int32_t gap = w * _dilationH - _kernelW * _dilationW;
        for (int32_t i = 0; i < _kernelH; i++)
        {
            for (int32_t j = 0; j < _kernelW; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += _dilationW;
            }
            p2 += gap;
        }
    }

    for (int32_t p=0; p<_numOutput; p++)
    {
        float* outptr = topBlob.channel(p);

        for (int32_t i = 0; i < outh; i++)
        {
            for (int32_t j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (_biasTerm)
                    sum = _biasData[p];

                const float* kptr = (const float*)_weightData + maxk * channels * p;

                // channels
                for (int32_t q=0; q<channels; q++)
                {
                    const Mat m = bottomBlob_bordered.channel(q);
                    const float* sptr = m.row(i*_strideH) + j*_strideW;

                    for (int32_t k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }

                    kptr += maxk;
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}
