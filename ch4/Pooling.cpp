#include "Pooling.h"
#include <float.h>
#include <algorithm>

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling() : oneBlobOnly(true)
{
}

int32_t Pooling::LoadParam(const ParamDict& pd)
{
    _poolingType = pd.get(0, 0);
    _kernelW = pd.get(1, 0);
    _kernelH = pd.get(11, _kernelW);
    _strideW = pd.get(2, 1);
    _strideH = pd.get(12, _strideW);
    _padLeft = pd.get(3, 0);
    _padRight = pd.get(14, _padLeft);
    _padTop = pd.get(13, _padLeft);
    _padBottom = pd.get(15, _padTop);
    _globalPooling = pd.get(4, 0);

    return 0;
}

int32_t Pooling::Forward(const Mat& bottomBlob, Mat& topBlob, const Option& opt) const
{
    int32_t w = bottomBlob.w;
    int32_t h = bottomBlob.h;
    int32_t channels = bottomBlob.c;
    size_t elemsize = bottomBlob.elemsize;

    if (_globalPooling)
    {
        topBlob.create(channels, elemsize, opt.blob_allocator);
        if (topBlob.empty())
            return -100;

        int32_t size = w * h;

        if (_poolingType == PoolMethod_MAX)
        {
            for (int32_t q=0; q<channels; q++)
            {
                const float* ptr = bottomBlob.channel(q);

                float max = ptr[0];
                for (int32_t i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                topBlob[q] = max;
            }
        }
        else if (_poolingType == PoolMethod_AVE)
        {
            for (int32_t q=0; q<channels; q++)
            {
                const float* ptr = bottomBlob.channel(q);

                float sum = 0.f;
                for (int32_t i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                topBlob[q] = sum / size;
            }
        }

        return 0;
    }

    Mat bottomBlob_bordered = bottomBlob;

    float pad_value = 0.f;
    if (_poolingType == PoolMethod_MAX)
    {
        pad_value = -FLT_MAX;
    }
    else if (_poolingType == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int32_t wtailpad = 0;
    int32_t htailpad = 0;

    int32_t wtail = (w + _padLeft + _padRight - _kernelW) % _strideW;
    int32_t htail = (h + _padTop + _padBottom - _kernelH) % _strideH;

    if (wtail != 0)
        wtailpad = _strideW - wtail;
    if (htail != 0)
        htailpad = _strideH - htail;

    copy_make_border(bottomBlob, bottomBlob_bordered, _padTop, _padBottom + htailpad, _padLeft, _padRight + wtailpad, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
    if (bottomBlob_bordered.empty())
        return -100;

    w = bottomBlob_bordered.w;
    h = bottomBlob_bordered.h;

    int32_t outw = (w - _kernelW) / _strideW + 1;
    int32_t outh = (h - _kernelH) / _strideH + 1;

    topBlob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (topBlob.empty())
        return -100;

    const int32_t maxk = _kernelW * _kernelH;

    // kernel offsets
    std::vector<int32_t> _space_ofs(maxk);
    int32_t* space_ofs = &_space_ofs[0];
    {
        int32_t p1 = 0;
        int32_t p2 = 0;
        int32_t gap = w - _kernelW;
        for (int32_t i = 0; i < _kernelH; i++)
        {
            for (int32_t j = 0; j < _kernelW; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (_poolingType == PoolMethod_MAX)
    {
        for (int32_t q=0; q<channels; q++)
        {
            const Mat m = bottomBlob_bordered.channel(q);
            float* outptr = topBlob.channel(q);

            for (int32_t i = 0; i < outh; i++)
            {
                for (int32_t j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*_strideH) + j*_strideW;

                    float max = sptr[0];

                    for (int32_t k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (_poolingType == PoolMethod_AVE)
    {
        for (int32_t q=0; q<channels; q++)
        {
            const Mat m = bottomBlob_bordered.channel(q);
            float* outptr = topBlob.channel(q);

            for (int32_t i = 0; i < outh; i++)
            {
                for (int32_t j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*_strideH) + j*_strideW;

                    float sum = 0;

                    for (int32_t k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix pad
            if (_padTop != 0)
            {
                const float scale = (float)_kernelH / (_kernelH - _padTop);

                outptr = topBlob.channel(q).row(0);
                for (int32_t i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (_padBottom + htailpad != 0)
            {
                const float scale = (float)_kernelH / (_kernelH - _padBottom - htailpad);

                outptr = topBlob.channel(q).row(outh - 1);
                for (int32_t i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (_padLeft != 0)
            {
                const float scale = (float)_kernelW / (_kernelW - _padLeft);

                outptr = topBlob.channel(q);
                for (int32_t i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (_padRight + wtailpad != 0)
            {
                const float scale = (float)_kernelW / (_kernelW - _padRight - wtailpad);

                outptr = topBlob.channel(q);
                outptr += outw - 1;
                for (int32_t i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}
