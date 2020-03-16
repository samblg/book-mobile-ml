// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "modelbin.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <istream>
#include <ostream>
#include "platform.h"

namespace ncnn {

static unsigned short float2half(float value);

Mat ModelBin::load(int w, int h, int type) const
{
    Mat m = load(w * h, type);
    if (m.empty())
        return m;

    return m.reshape(w, h);
}

Mat ModelBin::load(int w, int h, int c, int type) const
{
    Mat m = load(w * h * c, type);
    if (m.empty())
        return m;

    return m.reshape(w, h, c);
}

void ModelBin::dump(const float* data, int dataSize, bool needToQuantize, int quantizeLevel) const {
}

int ModelBin::quantizeWeight(float *data, size_t data_length, std::vector<unsigned short>& float16_weights)
{
    float16_weights.resize(data_length);

    for (size_t i = 0; i < data_length; i++)
    {
        float f = data[i];

        unsigned short fp16 = float2half(f);

        float16_weights[i] = fp16;
    }

    // magic tag for half-precision floating point
    return 0x01306B47;
}

bool ModelBin::quantizeWeight(float *data, size_t data_length, int quantize_level,
    std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index) {
    assert(quantize_level != 0);
    assert(data != NULL);
    assert(data_length > 0);

    if (data_length < static_cast<size_t>(quantize_level)) {
        fprintf(stderr, "No need quantize,because: data_length < quantize_level");
        return false;
    }

    quantize_table.reserve(quantize_level);
    quantize_index.reserve(data_length);

    // 1. Find min and max value
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    for (size_t i = 0; i < data_length; ++i)
    {
        if (max_value < data[i]) max_value = data[i];
        if (min_value > data[i]) min_value = data[i];
    }
    float strides = (max_value - min_value) / quantize_level;

    // 2. Generate quantize table
    for (int i = 0; i < quantize_level; ++i)
    {
        quantize_table.push_back(min_value + i * strides);
    }

    // 3. Align data to the quantized value
    for (size_t i = 0; i < data_length; ++i)
    {
        size_t table_index = int((data[i] - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);

        float low_value = quantize_table[table_index];
        float high_value = low_value + strides;

        // find a nearest value between low and high value.
        float targetValue = data[i] - low_value < high_value - data[i] ? low_value : high_value;

        table_index = int((targetValue - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);
        quantize_index.push_back(table_index);
    }

    return true;
}

#if NCNN_STDIO
ModelBinFromStdio::ModelBinFromStdio(FILE* _binfp) : binfp(_binfp)
{
}

Mat ModelBinFromStdio::load(int w, int type) const
{
    if (!binfp)
        return Mat();

    if (type == 0)
    {
        int nread;

        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;

        nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "ModelBin read flag_struct failed %d\n", nread);
            return Mat();
        }

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            int align_data_size = alignSize(w * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_data_size);
            nread = fread(float16_weights.data(), align_data_size, 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read float16_weights failed %d\n", nread);
                return Mat();
            }

            return Mat::from_float16(float16_weights.data(), w);
        }

        Mat m(w);
        if (m.empty())
            return m;

        if (flag != 0)
        {
            // quantized data
            float quantization_value[256];
            nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read quantization_value failed %d\n", nread);
                return Mat();
            }

            int align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read index_array failed %d\n", nread);
                return Mat();
            }

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[ index_array[i] ];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            nread = fread(m, w * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                fprintf(stderr, "ModelBin read weight_data failed %d\n", nread);
                return Mat();
            }
        }

        return m;
    }
    else if (type == 1)
    {
        Mat m(w);
        if (m.empty())
            return m;

        // raw data
        int nread = fread(m, w * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "ModelBin read weight_data failed %d\n", nread);
            return Mat();
        }

        return m;
    }
    else
    {
        fprintf(stderr, "ModelBin load type %d not implemented\n", type);
        return Mat();
    }

    return Mat();
}

void ModelBinFromStdio::dump(const float* data, int dataSize, bool needToQuantize, int quantizeLevel) const {
    int quantize_tag = 0;

    std::vector<float> quantize_table;
    std::vector<unsigned char> quantize_index;

    std::vector<unsigned short> float16_weights;

    // we will not quantize the bias values
    if (needToQuantize && quantizeLevel != 0)
    {
        if (quantizeLevel == 256)
        {
            quantize_tag = quantizeWeight((float *)data, dataSize, quantizeLevel, quantize_table, quantize_index);
        }
        else if (quantizeLevel == 65536)
        {
            quantize_tag = quantizeWeight((float *)data, dataSize, float16_weights);
        }
    }

    // write quantize tag first
    if (needToQuantize)
        fwrite(&quantize_tag, sizeof(int), 1, binfp);

    if (quantize_tag)
    {
        int p0 = ftell(binfp);
        if (quantizeLevel == 256)
        {
            // write quantize table and index
            fwrite(quantize_table.data(), sizeof(float), quantize_table.size(), binfp);
            fwrite(quantize_index.data(), sizeof(unsigned char), quantize_index.size(), binfp);
        }
        else if (quantizeLevel == 65536)
        {
            fwrite(float16_weights.data(), sizeof(unsigned short), float16_weights.size(), binfp);
        }
        // padding to 32bit align
        int nwrite = ftell(binfp) - p0;
        int nalign = alignSize(nwrite, 4);
        unsigned char padding[4] = { 0x00, 0x00, 0x00, 0x00 };
        fwrite(padding, sizeof(unsigned char), nalign - nwrite, binfp);
    }
    else
    {
        // write original data
        fwrite(data, sizeof(float), dataSize, binfp);
    }
}

ModelBinFromStream::ModelBinFromStream(std::istream& is) : _is(&is), _os(nullptr)
{
}

ModelBinFromStream::ModelBinFromStream(std::ostream& os) : _is(nullptr), _os(&os)
{
}

ModelBinFromStream::ModelBinFromStream(std::istream& is, std::ostream& os) : _is(&is), _os(&os)
{
}

Mat ModelBinFromStream::load(int w, int type) const
{
    if (!_is)
        return Mat();

    std::istream& is = *_is;
    if (type == 0)
    {
        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;

        is.read(reinterpret_cast<char*>(&flag_struct), sizeof(flag_struct));
        if (!is)
        {
            fprintf(stderr, "ModelBin read flag_struct failed\n");
            return Mat();
        }

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            int align_data_size = alignSize(w * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_data_size);
            is.read(reinterpret_cast<char*>(float16_weights.data()), align_data_size);
            if (!is)
            {
                fprintf(stderr, "ModelBin read float16_weights failed\n");
                return Mat();
            }

            return Mat::from_float16(float16_weights.data(), w);
        }

        Mat m(w);
        if (m.empty())
            return m;

        if (flag != 0)
        {
            // quantized data
            float quantization_value[256];
            is.read(reinterpret_cast<char*>(quantization_value), 256 * sizeof(float));
            if (!is)
            {
                fprintf(stderr, "ModelBin read quantization_value failed\n");
                return Mat();
            }

            int align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            is.read(reinterpret_cast<char*>(index_array.data()), align_weight_data_size);
            if (!is)
            {
                fprintf(stderr, "ModelBin read index_array failed\n");
                return Mat();
            }

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[index_array[i]];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            is.read(reinterpret_cast<char*>(m.data), w * sizeof(float));
            if (!is)
            {
                fprintf(stderr, "ModelBin read weight_data failed\n");
                return Mat();
            }
        }

        return m;
    }
    else if (type == 1)
    {
        Mat m(w);
        if (m.empty())
            return m;

        // raw data
        is.read(reinterpret_cast<char*>(m.data), w * sizeof(float));
        if (!is)
        {
            fprintf(stderr, "ModelBin read weight_data failed\n");
            return Mat();
        }

        return m;
    }
    else
    {
        fprintf(stderr, "ModelBin load type %d not implemented\n", type);
        return Mat();
    }

    return Mat();
}

void ModelBinFromStream::dump(const float* data, int dataSize, bool needToQuantize, int quantizeLevel) const {
    if (!_os) {
        return;
    }

    std::ostream& os = *_os;

    int quantize_tag = 0;

    std::vector<float> quantize_table;
    std::vector<unsigned char> quantize_index;

    std::vector<unsigned short> float16_weights;

    // we will not quantize the bias values
    if (needToQuantize && quantizeLevel != 0)
    {
        if (quantizeLevel == 256)
        {
            quantize_tag = quantizeWeight((float *)data, dataSize, quantizeLevel, quantize_table, quantize_index);
        }
        else if (quantizeLevel == 65536)
        {
            quantize_tag = quantizeWeight((float *)data, dataSize, float16_weights);
        }
    }

    // write quantize tag first
    if (needToQuantize) {
        //fwrite(&quantize_tag, sizeof(int), 1, binfp);
        os.write(reinterpret_cast<char*>(&quantize_tag), sizeof(int));
    }


    if (quantize_tag)
    {
        //int p0 = ftell(binfp);
        int p0 = os.tellp();
        if (quantizeLevel == 256)
        {
            // write quantize table and index
            //fwrite(quantize_table.data(), sizeof(float), quantize_table.size(), binfp);
            os.write(reinterpret_cast<char*>(quantize_table.data()), sizeof(float)* quantize_table.size());
            //fwrite(quantize_index.data(), sizeof(unsigned char), quantize_index.size(), binfp);
            os.write(reinterpret_cast<char*>(quantize_index.data()), sizeof(unsigned char) * quantize_index.size());
        }
        else if (quantizeLevel == 65536)
        {
            //fwrite(float16_weights.data(), sizeof(unsigned short), float16_weights.size(), binfp);
            os.write(reinterpret_cast<char*>(float16_weights.data()), sizeof(unsigned short) * float16_weights.size());
        }
        // padding to 32bit align
        int nwrite = static_cast<int>(os.tellp()) - p0;
        int nalign = alignSize(nwrite, 4);
        unsigned char padding[4] = { 0x00, 0x00, 0x00, 0x00 };
        //fwrite(padding, sizeof(unsigned char), nalign - nwrite, binfp);
        os.write(reinterpret_cast<char*>(padding), sizeof(unsigned char)* (nalign - nwrite));
    }
    else
    {
        // write original data
        //fwrite(data, sizeof(float), dataSize, binfp);
        os.write(reinterpret_cast<const char*>(data), dataSize);
    }
}
#endif // NCNN_STDIO    

ModelBinFromMemory::ModelBinFromMemory(const unsigned char*& _mem) : mem(_mem)
{
}

Mat ModelBinFromMemory::load(int w, int type) const
{
    if (!mem)
        return Mat();

    if (type == 0)
    {
        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;

        memcpy(&flag_struct, mem, sizeof(flag_struct));
        mem += sizeof(flag_struct);

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            Mat m = Mat::from_float16((unsigned short*)mem, w);
            mem += alignSize(w * sizeof(unsigned short), 4);
            return m;
        }

        if (flag != 0)
        {
            // quantized data
            const float* quantization_value = (const float*)mem;
            mem += 256 * sizeof(float);

            const unsigned char* index_array = (const unsigned char*)mem;
            mem += alignSize(w * sizeof(unsigned char), 4);

            Mat m(w);
            if (m.empty())
                return m;

            float* ptr = m;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[ index_array[i] ];
            }

            return m;
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            Mat m = Mat(w, (float*)mem);
            mem += w * sizeof(float);
            return m;
        }
    }
    else if (type == 1)
    {
        // raw data
        Mat m = Mat(w, (float*)mem);
        mem += w * sizeof(float);
        return m;
    }
    else
    {
        fprintf(stderr, "ModelBin load type %d not implemented\n", type);
        return Mat();
    }

    return Mat();
}

ModelBinFromMatArray::ModelBinFromMatArray(const Mat* _weights) : weights(_weights)
{
}

Mat ModelBinFromMatArray::load(int /*w*/, int /*type*/) const
{
    if (!weights)
        return Mat();

    Mat m = weights[0];
    weights++;
    return m;
}

// convert float to half precision floating point
static unsigned short float2half(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

} // namespace ncnn
