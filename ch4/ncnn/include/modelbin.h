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

#ifndef NCNN_MODELBIN_H
#define NCNN_MODELBIN_H

#include <stdio.h>
#include <vector>
#include "mat.h"
#include "platform.h"

namespace ncnn {

class Net;
class ModelBin
{
public:
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = uint8
    // load vec
    virtual Mat load(int w, int type) const = 0;
    // load image
    virtual Mat load(int w, int h, int type) const;
    // load dim
    virtual Mat load(int w, int h, int c, int type) const;

    virtual void dump(const float* data, int dataSize, bool needToQuantize = false, int quantizeLevel = 0) const;

    static int quantizeWeight(float *data, size_t data_length, ::std::vector<unsigned short>& float16_weights);
    static bool quantizeWeight(float *data, size_t data_length, int quantize_level,
        ::std::vector<float> &quantize_table, ::std::vector<unsigned char> &quantize_index);
    static size_t alignSize(size_t sz, int n);
};

inline size_t ModelBin::alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

#if NCNN_STDIO
class ModelBinFromStdio : public ModelBin
{
public:
    // construct from file
    ModelBinFromStdio(FILE* binfp);

    virtual Mat load(int w, int type) const;
    virtual void dump(const float* data, int dataSize, bool needToQuantize = false, int quantizeLevel = 0) const override;

protected:
    FILE* binfp;
};

class ModelBinFromStream : public ModelBin
{
public:
    ModelBinFromStream(std::istream& is);
    ModelBinFromStream(std::ostream& os);
    ModelBinFromStream(std::istream& is, std::ostream& os);

    virtual Mat load(int w, int type) const;
    virtual void dump(const float* data, int dataSize, bool needToQuantize = false, int quantizeLevel = 0) const override;

protected:
    std::istream* _is;
    std::ostream* _os;
};
#endif // NCNN_STDIO

class ModelBinFromMemory : public ModelBin
{
public:
    // construct from external memory
    ModelBinFromMemory(const unsigned char*& mem);

    virtual Mat load(int w, int type) const;

protected:
    const unsigned char*& mem;
};

class ModelBinFromMatArray : public ModelBin
{
public:
    // construct from weight blob array
    ModelBinFromMatArray(const Mat* weights);

    virtual Mat load(int w, int type) const;

protected:
    mutable const Mat* weights;
};

} // namespace ncnn

#endif // NCNN_MODELBIN_H
