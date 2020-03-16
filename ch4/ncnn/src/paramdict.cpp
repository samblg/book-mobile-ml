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

#include <ctype.h>
#include <iostream>
#include "paramdict.h"
#include "common/io/Serialize.h"
#include "platform.h"

namespace ncnn {

ParamDict::ParamDict()
{
    clear();
}

int ParamDict::get(int id, int def) const
{
    return params[id].loaded ? params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return params[id].loaded ? params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return params[id].loaded ? params[id].v : def;
}

void ParamDict::set(int id, int i)
{
    params[id].type = ParamType::Integer;
    params[id].loaded = 1;
    params[id].i = i;
}

void ParamDict::set(int id, unsigned int i)
{
    params[id].type = ParamType::Integer;
    params[id].loaded = 1;
    params[id].i = static_cast<int>(i);
}

void ParamDict::set(int id, long i)
{
    params[id].type = ParamType::Integer;
    params[id].loaded = 1;
    params[id].i = static_cast<int>(i);
}

void ParamDict::set(int id, long long i)
{
    params[id].type = ParamType::Integer;
    params[id].loaded = 1;
    params[id].i = static_cast<int>(i);
}

void ParamDict::set(int id, float f)
{
    params[id].type = ParamType::Float;
    params[id].loaded = 1;
    params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    params[id].type = ParamType::Array;
    params[id].loaded = 1;
    params[id].v = v;
}

void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        params[i].type = ParamType::Unknown;
        params[i].loaded = 0;
        params[i].v = Mat();
    }
}

#if NCNN_STDIO
#if NCNN_STRING
static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j=0; j<16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

int ParamDict::dumpParams(FILE* fp) const
{
    for (int paramIndex = 0; paramIndex != NCNN_MAX_PARAM_COUNT; ++ paramIndex) {
        auto& param = params[paramIndex];
        if (!param.loaded) {
            continue;
        }

        if (param.type == ParamType::Array) {
            int id = -23300 - paramIndex;

            const int elementCount = param.v.c * param.v.h * param.v.w;
            fprintf(fp, " %d=%d", id, elementCount);
            std::string elementFormat;
            if (param.v.elementType == Mat::ElementType::Float) {
                const float* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    fprintf(fp, ",%f", paramData[elementIndex]);
                }
            }
            else if (param.v.elementType == Mat::ElementType::Integer) {
                const int* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    fprintf(fp, ",%d", paramData[elementIndex]);
                }
            }
            else {
                std::cerr << "The element type of param " << paramIndex << " is unknown.";

                return -1;
            }
        }
        else if (param.type == ParamType::Float) {
            fprintf(fp, " %d=%f", paramIndex, param.f);
        }
        else if (param.type == ParamType::Integer) {
            fprintf(fp, " %d=%d", paramIndex, param.i);
        }
        else {
            std::cerr << "The type of param " << paramIndex << " is unknown.";
            
            return -1;
        }
    }

    return 0;
}

// Params format: <param count> [, <param>]
// Param format: <param id> <param type code> <param value>  
int ParamDict::dumpParamsToBinary(FILE* fp) const {
    using authen::io::Serialize;

    int loadedParamCount = 0;
    for (int paramIndex = 0; paramIndex != NCNN_MAX_PARAM_COUNT; ++ paramIndex) {
        auto& param = params[paramIndex];

        if (param.loaded) {
            ++ loadedParamCount;
        }
    }
    Serialize(fp, loadedParamCount);

    for (int paramIndex = 0; paramIndex != NCNN_MAX_PARAM_COUNT; ++ paramIndex) {
        auto& param = params[paramIndex];
        if (!param.loaded) {
            continue;
        }

        Serialize(fp, paramIndex);

        if (param.type == ParamType::Array) {
            Serialize(fp, ParamTypeCode::Array);

            // Array param value format
            // <element count> <element type> [, <element>]
            const int elementCount = param.v.c * param.v.h * param.v.w;
            Serialize(fp, elementCount);
            std::string elementFormat;
            if (param.v.elementType == Mat::ElementType::Float) {
                Serialize(fp, Mat::ElementTypeCode::Float);
                const float* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    Serialize(fp, paramData[elementIndex]);
                }
            }
            else if (param.v.elementType == Mat::ElementType::Integer) {
                Serialize(fp, Mat::ElementTypeCode::Integer);
                const int* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    Serialize(fp, paramData[elementIndex]);
                }
            }
            else {
                std::cerr << "The element type of param " << paramIndex << " is unknown.";

                return -1;
            }
        }
        else if (param.type == ParamType::Float) {
            Serialize(fp, ParamTypeCode::Float);
            Serialize(fp, param.f);
        }
        else if (param.type == ParamType::Integer) {
            Serialize(fp, ParamTypeCode::Integer);
            Serialize(fp, param.i);
        }
        else {
            std::cerr << "The type of param " << paramIndex << " is unknown.";

            return -1;
        }
    }

    return 0;
}

int ParamDict::dumpParamsToBinary(std::ostream& os) const {
    using authen::io::Serialize;

    int loadedParamCount = 0;
    for (int paramIndex = 0; paramIndex != NCNN_MAX_PARAM_COUNT; ++ paramIndex) {
        auto& param = params[paramIndex];

        if (!param.loaded) {
            ++ loadedParamCount;
        }
    }
    Serialize(os, loadedParamCount);

    for (int paramIndex = 0; paramIndex != NCNN_MAX_PARAM_COUNT; ++ paramIndex) {
        auto& param = params[paramIndex];
        if (!param.loaded) {
            continue;
        }

        Serialize(os, paramIndex);

        if (param.type == ParamType::Array) {
            Serialize(os, ParamTypeCode::Array);

            // Array param value format
            // <element count> <element type> [, <element>]
            const int elementCount = param.v.c * param.v.h * param.v.w;
            Serialize(os, elementCount);
            std::string elementFormat;
            if (param.v.elementType == Mat::ElementType::Float) {
                Serialize(os, Mat::ElementTypeCode::Float);
                const float* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    Serialize(os, paramData[elementIndex]);
                }
            }
            else if (param.v.elementType == Mat::ElementType::Integer) {
                Serialize(os, Mat::ElementTypeCode::Integer);
                const int* paramData = param.v;
                for (int elementIndex = 0; elementIndex < elementCount; ++ elementIndex) {
                    Serialize(os, paramData[elementIndex]);
                }
            }
            else {
                std::cerr << "The element type of param " << paramIndex << " is unknown.";

                return -1;
            }
        }
        else if (param.type == ParamType::Float) {
            Serialize(os, ParamTypeCode::Float);
            Serialize(os, param.f);
        }
        else if (param.type == ParamType::Integer) {
            Serialize(os, ParamTypeCode::Integer);
            Serialize(os, param.i);
        }
        else {
            std::cerr << "The type of param " << paramIndex << " is unknown.";

            return -1;
        }
    }

    return 0;
}

int ParamDict::load_param(FILE* fp)
{
    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (fscanf(fp, "%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            int nscan = fscanf(fp, "%d", &len);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v.create(len);
            params[id].type = ParamType::Array;

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element fail\n");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float* ptr = params[id].v;
                    nscan = sscanf(vstr, "%f", &ptr[j]);
                }
                else
                {
                    int* ptr = params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                }
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict parse array element fail\n");
                    return -1;
                }
            }
        }
        else
        {
            char vstr[16];
            int nscan = fscanf(fp, "%15s", vstr);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float) {
                params[id].type = ParamType::Float;
                nscan = sscanf(vstr, "%f", &params[id].f);
            }
            else {
                params[id].type = ParamType::Integer;
                nscan = sscanf(vstr, "%d", &params[id].i);
            }
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

        params[id].loaded = 1;
    }

    return 0;
}
#endif // NCNN_STRING

int ParamDict::load_param_bin(FILE* fp)
{
    clear();

//     binary 0
//     binary 100
//     binary 1
//     binary 1.250000
//     binary 3 | array_bit
//     binary 5
//     binary 0.1
//     binary 0.2
//     binary 0.4
//     binary 0.8
//     binary 1.0
//     binary -233(EOP)

    int id = 0;
    fread(&id, sizeof(int), 1, fp);

    while (id != -233)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            fread(&len, sizeof(int), 1, fp);

            params[id].v.create(len);

            float* ptr = params[id].v;
            fread(ptr, sizeof(float), len, fp);
        }
        else
        {
            fread(&params[id].f, sizeof(float), 1, fp);
        }

        params[id].loaded = 1;

        fread(&id, sizeof(int), 1, fp);
    }

    return 0;
}
#endif // NCNN_STDIO

int ParamDict::loadParamsFromBinary(FILE* fp) {
    using authen::io::Deserialize;

    clear();

    int paramCount = 0;
    Deserialize(fp, paramCount);

    for (int paramIndex = 0; paramIndex != paramCount; ++ paramIndex) {
        int paramId = 0;
        Deserialize(fp, paramId);

        int paramTypeCode = -1;
        Deserialize(fp, paramTypeCode);

        if (paramTypeCode == ParamDict::ParamTypeCode::Integer) {
            params[paramId].type = ParamDict::ParamType::Integer;
            Deserialize(fp, params[paramId].i);
        }
        else if (paramTypeCode == ParamDict::ParamTypeCode::Float) {
            params[paramId].type = ParamDict::ParamType::Float;
            Deserialize(fp, params[paramId].f);
        }
        else if (paramTypeCode == ParamDict::ParamTypeCode::Array) {
            params[paramId].type = ParamDict::ParamType::Array;

            int elementCount = 0;
            Deserialize(fp, elementCount);

            int elementTypeCode = -1;
            Deserialize(fp, elementTypeCode);

            Mat elementValue(elementCount);
            if (elementTypeCode == Mat::ElementTypeCode::Float) {
                elementValue.elementType == Mat::ElementType::Float;
                float* paramData = elementValue;
                Deserialize(fp, paramData, elementCount);
                params[paramId].v = elementValue;
            }
            else if (elementTypeCode == Mat::ElementTypeCode::Integer) {
                elementValue.elementType == Mat::ElementType::Integer;
                int* paramData = elementValue;
                Deserialize(fp, paramData, elementCount);
                params[paramId].v = elementValue;
            }
            else {
                std::cerr << "The element type of param " << paramId << " fp unknown.";

                return -1;
            }
        }
        else {
            std::cerr << "The type of param " << paramId << " fp unknown.";

            return -1;
        }

        params[paramId].loaded = 1;
    }

    return 0;
}

int ParamDict::loadParamsFromBinary(std::istream& is) {
    using authen::io::Deserialize;

    clear();

    int paramCount = 0;
    Deserialize(is, paramCount);

    for (int paramIndex = 0; paramIndex != paramCount; ++ paramIndex) {
        int paramId = 0;
        Deserialize(is, paramId);

        int paramTypeCode = -1;
        Deserialize(is, paramTypeCode);

        if (paramTypeCode == ParamDict::ParamTypeCode::Integer) {
            params[paramId].type = ParamDict::ParamType::Integer;
            Deserialize(is, params[paramId].i);
        }
        else if (paramTypeCode == ParamDict::ParamTypeCode::Float) {
            params[paramId].type = ParamDict::ParamType::Float;
            Deserialize(is, params[paramId].f);
        }
        else if (paramTypeCode == ParamDict::ParamTypeCode::Array) {
            params[paramId].type = ParamDict::ParamType::Array;

            int elementCount = 0;
            Deserialize(is, elementCount);

            int elementTypeCode = -1;
            Deserialize(is, elementTypeCode);

            Mat elementValue(elementCount);
            if (elementTypeCode == Mat::ElementTypeCode::Float) {
                elementValue.elementType == Mat::ElementType::Float;
                float* paramData = elementValue;
                Deserialize(is, paramData, elementCount);
                params[paramId].v = elementValue;
            }
            else if (elementTypeCode == Mat::ElementTypeCode::Integer) {
                elementValue.elementType == Mat::ElementType::Integer;
                int* paramData = elementValue;
                Deserialize(is, paramData, elementCount);
                params[paramId].v = elementValue;
            }
            else {
                std::cerr << "The element type of param " << paramId << " is unknown.";

                return -1;
            }
        }
        else {
            std::cerr << "The type of param " << paramId << " is unknown.";

            return -1;
        }

        params[paramId].loaded = 1;
    }

    return 0;
}

int ParamDict::load_param(const unsigned char*& mem)
{
    clear();

    int id = *(int*)(mem);
    mem += 4;

    while (id != -233)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = *(int*)(mem);
            mem += 4;

            params[id].v.create(len);

            memcpy(params[id].v.data, mem, len * 4);
            mem += len * 4;
        }
        else
        {
            params[id].f = *(float*)(mem);
            mem += 4;
        }

        params[id].loaded = 1;

        id = *(int*)(mem);
        mem += 4;
    }

    return 0;
}

int ParamDict::getParamsCount() const {
    int paramsCount = 0;
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i ++) {
        if (params[i].loaded) {
            paramsCount ++;
        }
    }

    return paramsCount;
}

} // namespace ncnn
