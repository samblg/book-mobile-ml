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

#include "platform.h"
#include "layer.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    num_threads = get_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;
}

static Option g_default_option;

const Option& get_default_option()
{
    return g_default_option;
}

int set_default_option(const Option& opt)
{
    if (opt.num_threads <= 0)
    {
        fprintf(stderr, "invalid option num_threads %d\n", opt.num_threads);
        return -1;
    }

    g_default_option = opt;

    return 0;
}

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::loadBinaryParams(std::istream& is) {
    return 0;
}

int Layer::loadModel(std::istream& is) {
    return 0;
}

int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
{
    return -1;
}


#include "layer_declaration.h"

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

namespace LayerType {
#if NCNN_STRING
	struct TypeIndexPair {
		const char* name;
		int32_t typeIndex;
	};

	static const TypeIndexPair IndexMapper[] = {
        { "AbsVal", 0 },
        { "ArgMax", 1 },
        { "BatchNorm", 2 },
        { "Bias", 3 },
        { "BNLL", 4 },
        { "Concat", 5 },
        { "Convolution", 6 },
        { "Crop", 7 },
        { "Deconvolution", 8 },
        { "Dropout", 9 },
        { "Eltwise", 10 },
        { "ELU", 11 },
        { "Embed", 12 },
        { "Exp", 13 },
        { "Flatten", 14 },
        { "InnerProduct", 15 },
        { "Input", 16 },
        { "Log", 17 },
        { "LRN", 18 },
        { "MemoryData", 19 },
        { "MVN", 20 },
        { "Pooling", 21 },
        { "Power", 22 },
        { "PReLU", 23 },
        { "Proposal", 24 },
        { "Reduction", 25 },
        { "ReLU", 26 },
        { "Reshape", 27 },
        { "ROIPooling", 28 },
        { "Scale", 29 },
        { "Sigmoid", 30 },
        { "Slice", 31 },
        { "Softmax", 32 },
        { "Split", 33 },
        { "SPP", 34 },
        { "TanH", 35 },
        { "Threshold", 36 },
        { "Tile", 37 },
        { "RNN", 38 },
        { "LSTM", 39 },
        { "BinaryOp", 40 },
        { "UnaryOp", 41 },
        { "ConvolutionDepthWise", 42 },
        { "Padding", 43 },
        { "Squeeze", 44 },
        { "ExpandDims", 45 },
        { "Normalize", 46 },
        { "Permute", 47 },
        { "PriorBox", 48 },
        { "DetectionOutput", 49 },
        { "Interp", 50 },
        { "DeconvolutionDepthWise", 51 },
        { "ShuffleChannel", 52 },
        { "InstanceNorm", 53 },
        { "Clip", 54 },
        { "Reorg", 55 },
        { "YoloDetectionOutput", 56 },
        { "Quantize", 57 },
        { "Dequantize", 58 },
        { "PadChannel", 59 }
	};

	static const int IndexMapperSize = sizeof(IndexMapper) / sizeof(TypeIndexPair);
#endif // NCNN_STRING

} // namespace LayerType

#if NCNN_STRING
int layer_to_creator_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        //std::cout << layer_registry[i].name << std::endl;
        if (strcmp(type, layer_registry[i].name) == 0)
        {
            return i;
        }
    }

    fprintf(stderr, "layer %s not exists\n", type);
    return -1;
}

int layer_to_index(const char* type)
{
	for (int i = 0; i < LayerType::IndexMapperSize; i++)
	{
		if (strcmp(type, LayerType::IndexMapper[i].name) == 0)
		{
			return LayerType::IndexMapper[i].typeIndex;
		}
	}

	fprintf(stderr, "layer %s not exists\n", type);
	return -1;
}

const char* index_to_layer_type(int32_t typeIndex) {
	for (int i = 0; i < LayerType::IndexMapperSize; i++)
	{
		if (LayerType::IndexMapper[i].typeIndex == typeIndex)
		{
			return LayerType::IndexMapper[i].name;
		}
	}

	fprintf(stderr, "layer type index %s not exists\n", typeIndex);
	return nullptr;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}

#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
    {
        fprintf(stderr, "layer index %d not exists\n", index);
        return 0;
    }

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
    {
        fprintf(stderr, "layer index %d not enabled\n", index);
        return 0;
    }

	//std::cout << index << " " << layer_registry[index].name << ": ";

    return layer_creator();
}

} // namespace ncnn
