#include "Layer.h"

#include <stdio.h>
#include <string.h>
#include "cpu.h"

Layer::Layer()
{
}

Layer::~Layer()
{
}

int32_t Layer::LoadParam(const ParamDict& /*pd*/)
{
    return 0;
}

int32_t Layer::LoadModel(const ModelBin& /*mb*/)
{
    return 0;
}

int32_t Layer::Forward(const std::vector<Mat>& bottomBlobs, std::vector<Mat>& topBlobs) const
{
    if (!_supportInplace)
        return -1;

    topBlobs = bottomBlobs;
    for (int32_t i = 0; i < (int32_t)topBlobs.size(); i++)
    {
        topBlobs[i] = bottomBlobs[i].clone(opt.blob_allocator);
        if (topBlobs[i].empty())
            return -100;
    }

    return ForwardInplace(topBlobs, _opt);
}

int32_t Layer::Forward(const Mat& bottomBlob, Mat& topBlob) const
{
    if (!_supportInplace)
        return -1;

    topBlob = bottomBlob.clone(_opt.blob_allocator);
    if (topBlob.empty())
        return -100;

    return ForwardInplace(topBlob, _opt);
}

int32_t Layer::ForwardInplace(std::vector<Mat>& /*bottom_topBlobs*/) const
{
    return -1;
}

int32_t Layer::ForwardInplace(Mat& /*bottom_topBlob*/) const
{
    return -1;
}

static const LayerRegistryEntry layerRegistry[] =
{
#include "LayerRegistry.h"
};

static const int32_t layerRegistryEntryCount = sizeof(layerRegistry) / sizeof(LayerRegistryEntry);

int32_t LayerToIndex(const char* type)
{
    for (int32_t i = 0; i < layerRegistryEntryCount; i++)
    {
        if (strcmp(type, layerRegistry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* CreateLayer(const char* type)
{
    int32_t index = LayerToIndex(type);
    if (index == -1)
        return 0;

    return CreateLayer(index);
}

Layer* CreateLayer(int32_t index)
{
    if (index < 0 || index >= layerRegistryEntryCount)
        return 0;

    LAYER_CREATOR_FUNC layerCreator = layerRegistry[index].creator;
    if (!layerCreator)
        return 0;

    return layerCreator();
}
