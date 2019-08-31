#include "net.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdio.h>
#include <string.h>

Net::Net()
{
}

Net::~Net()
{
    Clear();
}

int32_t Net::RegisterCustomLayer(const char* type, LAYER_CREATOR_FUNC creator)
{
    int32_t typeindex = LayerToIndex(type);
    if (typeindex != -1)
    {
        fprintf(stderr, "can not register build-in layer type %s\n", type);
        return -1;
    }

    int32_t custom_index = CustomLayerToIndex(type);
    if (custom_index == -1)
    {
        struct LayerRegistryEntry entry = { type, creator };
        _customLayerRegistry.push_back(entry);
    }
    else
    {
        fprintf(stderr, "overwrite existing custom layer type %s\n", type);
        _customLayerRegistry[custom_index].name = type;
        _customLayerRegistry[custom_index].creator = creator;
    }

    return 0;
}

int32_t Net::RegisterCustomLayer(int32_t index, LAYER_CREATOR_FUNC creator)
{
    int32_t custom_index = index & ~LayerType::CustomBit;
    if (index == custom_index)
    {
        fprintf(stderr, "can not register build-in layer index %d\n", custom_index);
        return -1;
    }

    if ((int32_t)_customLayerRegistry.size() <= custom_index)
    {
        struct LayerRegistryEntry dummy = { "", 0 };
        _customLayerRegistry.resize(custom_index + 1, dummy);
    }

    if (_customLayerRegistry[custom_index].creator)
    {
        fprintf(stderr, "overwrite existing custom layer index %d\n", custom_index);
    }

    _customLayerRegistry[custom_index].creator = creator;
    return 0;
}

int32_t Net::LoadParam(const unsigned char* mem)
{
    if ((unsigned long)mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", mem);
        return 0;
    }

    const unsigned char* cmem = mem;

    int32_t magic = *(int32_t*)(cmem);
    cmem += 4;

    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return 0;
    }

    int32_t layerCount = *(int32_t*)(cmem);
    cmem += 4;

    int32_t blobCount = *(int32_t*)(cmem);
    cmem += 4;

    _layers.resize(layerCount);
    _blobs.resize(blobCount);

    ParamDict pd;
    for (int32_t i = 0; i < layerCount; i++)
    {
        int32_t typeIndex = *(int32_t*)cmem;
        cmem += 4;

        int32_t bottomCount = *(int32_t*)cmem;
        cmem += 4;

        int32_t topCount = *(int32_t*)cmem;
        cmem += 4;

        Layer* layer = CreateLayer(typeIndex);
        if (!layer)
        {
            int32_t customIndex = typeIndex & ~LayerType::CustomBit;
            layer = CreateCustomLayer(customIndex);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeIndex);
            Clear();
            return 0;
        }

        layer->_bottoms.resize(bottomCount);
        for (int32_t j = 0; j< bottomCount; j++)
        {
            int32_t bottomblobIndex = *(int32_t*)cmem;
            cmem += 4;

            Blob& blob = _blobs[bottomblobIndex];

            blob.consumers.push_back(i);

            layer->_bottoms[j] = bottomblobIndex;
        }

        layer->_tops.resize(topCount);
        for (int32_t j = 0; j < topCount; j++)
        {
            int32_t topblobIndex = *(int32_t*)cmem;
            cmem += 4;

            Blob& blob = _blobs[topblobIndex];

            blob.producer = i;
            layer->tops[j] = topblobIndex;
        }

        // layer specific params
        int32_t pdlr = pd.LoadParam(cmem);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict LoadParam failed\n");
            continue;
        }

        int32_t lr = layer->LoadParam(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer LoadParam failed\n");
            continue;
        }

        _layers[i] = layer;
    }

    return cmem - mem;
}

int32_t Net::LoadModel(const unsigned char* mem)
{
    if (_layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return 0;
    }

    if ((unsigned long)mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", mem);
        return 0;
    }

    const unsigned char* cmem = mem;
    ModelBinFromMemory mb(cmem);
    for (size_t i= 0; i < _layers.size(); i++)
    {
        Layer* layer = _layers[i];

        int32_t lRet = layer->LoadModel(mb);
        if (lRet != 0)
        {
            fprintf(stderr, "layer LoadModel failed\n");
            return -1;
        }
    }

    return cmem - mem;
}

void Net::Clear()
{
    _blobs.Clear();
    for (size_t i = 0; i < _layers.size(); i++)
    {
        delete _layers[i];
    }
    _layers.Clear();
}

Extractor Net::CreateExtractor() const
{
    return Extractor(this, _blobs.size());
}

int32_t Net::FindBlobIndexByName(const char* name) const
{
    for (size_t i = 0; i < _blobs.size(); i++)
    {
        const Blob& blob = _blobs[i];
        if (blob.name == name)
        {
            return i;
        }
    }

    fprintf(stderr, "FindBlobIndexByName %s failed\n", name);
    return -1;
}

int32_t Net::FindLayerIndexByName(const char* name) const
{
    for (size_t i = 0; i < _layers.size(); i++)
    {
        const Layer* layer = _layers[i];
        if (layer->name == name)
        {
            return i;
        }
    }

    fprintf(stderr, "FindLayerIndexByName %s failed\n", name);
    return -1;
}

int32_t Net::CustomLayerToIndex(const char* type)
{
    const int32_t customLayerRegistryEntryCount = _customLayerRegistry.size();
    for (int32_t i = 0; i < customLayerRegistryEntryCount; i++)
    {
        if (strcmp(type, _customLayerRegistry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* Net::CreateCustomLayer(const char* type)
{
    int32_t index = CustomLayerToIndex(type);
    if (index == -1)
        return 0;

    return CreateCustomLayer(index);
}

Layer* Net::CreateCustomLayer(int32_t index)
{
    const int32_t customLayerRegistryEntryCount = _customLayerRegistry.size();
    if (index < 0 || index >= customLayerRegistryEntryCount)
        return 0;

    LAYER_CREATOR_FUNC layerCreator = _customLayerRegistry[index].creator;
    if (!layerCreator)
        return 0;

    return layerCreator();
}

int32_t Net::Forward_layer(int32_t layerIndex, std::vector<Mat>& blobMats, Option& opt) const
{
    const Layer* layer = _layers[layerIndex];

    if (layer->_oneBlobOnly)
    {
        // load bottom blob
        int32_t bottomblobIndex = layer->_bottoms[0];
        int32_t topblobIndex = layer->_tops[0];

        if (blobMats[bottomblobIndex].dims == 0)
        {
            int32_t ret = ForwardLayer(_blobs[bottomblobIndex].producer, blobMats, opt);
            if (ret != 0)
                return ret;
        }

        Mat bottomBlob = blobMats[bottomblobIndex];

        Mat topBlob;
        int32_t ret = layer->Forward(bottomBlob, topBlob, opt);
        if (ret != 0)
            return ret;

        // store top blob
        blobMats[topblobIndex] = topBlob;
    }
    else
    {
        // load bottom _blobs
        std::vector<Mat> bottomBlobs;
        bottomBlobs.resize(layer->_bottoms.size());
        for (size_t i = 0; i < layer->_bottoms.size(); i++)
        {
            int32_t bottomBlobIndex = layer->_bottoms[i];

            if (blobMats[bottomBlobIndex].dims == 0)
            {
                int32_t ret = ForwardLayer(_blobs[bottomBlobIndex].producer, blobMats, opt);
                if (ret != 0)
                    return ret;
            }

            bottomBlobs[i] = blobMats[bottomBlobIndex];
        }

        std::vector<Mat> topBlobs;
        topBlobs.resize(layer->_tops.size());
        int32_t ret = layer->Forward(bottomBlobs, topBlobs, opt);
        if (ret != 0)
            return ret;

        // store top _blobs
        for (size_t i = 0; i< layer->_tops.size(); i++)
        {
            int32_t topBlobIndex = layer->_tops[i];
            blobMats[topBlobIndex] = topBlobs[i];
        }
    }

    return 0;
}

Extractor::Extractor(const Net* net, int32_t blobCount) : _net(net)
{
    _blobMats.resize(blobCount);
    _opt = GetDefaultOption();
}

int32_t Extractor::Input(int32_t blobIndex, const Mat& in)
{
    if (blobIndex < 0 || blobIndex >= (int32_t)_blobMats.size())
        return -1;

    _blobMats[blobIndex] = in;

    return 0;
}

int32_t Extractor::Extract(int32_t blobIndex, Mat& feat)
{
    if (blobIndex < 0 || blobIndex >= (int32_t)_blobMats.size())
        return -1;

    int32_t ret = 0;

    if (_blobMats[blobIndex].dims == 0)
    {
        int32_t layerIndex = net->_blobs[blobIndex].producer;
        ret = net->ForwardLayer(layerIndex, _blobMats, _opt);
    }

    feat = _blobMats[blobIndex];

    return ret;
}

int32_t Extractor::Input(const char* blobName, const Mat& in)
{
    int32_t blobIndex = net->FindBlobIndexByName(blobName);
    if (blobIndex == -1)
        return -1;

    _blobMats[blobIndex] = in;

    return 0;
}

int32_t Extractor::Extract(const char* blobName, Mat& feat)
{
    int32_t blobIndex = net->FindBlobIndexByName(blobName);
    if (blobIndex == -1)
        return -1;

    int32_t ret = 0;
    if (_blobMats[blobIndex].dims == 0)
    {
        int32_t layerIndex = net->_blobs[blobIndex].producer;
        ret = net->ForwardLayer(layerIndex, _blobMats, _opt);
    }

    feat = _blobMats[blobIndex];

    return ret;
}
