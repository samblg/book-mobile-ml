#ifndef NET_H
#define NET_H

#include <stdio.h>
#include <vector>
#include "Blob.h"
#include "Layer.h"
#include "mat.h"

class Extractor;
class Net
{
public:
    Net();
    virtual ~Net();

    int32_t RegisterCustomLayer(const char* type, LAYER_CREATOR_FUNC creator);
    int32_t RegisterCustomLayer(int32_t index, LAYER_CREATOR_FUNC creator);

    int32_t LoadParam(const unsigned char* mem);
    int32_t LoadModel(const unsigned char* mem);

    void Clear();
    Extractor CreateExtractor() const;

protected:
    friend class Extractor;

    int32_t FindBlobIndexByName(const char* name) const;
    int32_t FindLayerIndexByName(const char* name) const;
    int32_t CustomLayerToIndex(const char* type);
    Layer* CreateCustomLayer(const char* type);

    Layer* CreateCustomLayer(int32_t index);
    int32_t ForwardLayer(int32_t layerIndex, std::vector<Mat>& blobMats, Option& opt) const;

protected:
    std::vector<Blob> _blobs;
    std::vector<Layer*> _layers;

    std::vector<LayerRegistryEntry> _customLayerRegistry;
};

class Extractor
{
public:
    int32_t Input(const char* blobName, const Mat& in);
    int32_t Extract(const char* blobName, Mat& feat);

    int32_t Input(int32_t blobIndex, const Mat& in);
    int32_t Extract(int32_t blobIndex, Mat& feat);

protected:
    friend Extractor Net::CreateExtractor() const;
    Extractor(const Net* net, int32_t blobCount);

private:
    const Net* _net;
    std::vector<Mat> _blobMats;
    Option _opt;
};

#endif

