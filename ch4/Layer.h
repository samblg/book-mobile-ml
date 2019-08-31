#ifndef LAYER_H
#define LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include <cstdint>
#include "mat.h"
#include "modelbin.h"
#include "paramdict.h"

class Layer
{
public:
    Layer();
    virtual ~Layer();

    virtual int32_t LoadParam(const ParamDict& pd);
    virtual int32_t LoadModel(const ModelBin& mb);

public:
    virtual int32_t Forward(const std::vector<Mat>& bottomBlobs, std::vector<Mat>& topBlobs) const;
    virtual int32_t Forward(const Mat& bottomBlob, Mat& topBlob) const;
    virtual int32_t ForwardInplace(std::vector<Mat>& bottomTopBlobs) const;
    virtual int32_t ForwardInplace(Mat& bottomTopBlob) const;

protected:
    bool _oneBlobOnly {false};
    bool _supportInplace {false};
    std::string _type;
    std::string _name;
    std::vector<int32_t> _bottoms;
    std::vector<int32_t> _tops;
};

typedef Layer* (*LAYER_CREATOR_FUNC)();

struct LayerRegistryEntry
{
    const char* name;
    LAYER_CREATOR_FUNC creator;
};

int32_t LayerToIndex(const char* type);
Layer* CreateLayer(const char* type);
Layer* CreateLayer(int32_t index);

#define DEFINE_LAYER_CREATOR(NAME) \
    Layer* NAME##LayerCreator() { return new NAME; }

#endif

