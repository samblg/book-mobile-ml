#ifndef LAYER_TYPE_H
#define LAYER_TYPE_H

namespace LayerType {
enum
{
#include "LayerTypeEnum.h"
    CustomBit = (1<<8),
};
}

#endif
