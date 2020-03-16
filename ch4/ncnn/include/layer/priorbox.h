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

#ifndef LAYER_PRIORBOX_H
#define LAYER_PRIORBOX_H

#include "layer.h"
#include <vector>

namespace ncnn {

class PriorBox : public Layer
{
public:
    PriorBox();

    void init();
    virtual int load_param(const ParamDict& pd);
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    std::vector<float> matToVector(const Mat& mat);

public:
    // min_sizes
    std::vector<float> min_sizes_;
    // max_sizes
    std::vector<float> max_sizes_;
    // aspect_ratios
    std::vector<float> aspect_ratios_;
    // flip
    bool flip_;
    int num_priors_;
    // clip
    bool clip_;
    // variances
    std::vector<float> variance_;

    // image_width
    int img_w_;
    // image_height
    int img_h_;
    // step_width
    float step_w_;
    // step_height
    float step_h_;

    // offset
    float offset_;
};

} // namespace ncnn

#endif // LAYER_PRIORBOX_H
