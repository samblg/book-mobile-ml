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

#include "priorbox.h"
#include <algorithm>
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(PriorBox)

PriorBox::PriorBox()
{
    one_blob_only = false;
    support_inplace = false;
}

void PriorBox::init()
{
    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.);
    if (min_sizes_.size() == 6)
        num_priors_ = 24;                                                            //slimzf
    else if (min_sizes_.size() == 5)
        num_priors_ = 23;
    else if (min_sizes_.size() == 3)
        num_priors_ = 21;
    else
        num_priors_ = aspect_ratios_.size() * min_sizes_.size();
}

int PriorBox::load_param(const ParamDict& pd)
{
    Mat min_sizes = pd.get(0, Mat());
    Mat max_sizes = pd.get(1, Mat());
    Mat aspect_ratios = pd.get(2, Mat());
    float variances[4];
    variances[0] = pd.get(3, 0.f);
    variances[1] = pd.get(4, 0.f);
    variances[2] = pd.get(5, 0.f);
    variances[3] = pd.get(6, 0.f);

    int flip = pd.get(7, 1);
    int clip = pd.get(8, 0);
    int image_width = pd.get(9, 0);
    int image_height = pd.get(10, 0);
    float step_width = pd.get(11, -233.f);
    float step_height = pd.get(12, -233.f);
    float offset = pd.get(13, 0.f);

    min_sizes_ = matToVector(min_sizes);
    max_sizes_ = matToVector(max_sizes);
    aspect_ratios_ = matToVector(aspect_ratios);
    flip_ = flip;
    num_priors_;
    clip_ = clip;
    variance_ = std::vector<float>(variances, variances + 4);

    img_w_ = image_width;
    img_h_ = image_height;
    step_w_ = step_width;
    step_h_ = step_height;

    offset_ = offset;

    init();

    return 0;
}

std::vector<float> PriorBox::matToVector(const Mat& mat) {
    return std::vector<float>(mat.floatData(), mat.floatData() + mat.c * mat.w * mat.h);
}

int PriorBox::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const int layer_width = bottom_blobs[0].w;
    const int layer_height = bottom_blobs[0].h;
    int img_width, img_height;
    //if (img_h_ == 0 || img_w_ == 0) {
    img_width = bottom_blobs[1].w;
    img_height = bottom_blobs[1].h;
    //} else {
    // img_width = img_w_;
    //img_height = img_h_;
    //}

    float step_w, step_h;
    if (step_w_ == 0 || step_h_ == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }
    else {
        step_w = step_w_;
        step_h = step_h_;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(layer_width * layer_height * num_priors_ * 4, 2);
    float* top_data = top_blob.floatData();
    int dim = layer_height * layer_width * num_priors_ * 4;
    int idx = 0;

    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width; ++w) {
            float center_x = (w + offset_) * step_w;
            float center_y = (h + offset_) * step_h;
            float box_width, box_height;
            for (int s = 0; s < min_sizes_.size(); ++s) {
                int min_size_ = min_sizes_[s];
                if (min_size_ == 16) {
                    for (int i = -4; i<4; i++) {
                        for (int j = -4; j<4; j++) {
                            box_width = box_height = min_size_;
                            top_data[idx++] = (center_x + j * 4 - (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 4 - (box_width - 1) / 2.) / img_height;
                            top_data[idx++] = (center_x + j * 4 + (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 4 + (box_width - 1) / 2.) / img_height;
                        }
                    }
                }
                else if (min_size_ == 32) {
                    for (int i = -2; i<2; i++) {
                        for (int j = -2; j<2; j++) {
                            box_width = box_height = min_size_;
                            top_data[idx++] = (center_x + j * 8 - (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 8 - (box_width - 1) / 2.) / img_height;
                            top_data[idx++] = (center_x + j * 8 + (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 8 + (box_width - 1) / 2.) / img_height;
                        }
                    }
                }
                else if (min_size_ == 64) {
                    for (int i = -1; i<1; i++) {
                        for (int j = -1; j<1; j++) {
                            box_width = box_height = min_size_;
                            top_data[idx++] = (center_x + j * 16 - (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 16 - (box_width - 1) / 2.) / img_height;
                            top_data[idx++] = (center_x + j * 16 + (box_width - 1) / 2.) / img_width;
                            top_data[idx++] = (center_y + i * 16 + (box_width - 1) / 2.) / img_height;
                        }
                    }
                }
                else {
                    box_width = box_height = min_size_;
                    // xmin
                    top_data[idx++] = (center_x - (box_width - 1) / 2.) / img_width;
                    // ymin
                    top_data[idx++] = (center_y - (box_width - 1) / 2.) / img_height;
                    // xmax
                    top_data[idx++] = (center_x + (box_width - 1) / 2.) / img_width;
                    // ymax
                    top_data[idx++] = (center_y + (box_width - 1) / 2.) / img_height;
                }

                if (max_sizes_.size() > 0) {
                    int max_size_ = max_sizes_[s];
                    box_width = box_height = sqrt(min_size_ * max_size_);
                    top_data[idx++] = (center_x - box_width / 2.) / img_width;
                    top_data[idx++] = (center_y - box_height / 2.) / img_height;
                    top_data[idx++] = (center_x + box_width / 2.) / img_width;
                    top_data[idx++] = (center_y + box_height / 2.) / img_height;
                }

                for (int r = 0; r < aspect_ratios_.size(); ++r) {
                    float ar = aspect_ratios_[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }
                    box_width = min_size_ * sqrt(ar);
                    box_height = min_size_ / sqrt(ar);
                    top_data[idx++] = (center_x - box_width / 2.) / img_width;
                    top_data[idx++] = (center_y - box_height / 2.) / img_height;
                    top_data[idx++] = (center_x + box_width / 2.) / img_width;
                    top_data[idx++] = (center_y + box_height / 2.) / img_height;
                }
            }
        }
    }

    top_data += top_blob.w;
    if (variance_.size() == 1) {
        //caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
    }
    else {
        int count = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int i = 0; i < num_priors_; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        top_data[count] = variance_[j];
                        ++count;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
