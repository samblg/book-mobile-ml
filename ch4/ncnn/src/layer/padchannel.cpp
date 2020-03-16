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

#include "padchannel.h"
#include "common/io/Serialize.h"

using authen::io::Deserialize;

namespace ncnn {

DEFINE_LAYER_CREATOR(PadChannel)

PadChannel::PadChannel()
{
    one_blob_only = true;
    support_inplace = false;
}

int PadChannel::load_param(const ParamDict& pd) {
    _num_channels_to_pad = pd.get(0, 0);

    return 0;
}

int PadChannel::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const float* bottom_data = bottom_blob;
    int channels = bottom_blob.c;
    int dim = bottom_blob.cstep;
    int channel_by_dim = channels * dim;

    int top_channels = channels + _num_channels_to_pad;
    top_blob.create(bottom_blob.w, bottom_blob.h, top_channels);
    float* top_data = top_blob;

    memcpy(top_data, bottom_data, channel_by_dim * sizeof(float));
    top_data += channel_by_dim;
    memset(top_data, 0, _num_channels_to_pad * dim * sizeof(float));

    return 0;
}

} // namespace ncnn
