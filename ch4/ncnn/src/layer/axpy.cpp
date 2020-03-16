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

#include "axpy.h"

namespace ncnn {

    DEFINE_LAYER_CREATOR(Axpy)

    Axpy::Axpy()
    {
        
    }

    int Axpy::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
    {
        int w = bottom_blobs[1].w;
        int h = bottom_blobs[1].h;
        int channels = bottom_blobs[0].c;
        int size = w * h;

        top_blobs[0].create(w, h, channels);

        //Y = Y + X * scale;
#pragma omp parallel for
        for (int q = 0; q<channels; q++)
        {
            const float* scale_ptr = bottom_blobs[0].channel(q);
            const float* X_ptr = bottom_blobs[1].channel(q);           
            const float* Y_ptr = bottom_blobs[2].channel(q);

            float* outptr = top_blobs[0].channel(q);

            for (int i = 0; i<size; i++)
            {
                outptr[i] = Y_ptr[i] + (X_ptr[i] * scale_ptr[0]);
               // std::cout << outptr[i] << ",";
            }
        }

        //DEBUG
        float* tempoutptr = top_blobs[0].channel(0);
        const float* tempscale_ptr = bottom_blobs[0].channel(0);
        const float* tempX_ptr = bottom_blobs[1].channel(0);
        const float* tempY_ptr = bottom_blobs[2].channel(0);
        //for (int i = 0; i < 20; i++)
        //{
        //    std::cout << tempscale_ptr[i] << ", ";
        //}
        //std::cout << std::endl;
        //std::cout << std::endl;
        //for (int i = 0; i < 20; i++)
        //{
        //    std::cout << tempX_ptr[i] << ",";
        //}
        //std::cout << std::endl;
        //std::cout << std::endl;
        //for (int i = 0; i < 20; i++)
        //{
        //    std::cout << tempY_ptr[i] << ",";
        //}
        //std::cout << std::endl;
        //std::cout << std::endl;
        //for (int i = 0; i < 20; i++)
        //{
        //    std::cout << tempoutptr[i] << ",";
        //}
        //std::cout << std::endl;
        //std::cout << "================================" << std::endl;
        //getchar();
        return 0;
    }

//    int Axpy::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
//    {
//        int w = bottom_blobs[0].w;
//        int h = bottom_blobs[0].h;
//        int channels = bottom_blobs[0].c;
//        int size = w * h;
//
//        top_blobs[0].create(w, h, channels);
//
//        //Y = Y + X * scale;
//#pragma omp parallel for
//        for (int q = 0; q<channels; q++)
//        {
//            const float* scale_ptr = bottom_blobs[0].channel(q);
//            const float* X_ptr = bottom_blobs[1].channel(q);
//            const float* Y_ptr = bottom_blobs[2].channel(q);
//
//            float* outptr = top_blobs[0].channel(q);
//
//            for (int i = 0; i<size; i++)
//            {
//                outptr[i] += (X_ptr[i] * scale_ptr[i]);
//                std::cout << outptr[i] << ",";
//            }
//        }
//
//        return 0;
//    }


} // namespace ncnn
