/*
Author: Bi Sai 
Date: 2014/06/18
This program is a reimplementation of algorithms in "Face Alignment by Explicit 
Shape Regression" by Cao et al.
If you find any bugs, please email me: soundsilencebisai-at-gmail-dot-com

Copyright (c) 2014 Bi Sai 
The MIT License (MIT)
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include "common/minicv/minicv.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility>

using namespace std;

class BoundingBox{
    public:
        double start_x;
        double start_y;
        double width;
        double height;
        double centroid_x;
        double centroid_y;
        BoundingBox(){
            start_x = 0;
            start_y = 0;
            width = 0;
            height = 0;
            centroid_x = 0;
            centroid_y = 0;
        }; 
};


class Fern{
    private:
        int fern_pixel_num_;
        int landmark_num_;
        minicv::Mat_<int> selected_nearest_landmark_index_;
        minicv::Mat_<double> threshold_;
        minicv::Mat_<int> selected_pixel_index_;
        minicv::Mat_<double> selected_pixel_locations_;
        vector<minicv::Mat_<double> > bin_output_;
    public:
        vector<minicv::Mat_<double> > Train(const vector<vector<double> >& candidate_pixel_intensity,
                                    const minicv::Mat_<double>& covariance,
                                    const minicv::Mat_<double>& candidate_pixel_locations,
                                    const minicv::Mat_<int>& nearest_landmark_index,
                                    const vector<minicv::Mat_<double> >& regression_targets,
                                    int fern_pixel_num);
        minicv::Mat_<double> Predict(const minicv::Mat_<uchar>& image,
                             const minicv::Mat_<double>& shape,
                             const minicv::Mat_<double>& rotation,
                             const BoundingBox& bounding_box,
                             double scale);
        void Read(ifstream& fin);
		void ReadBinary(ifstream& fin);
		void ReadBinary(ifstream& fin, float norm_coeff);
        void Write(ofstream& fout);
		void WriteBinary(ofstream& fout);
};

class FernCascade{
    public:
        vector<minicv::Mat_<double> > Train(const vector<minicv::Mat_<uchar> >& images,
                                    const vector<minicv::Mat_<double> >& current_shapes,
                                    const vector<minicv::Mat_<double> >& ground_truth_shapes,
                                    const vector<BoundingBox> & bounding_box,
                                    const minicv::Mat_<double>& mean_shape,
                                    int second_level_num,
                                    int candidate_pixel_num,
                                    int fern_pixel_num);  
        minicv::Mat_<double> Predict(const minicv::Mat_<uchar>& image,
                          const BoundingBox& bounding_box, 
                          const minicv::Mat_<double>& mean_shape,
                          const minicv::Mat_<double>& shape);
        void Read(ifstream& fin);
		void ReadBinary(ifstream& fin);
		void ReadBinary(ifstream& fin, float norm_coeff);
        void Write(ofstream& fout);
		void WriteBinary(ofstream& fout);
    private:
        vector<Fern> ferns_;
        int second_level_num_;
};

class ShapeRegressor{
    public:
        ShapeRegressor(); 
        void Train(const vector<minicv::Mat_<uchar> >& images,
                   const vector<minicv::Mat_<double> >& ground_truth_shapes,
                   const vector<BoundingBox>& bounding_box,
                   int first_level_num, int second_level_num,
                   int candidate_pixel_num, int fern_pixel_num,
                   int initial_num);
        minicv::Mat_<double> Predict(const minicv::Mat_<uchar>& image, const BoundingBox& bounding_box, int initial_num);
        void Read(ifstream& fin);
		void ReadBinary(ifstream& fin);
		void ReadBinary(ifstream& fin, float norm_coeff);
        void Write(ofstream& fout);
		void WriteBinary(ofstream& fout);
        bool Load(string path);
        void Save(string path);
    private:
        int first_level_num_;
        int landmark_num_;
        vector<FernCascade> fern_cascades_;
        minicv::Mat_<double> mean_shape_;
        vector<minicv::Mat_<double> > training_shapes_;
        vector<BoundingBox> bounding_box_;
};

minicv::Mat_<double> GetMeanShape(const vector<minicv::Mat_<double> >& shapes,
                          const vector<BoundingBox>& bounding_box);
minicv::Mat_<double> ProjectShape(const minicv::Mat_<double>& shape, const BoundingBox& bounding_box);
minicv::Mat_<double> ReProjectShape(const minicv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const minicv::Mat_<double>& shape1, const minicv::Mat_<double>& shape2,
                         minicv::Mat_<double>& rotation,double& scale);
double calculate_covariance(const vector<double>& v_1, 
                            const vector<double>& v_2);
#endif
