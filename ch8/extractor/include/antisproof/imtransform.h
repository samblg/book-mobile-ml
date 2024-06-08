#pragma once

#include "common/AmType.h"

#include <stdio.h>
#include <fstream>
#include <vector>
#include "common/minicv/minicv.h"

using namespace std;

#define MAX_LANDMARKS_NUM 27
#define STD_WIDTH 148
#define STD_HEIGHT 148

minicv::Mat uchar2Mat(uchar *buffer,int width,int height);

void Mat2uchar(minicv::Mat img,uchar* dst,int width,int height);

void sim_params_from_points(const TPointF dstKeyPoints[],  
	const TPointF srcKeyPoints[], int count,float* a, float* b, float* tx, float* ty);

void sim_transform_landmark(const TLandmark1* landmark, TPointF* dst, 
	int count, float a, float b, float tx, float ty);

void sim_transform_image(const unsigned char* gray, int width, int height, int pitch,
	unsigned char* dst, int width1, int height1,float a, float b, float tx, float ty);

minicv::Mat sim_transform_image_1channel_anti(minicv::Mat img, int left_eye_x,
					int left_eye_y, int right_eye_x, int right_eye_y);

minicv::Mat sim_transform_image_3channels_anti(minicv::Mat img, int left_eye_x,
					int left_eye_y, int right_eye_x, int right_eye_y);
