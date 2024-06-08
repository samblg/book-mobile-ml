#ifndef POSE_H
#define POSE_H

#include "common/AmType.h"

typedef struct tagPoint3DF {
	float x;
	float y;
	float z;
} TPoint3DF;

typedef TLandmark1 egp_NodePtrArr;

float PoseEstimation(const TLandmarks1* landmarks, const int* ids, const TPoint3DF* kps, float* roll, float* yaw, float* pitch);
float PoseEstimation(const egp_NodePtrArr* nodes, int pt_num, const int* ids, const TPoint3DF* kps, float* roll, float* yaw, float* pitch);
void PoseEstimation(const egp_NodePtrArr* nodes, int pt_num, const TPoint3DF* kps, float* roll, float* yaw, float* pitch);

#endif
