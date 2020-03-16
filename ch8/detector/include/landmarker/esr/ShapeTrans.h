#pragma once

#include "landmarker/esr/PreDefinedFile.h"
#include "math.h"
#include "string.h"
#include "float.h"
#include <iostream>

extern const SFTransStruct _SFTrans[65536];
inline void NormShape(POINT_FLOAT *pShapeMat, int pNumSample, int pNumPnt, POINT_FLOAT *pNormShapeMat, AUTHEN_RECT pBoundBox, POINT_FLOAT *pMeanShape)
{
	POINT_FLOAT *pShape = NULL;
	POINT_FLOAT *pShapeSubMat = NULL;
	POINT_FLOAT *pNormShapeSubMat = NULL;
	POINT_FLOAT tCenter;
	float tX_half = 0.0f;
	float tY_half = 0.0f;
	float tX_half_inv = 0.0f;
	float tY_half_inv = 0.0f;
	for (int i = 0; i<pNumSample; ++i)
	{
		pShapeSubMat = pShapeMat + i*pNumPnt;
		pNormShapeSubMat = pNormShapeMat + i*pNumPnt;

		tX_half = pBoundBox.width / 2.0f;
		tY_half = pBoundBox.height / 2.0f;
		tX_half_inv = 1.0f / tX_half;
		tY_half_inv = 1.0f / tY_half;
		tCenter.x = pBoundBox.left + tX_half;
		tCenter.y = pBoundBox.top + tY_half;

		for (int j = 0; j<pNumPnt; ++j)
		{
			pNormShapeSubMat[j].x = (pShapeSubMat[j].x - tCenter.x) * tX_half_inv;
			pNormShapeSubMat[j].y = (pShapeSubMat[j].y - tCenter.y) * tY_half_inv;
		}
	}

	memset(pMeanShape, 0, sizeof(POINT_FLOAT) * pNumPnt);
	for (int i = 0; i<pNumSample; ++i)
	{
		pShape = pNormShapeMat + i * pNumPnt;
		for (int j = 0; j<pNumPnt; ++j)
		{
			pMeanShape[j].x += pNormShapeMat[j].x;
			pMeanShape[j].y += pNormShapeMat[j].y;
		}
	}
	float ratio = 1.0f / pNumSample;
	for (int i = 0; i<pNumPnt; ++i)
	{
		pMeanShape[i].x *= ratio;
		pMeanShape[i].y *= ratio;
	}
};
inline void InverseNorm(const POINT_FLOAT *pNormShape, AUTHEN_RECT pBoundBox, int pNumPnt, POINT_FLOAT *pInvShape)
{
	POINT_FLOAT tCenter;
	float tX_half = pBoundBox.width / 2;
	float tY_half = pBoundBox.height / 2;
	tCenter.x = pBoundBox.left + tX_half;
	tCenter.y = pBoundBox.top + tY_half;

	for (int i = 0; i<pNumPnt; ++i)
	{
		pInvShape[i].x = pNormShape[i].x * tX_half + tCenter.x;
		pInvShape[i].y = pNormShape[i].y * tY_half + tCenter.y;
	}
};
inline void InverseNorm_SHORT(const POINT_SHORT *pNormShape, AUTHEN_RECT pBoundBox, int pNumPnt, POINT_FLOAT *pInvShape)
{
	POINT_FLOAT tCenter;
	float tX_half = pBoundBox.width / 2;
	float tY_half = pBoundBox.height / 2;
	tCenter.x = pBoundBox.left + tX_half;
	tCenter.y = pBoundBox.top + tY_half;

	for (int i = 0; i < pNumPnt; ++i)
	{
		pInvShape[i].x = _SFTrans[(unsigned short)(pNormShape[i].x)].floatdata * tX_half + tCenter.x;
		pInvShape[i].y = _SFTrans[(unsigned short)(pNormShape[i].y)].floatdata * tY_half + tCenter.y;
	}
};

inline void PermuteShape(const POINT_SHORT *pNormShape, AUTHEN_RECT pBoundRect, int pNumSamples, int pNumPnts, POINT_FLOAT *pPermuteShapeSet, int pPermuteNum, int *pImgId)
{
	if (pPermuteNum == 1 || RANDOM_INITIAL == 0)
	{
		for (int i = 0; i < pPermuteNum; ++i)
		{
			InverseNorm_SHORT(pNormShape + i * pNumPnts, pBoundRect, pNumPnts, pPermuteShapeSet + i * pNumPnts);
		} 
       
        //float initX = 0.0f;
        //float initY = 0.0f;
		/*for (int i = 0; i < pPermuteNum; ++i)
        {
            float initX = 0.0f;
            float initY = 0.0f;
            for(int j = 0; j < 25; j++)
            {
                initX += (pNormShape + i * pNumPnts)[j].x;
                initY += (pNormShape + i * pNumPnts)[j].y;       
            }
            std::cout<<"initX "<< (i + 1) <<":"<<initX<<"  ";
            std::cout<<"initY "<< (i + 1) <<":"<<initY<<"  ";
        }
        std::cout<<std::endl;*/

	}
	else
	{
		int tMinData = 0;
		int tMaxData = pNumSamples - 1;
		int tRange = tMaxData - tMinData;
		int tidx = 0;
		int tRandSeed = 0;
		float tmp = 1.0f / (RAND_MAX + 1.0f);
		memset(pImgId, 0, pPermuteNum*sizeof(int));
		srand(tRandSeed);
		while (1)
		{
			int Idx = tMinData + (int)(tRange * rand() * tmp);
			if (Idx == -1 || Idx>tMaxData || Idx<tMinData)
				continue;

			pImgId[tidx++] = Idx;
			if (tidx >= pPermuteNum)
				break;
		}

		for (int i = 0; i < pPermuteNum; ++i)
		{
			InverseNorm_SHORT(pNormShape + pImgId[i] * pNumPnts, pBoundRect, pNumPnts, pPermuteShapeSet + i * pNumPnts);
		}
	}
};
