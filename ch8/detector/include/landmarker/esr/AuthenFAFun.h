#pragma once
#include "landmarker/esr/PreDefinedFile.h"
#define MY_MAX(a,b) ((a)>(b)?(a):(b)) 
#define MY_MIN(a,b) ((a)<(b)?(a):(b)) 
class ESRCLASS
{
public:
	int mRandFeaPntNum;
	int mTrainImgNum;
	int mLankmarkNum;
	int mInitialShapeNum;
	int mRefFeaNum;

	int   *mImgIdx;
	float **mALLFeaData;
	float *mRefFeaData;
	float *mPrdictShapes;

	CASCADE_REG mCascade;

	POINT_FLOAT *mInitShapeMat;
	POINT_FLOAT *mNormMeanShape;
	POINT_FLOAT *mNormInitialShapeMat;
	POINT_FLOAT *mRegShapeOffset;

public:
    bool LoadModel(const char *pModelPath);
    bool InitialModel(const char *pModelPath);
	void FreeModel();

	void CalFeatures(unsigned char *pData, int pWidth, int pHeight, AUTHEN_RECT pBoundRect, const FEAPOINT_LOCATION *pFeaPntLocationLine);//calculate mALLFeaData
	void DeltaShapeEstmate(STRONGREG *pStrongReg);
	void Prediction(unsigned char *pData, int pWidth, int pHeight, AUTHEN_RECT pRect, float  *pPredShape);
};

