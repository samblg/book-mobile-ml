#ifndef MODEL_H
#define MODEL_H

#include "landmarker/zhu/featpyramid.h"
#include "landmarker/zhu/config.h"

#include <vector>
#include <fstream>

class _nmatrix
{
public:
	sType *value;
	int size[2];
	_nmatrix(){value=NULL;}
};

class _partBias
{
	// partBias_x_min = [1-binSizePart*2 -binSizePart 1-binSizePart*2 -binSizePart];
	// partBias_x_max = partBias_x_min + binSizePart*3 - 1;
	// partBias_y_min = [1-binSizePart*2 1-binSizePart*2 -binSizePart -binSizePart];
	// partBias_y_max = partBias_y_min + binSizePart*3 - 1;
public:
	std::vector<int> partBias_x_min;
	std::vector<int> partBias_y_min;
	std::vector<int> partBias_x_max;
	std::vector<int> partBias_y_max;
};

class _matrixlist
{
public:
	vector<_nmatrix> Mats;
	int MatNum;
	void init(ifstream& filein, int _MatNum, int height, int width)
	{
		if (sizeof(sType) == sizeof(float))
		{
			MatNum = _MatNum;
			Mats.resize(MatNum);
			for (int i=0;i<MatNum;i++)
			{
				Mats[i].size[0] = height;
				Mats[i].size[1] = width;
				if (Mats[i].value == NULL)
				{
					Mats[i].value =new sType[height*width];
					filein.read((char*)Mats[i].value,sizeof(sType)*height*width);
				}
			}
		}
		else
		{
			MatNum = _MatNum;
			Mats.resize(MatNum);
			for (int i=0;i<MatNum;i++)
			{
				Mats[i].size[0] = height;
				Mats[i].size[1] = width;
				if (Mats[i].value == NULL)
				{
					Mats[i].value =new sType[height*width];
					double * tempData = new double[height*width];
					filein.read((char*)tempData,sizeof(double)*height*width);
					for (int j=0;j<height*width;j++)
					{
						Mats[i].value[j] = static_cast<sType>(tempData[j]);
					}
					delete []tempData;
				}
			}
		}
	}
	~_matrixlist()
	{
		for(int i=0;i<MatNum;i++)
		{
			if (Mats[i].value!= NULL)
			{
				delete [] Mats[i].value;
			    Mats[i].value = NULL;
			}
		}
	}
};

struct shapeXY
{
	vector<sType> shapeX;
	vector<sType> shapeY;
};

#ifndef WIN32
struct RECT
{
	int left;
	int right;
	int top;
	int bottom;
};
#endif // WIN32

class Model 
{
public:
	int sbin;
	_partBias partBias;

	// Feature
	int biasNum;
    int binSizeRoot;
    int binSizePart;
    int featDimP;

	// Point Model
	int pointNum1;
	int iterNum1;
	int featDim1;

	int pointNum2;
	int iterNum2;
	int featDim2;
	
	// Bbox Regression
	int bboxAnchorNum;
	sType bboxExtendScale;
	int featDimBox;

	// Setting
	sType meanDE;
	
	Model() {};

	Model(const string& filename) 
	{
		initmodel(filename); 
	};

	~Model();

	
	int dx;
	int dy;
	int* best_o_t;
	sType* v_t;

	_feat tfeat;

	unsigned char HOGGradient[511*511];

	shapeXY meanShape1;
	_matrixlist regressionmatrix1;
	_matrix featMat1;

	shapeXY meanShape2;
	_matrixlist regressionmatrix2;
	_matrix featMat2;

	sType bboxAdapt[4*5];
	_matrixlist regressionmatrixBox;
	_matrix featMatBox;

	vector<int> pts68to25[25];

	float *score_w;

	void initmodel(const string&);
	void freemodel();

    _vxvy_r vxvy_r;
	_vxvy_p vxvy_p;
};

#endif
