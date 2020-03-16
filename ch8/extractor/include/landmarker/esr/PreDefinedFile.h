#pragma once

#define RANDOM_INITIAL   0//0-用所有NUM_INITIALSHAPE个形状初始化；1-随机选择NUM_INITIALSHAPE个形状初始化
#define NUM_INITIALSHAPE 11//初始化NUM_INITIALSHAPE次

typedef union _SFTransStruct
{
	unsigned int intdata;
	float		 floatdata;
}SFTransStruct;

typedef struct _POINT_FLOAT
{
	float x;
	float y;
}POINT_FLOAT;

typedef struct _POINT_SHORT
{
	unsigned short x;
	unsigned short y;
}POINT_SHORT;

typedef struct _AUTHEN_RECT
{
	float left;
	float top;
	float width;
	float height;
}AUTHEN_RECT;

typedef struct _FEAPOINT_LOCATION
{
	unsigned short LMK_first;
	unsigned short LMK_second;
	float percent;
}FEAPOINT_LOCATION;

typedef struct _WEAKREG
{
	int     Depth;             
	int		Bins;           
	unsigned short   *Thrs;           
	unsigned short   *Fids;            
	POINT_SHORT      *MeanOffset; 
}WEAKREG;

typedef struct _STRONGREG
{
	int      NumWeakReg;   
	int      NumFeaPoint; 
	FEAPOINT_LOCATION   *FeaPntLoc;
	WEAKREG             *WReg;

}STRONGREG;

typedef struct _CASCADE_REG
{
	int         NumStage;
	int         Type;    
	POINT_SHORT *InitialShape; //norm
	STRONGREG   *SReg;
}CASCADE_REG;
