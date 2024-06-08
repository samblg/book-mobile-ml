#include "config.h"
#include "AmMat.h"
#include <iostream>

class _feat
{
public:
	int size[3];
	sType *ft;
	_feat(){ft=NULL;
    }
	~_feat()
	{
		if (ft)
		{
			delete [] ft;
			ft=NULL;
		}
	}
};

class _valindex
{
public:
	sType value;
	int index;
	sType area;
	sType x1;
	sType x2;
	sType y1;
	sType y2;
	_valindex(){;}
	~_valindex(){;}
};

class _matrix
{
public:
	sType *value;
	int size[2];
	_matrix(){value=NULL;}
	~_matrix()
	{
		if (value)
		{
			delete []value;
			value=NULL;
		}
	}
};

class _pyra
{
public:
	friend class Model;
	_feat* feat;
	sType* scales;
	int imsize[2];
	int padx;
	int pady;
	int scalenumber;
	~_pyra()
	{
		if (feat)
		{
			delete[] feat;
			delete[] scales;
		}
		feat=NULL;
		scales=NULL;
	}
};
class _totalpyrafeat
{
public:
	int numlevels;
	int *featdimsprod;
	int numfeatures;
	int pcadim;
	_pyra pyra;
	_pyra pyrapro;
	~_totalpyrafeat()
	{
		if (featdimsprod)
		{
			delete[] featdimsprod;
			pyra.~_pyra();
			pyrapro.~_pyra();
		}
	}
	void setdimsprod()
	{
		numlevels=pyra.scalenumber;
		numfeatures=pyra.feat[0].size[2];
		pcadim=pyrapro.feat[0].size[2];
		featdimsprod=new int[numlevels];
		for (int i=0;i<numlevels;i++)
		{
			featdimsprod[i]=pyra.feat[i].size[0]*pyra.feat[i].size[1];
		}
	}
};


class _vxvy_r
{
public:
	sType vx0[_binSizeRoot][_binSizeRoot];
	sType vy0[_binSizeRoot][_binSizeRoot];
	sType vx1[_binSizeRoot][_binSizeRoot];
	sType vy1[_binSizeRoot][_binSizeRoot];
	int ixp[_binSizeRoot];
	int iyp[_binSizeRoot];
	sType vx0vy0[_binSizeRoot][_binSizeRoot];
	sType vx0vy1[_binSizeRoot][_binSizeRoot];
	sType vx1vy0[_binSizeRoot][_binSizeRoot];
	sType vx1vy1[_binSizeRoot][_binSizeRoot];
	int patchwidth;
};

class _vxvy_p
{
public:
	sType vx0[_binSizePart][_binSizePart];
	sType vy0[_binSizePart][_binSizePart];
	sType vx1[_binSizePart][_binSizePart];
	sType vy1[_binSizePart][_binSizePart];
	int ixp[_binSizePart];
	int iyp[_binSizePart];
	sType vx0vy0[_binSizeRoot][_binSizeRoot];
	sType vx0vy1[_binSizeRoot][_binSizeRoot];
	sType vx1vy0[_binSizeRoot][_binSizeRoot];
	sType vx1vy1[_binSizeRoot][_binSizeRoot];
	int patchwidth;
};

void features(const AmMat& img,_feat& feat,int bins);
void features_lookup(const AmMat&, _feat&, int, unsigned char*);
void features_lookup(_feat& feat,int bins, int* best_o_t, sType* v_t, int x_min,int y_min,int buf);
void features_lookup(_feat& feat,int bins, int* best_o_t, sType* v_t, int x_min,int y_min,int buf,_vxvy_p &vxvy);
void features_lookup(_feat& feat,int bins, int* best_o_t, sType* v_t, int x_min,int y_min,int buf,_vxvy_r &vxvy);
