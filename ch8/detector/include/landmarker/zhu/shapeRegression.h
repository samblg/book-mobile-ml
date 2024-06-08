#include "landmarker/zhu/config.h"
#include "landmarker/zhu/model.h"
#include "landmarker/zhu/AmMat.h"

class _bbox
{
public:
	sType rootcoord[4];
	sType score;
	int _componentid;
	//sType _partPos[_partnum*4];

	_bbox()
	{
		score = 0.0;
		_componentid = 1;
	}
	~_bbox()
	{}
};
void shapeRegression(Model&, AmMat&, _bbox, shapeXY&, float&, float&, float &, float &);
//void shapeRegression2(Model&, AmMat*, _bbox, shapeXY&, shapeXY&, float&, float&, float &, float &);