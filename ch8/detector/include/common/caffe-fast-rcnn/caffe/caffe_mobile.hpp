#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include "caffe/caffe.hpp"
#include "caffe/detect_util.h"

#include <string>
#include <vector>

using std::string;

typedef struct feaData
{
	float fea[256];
	string name;
}feaData;

class KTimer;

namespace caffe {

class CaffeMobile
{
public:
    CaffeMobile(std::string packagePath, string modelPath, string weightsPath);
	~CaffeMobile();

    float test(string img_path);
    void setDeviceId(int deviceId);
    int deviceId() const;

	std::vector<caffe::bbox_fu> feaExtractionMat(float* imageData, int imageWidth, int imageHeight, float *fea, float minObjectSize, KTimer* timer = nullptr);
	
	float ComputeSimCos(float *Afea, float *Bfea);
	
	int getMostSimilarityIndex(float *feadst);
	
	float getmaxCos();


private:
    Net<float> *caffe_net;
    std::vector <feaData> feaDataSet;
    int index;
    float maxCos;
    int _deviceId;
};

} // namespace caffe

#endif
