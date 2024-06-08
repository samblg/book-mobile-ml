#pragma once

#include "landmarker/lei/FaceAlignment.h"
#include "common/AmType.h"
#include "landmarker/Landmarker.h"

class LeiLandmarker : public Landmarker {
public:
	LeiLandmarker(const std::string& dataDirPath);
	~LeiLandmarker();
	virtual void setParam(AmLandmarkerParam* param) override;
	virtual void getLandmark(unsigned char* bgr, int width, int height,
		int pitch, AmFaceRect* rect, AmLandmarkResult* result) override;
	void landmark(uchar* imageData, int imageWidth, int imageHeight, AmFaceRect* rect, AmLandmarkResult* result, float &yaw, float &pitch, float &roll);
	
	void getGrayImage(DLImage* image);

private:
	static const char* DETECTOR_MODEL_DIR;
	static const char* LANDMARK_MODEL_FILE;

	void initPaths();
	void initImages(unsigned char* imageData, int imageWidth, int imageHeight);
	int detectFaces(AmFaceRect* results);
	BoundingBox getFaceBoundingBox(const AmFaceRect* detectedFaceRect);
	void scaleFaceBoudingBox(BoundingBox& faceBoundingBox);
	void predict(const BoundingBox& faceBoundingBox, DLPoint* landmarkPoints);
	
	ShapeRegressor* getRegressor();
	void initRegressor();
	void releaseRegressor();
	
private:
    minicv::Mat _colorImage;
    minicv::Mat _grayImage;
	int _initNumber;
	ShapeRegressor* _regressor;
	std::string _dataDirPath;
	std::string _landmarkModelPath;
};

