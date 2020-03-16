#include "detectNet.h"
#include "common/Exception.h"

#include "resunpack/resunpack.hpp"

#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#ifdef WIN32
#undef max
#undef min
#endif

const TPoint3DF CAFFE_LANDMARK_KPS[] = {
    { -31964.1012369792, -34082.0214843750, 92502.0611979167 },
    { 31131.1191406250, -34101.7893880208, 92279.8593750000 },
    { -57454.2578125000, -47999.3007812500, 79612.7968750000 },
    { -19306.5332031250, -53098.0859375000, 107505.515625000 },
    { 18457.7910156250, -53323.0195312500, 107447.601562500 },
    { 56890.7382812500, -48158.5625000000, 79479.7734375000 },
    { -146.368896484375, -34128.0859375000, 110560.226562500 },
    { 2.72319674491882, 111.864158630371, 131227.406250000 },
    { -192.024765014648, 24960.4531250000, 116207.859375000 },
    { -154.684158325195, 40559.1953125000, 112943.679687500 },
    { -25539.5566406250, 33627.8476562500, 98388.8984375000 },
    { 24523.9921875000, 33622.1679687500, 98218.9375000000 },
    { 11689.0810546875, 9230.85156250000, 108837.046875000 },
    { -12155.0888671875, 9290.84765625000, 108878.031250000 },
    { -262.772941589355, 31417.9765625000, 111709.699218750 },
    { -174.798309326172, 11534.9863281250, 115540.867187500 },
    { -19522.4179687500, -33381.6484375000, 92335.7500000000 },
    { -31909.4716796875, -30979.1679687500, 93353.7539062500 },
    { -43420.1523437500, -33702.6445312500, 86386.4921875000 },
    { 42852.0781250000, -33576.5312500000, 86324.2031250000 },
    { 31121.0175781250, -31050.3291015625, 93217.2617187500 },
    { 18462.4433593750, -33379.6835937500, 92001.1796875000 },
    { -38618.1367187500, -56473.7773437500, 99980.5781250000 },
    { 37875.5585937500, -56829.4687500000, 99749.8515625000 },
    { -54.8117485046387, 77982.6171875000, 95650.3437500000 },
};


static Mat loadBinaryBlob(std::istream& blobFile) {
    int num = 0;
    int channels = 0;
    int width = 0;
    int height = 0;

    blobFile.read(reinterpret_cast<char*>(&num), sizeof(int));
    blobFile.read(reinterpret_cast<char*>(&channels), sizeof(int));
    blobFile.read(reinterpret_cast<char*>(&width), sizeof(int));
    blobFile.read(reinterpret_cast<char*>(&height), sizeof(int));

    Mat blob(width, height, channels);
    float* blobData = blob.floatData();
    blobFile.read(reinterpret_cast<char*>(blobData), width * height * channels * sizeof(float));

    return blob;
}

const float RModel[4][4] = {
    { 0.1801f, 0.2433f, -0.1801f, -0.2433f },
    { 0.1177f, 0.1699f, -0.1177f, -0.1699f },
    { -0.1988f, -0.2434f, 0.1988f, 0.2434f },
    { -0.1899f, -0.3874f, 0.1899f, 0.3874f }
};

static void bboxRegression(float* bbox_dt_init, float* bbox_out)
{
    float center_x = (bbox_dt_init[2] + bbox_dt_init[0]) / 2;
    float center_y = (bbox_dt_init[3] + bbox_dt_init[1]) / 2;
    float llength = ((bbox_dt_init[2] - bbox_dt_init[0]) + (bbox_dt_init[3] - bbox_dt_init[1])) / 4;
    float bboxes_norm[4];
    bboxes_norm[0] = (bbox_dt_init[0] - center_x) / llength;
    bboxes_norm[1] = (bbox_dt_init[1] - center_y) / llength;
    bboxes_norm[2] = (bbox_dt_init[2] - center_x) / llength;
    bboxes_norm[3] = (bbox_dt_init[3] - center_y) / llength;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            bbox_out[i] += RModel[i][j] * bboxes_norm[j];
        }
        bbox_out[i] *= llength;
    }

    bbox_out[0] += center_x;
    bbox_out[1] += center_y;
    bbox_out[2] += center_x;
    bbox_out[3] += center_y;
}

static void bboxAdapt(float* bbox_dt_init, float* bbox_out, int width, int height)
{
    float* regression_box = new float[4];
    memset(regression_box, 0, sizeof(float)* 4);
    bboxRegression(bbox_dt_init, regression_box);

    //std::cout << regression_box[0] << "," << regression_box[1] << "," << regression_box[2] << "," << regression_box[3] << std::endl;

    float extend_scale = 1.2f;
    float llength = ((regression_box[2] - regression_box[0]) + (regression_box[3] - regression_box[1])) / 2;
    llength = round(llength * extend_scale);
    float center_x = (regression_box[2] + regression_box[0]) / 2;
    float center_y = (regression_box[3] + regression_box[1]) / 2;

    bbox_out[0] = round(center_x - llength / 2);
    bbox_out[1] = round(center_y - llength / 2);
    bbox_out[2] = bbox_out[0] + llength;
    bbox_out[3] = bbox_out[1] + llength;

    if (bbox_out[0] < 0)
        bbox_out[0] = 0;
    if (bbox_out[2] > width)
        bbox_out[2] = width;
    if (bbox_out[1] < 0)
        bbox_out[1] = 0;
    if (bbox_out[3] > height)
        bbox_out[3] = height;
    delete[] regression_box;
}

namespace learnml {

DetectNet::DetectNet() :
		_scale(0.0f)
{

}

DetectNet::DetectNet(const char* modelPath) : 
		_scale(0.0f)
{
    using learnml::resunpack::PackageManager;
    using learnml::resunpack::Package;
    using learnml::resunpack::ResunpackException;

    //load detect Net
    try {
        PackageManager& packageManager = PackageManager::GetInstance();

        std::string modelDirPath = modelPath;
        Package& detectPackage = packageManager.openPackage(modelDirPath + "/Detector.pkg");
        std::istream& detectNetParamsFile = dynamic_cast<std::istream&>(detectPackage.openResource("Detector.parambin"));
        std::istream& detectModelFile = dynamic_cast<std::istream&>(detectPackage.openResource("Detector.bin"));

        detectNet = new Net();
        detectNet->loadBinaryParams(detectNetParamsFile);
        detectNet->loadModel(detectModelFile);

        //load landmark Net
        Package& landmarkPackage = packageManager.openPackage(modelDirPath + "/Landmarker.pkg");
        std::istream& landmarkMeanFile = dynamic_cast<std::istream&>(landmarkPackage.openResource("Landmarker.mean"));
        std::istream& landmarkParamsFile = dynamic_cast<std::istream&>(landmarkPackage.openResource("Landmarker.parambin"));
        std::istream& landmarkModelFile = dynamic_cast<std::istream&>(landmarkPackage.openResource("Landmarker.bin"));

        landmarkNet = new Net();
        landmark_mean = loadBinaryBlob(landmarkMeanFile);
        landmarkNet->loadBinaryParams(landmarkParamsFile);
        landmarkNet->loadModel(landmarkModelFile);

        detectPackage.closeResource(detectNetParamsFile);
        detectPackage.closeResource(detectModelFile);
        detectPackage.closeResource(landmarkMeanFile);
        detectPackage.closeResource(landmarkParamsFile);
        detectPackage.closeResource(landmarkModelFile);

        packageManager.closePackage(detectPackage);
        packageManager.closePackage(landmarkPackage);
    }
    catch ( const ResunpackException& ) {
        throw Exception(AMCORE_INVALID_MODEL, "Open model of detector failed");
    }
}

int DetectNet::SetParam(MlDetectorParam* param) {
	if (!param) {
		return -1;
	}

	_scale = param->scaleFactor;

	return 0;
}

float bgrMeans[] = { 104.f, 117.f, 123.f };
float grayMeans[] = { 127.5f };

int DetectNet::DetectObjectRects(unsigned char* imageData, int width, int height, int pitch, int bpp, float threshold,
    int maxCount, MlObjectRect* objectRects)
{
    Extractor detectEngine = detectNet->create_extractor();
    
    float scale = 1.0f;
    if(_scale == 0.0f)
    {
        float scaleX = width / 640.0f;
        float scaleY = height / 640.0f;
        
        if ( scaleX > 1.0f || scaleY > 1.0f )
            scale = std::max(scaleX, scaleY);
        
        scaleX = width / 100.0f;
        scaleY = height / 100.0f;
        if ( scaleX < 1.0f || scaleY < 1.0f )
            scale = 0.5;
    }
    else
        scale = _scale;

    int newWidth = ((int)((width / scale) / 4)) * 4;
    int newHeight = height / scale;
    
    Mat in;
    if ( bpp == 8 )
    {
        in = Mat::from_pixels_resize(imageData, Mat::PIXEL_GRAY, width, height, newWidth, newHeight);
    }
    else if ( bpp == 24 )
    {
        in = Mat::from_pixels_resize(imageData, Mat::PIXEL_BGR2GRAY, width, height, newWidth, newHeight);
    }

    in.substract_mean_normalize(grayMeans, 0);

    detectEngine.set_num_threads(THREAD_COUNT);
    Mat out;
    detectEngine.input("data", in);
    detectEngine.extract("detection_out", out);

    int objectCount = out.h;
    int feaSize = out.w;
    float* result_ptr = out.floatData();

    std::vector<MlObjectRect> tempRects;
    for ( int i = 0; i < objectCount; i++ )
    {
        MlObjectRect tempRect;
        tempRect.score = result_ptr[1];
        tempRect.left = result_ptr[2] * width;
        tempRect.top = result_ptr[3] * height;
        tempRect.right = result_ptr[4] * width;
        tempRect.bottom = result_ptr[5] * height;
        result_ptr += feaSize;
        if ( tempRect.score > threshold )
        {
            tempRects.push_back(tempRect);
        }
    }

    std::sort(tempRects.begin(), tempRects.end(),
        [](const MlObjectRect& rect1, const MlObjectRect& rect2) -> bool {
        float square1 = (rect1.right - rect1.left) * (rect1.bottom - rect1.top);
        float square2 = (rect2.right - rect2.left) * (rect2.bottom - rect2.top);

        return square1 > square2;
    });

    objectCount = tempRects.size();
    if ( objectCount > maxCount )
        objectCount = maxCount;

    for ( int i = 0; i < objectCount; i++ )
    {
        objectRects[i] = tempRects[i];
    }
    return objectCount;
}

int DetectNet::DetectObjectByRect(unsigned char* imageData, int width, int height, int pitch, int bpp, MlObjectRect* objectRect, MlLandmarkResult* objectLandMarkResult)
{
    Extractor landmarkEngine = landmarkNet->create_extractor();

    int channels = 3;

    float bbox_dt_init[4];
    bbox_dt_init[0] = objectRect->left;
    bbox_dt_init[1] = objectRect->top;
    bbox_dt_init[2] = objectRect->right;
    bbox_dt_init[3] = objectRect->bottom;

    float bbox_out[4];
    memset(bbox_out, 0, sizeof(float) * 4);
    bboxAdapt(bbox_dt_init, bbox_out, width, height);

    int left = bbox_out[0];
    int top = bbox_out[1];
    int right = bbox_out[2];
    int bottom = bbox_out[3];

    int crop_width = bbox_out[2] - bbox_out[0];
    int crop_height = bbox_out[3] - bbox_out[1];

    unsigned char* cropPixels = new unsigned char[crop_width * crop_height * 3];
    for ( int i = top; i < bottom; i++ )
    {
        for ( int j = left; j < right; j++ )
        {
            for ( int c = 0; c < channels; c++ )
                cropPixels[((i - top) * crop_width + j - left) * channels + c] = imageData[(i * width + j) * channels + c];
        }
    }
    int new_width = 120;
    int new_height = 120;

    // chw
    Mat in =
        Mat::from_pixels_resize(cropPixels, Mat::PIXEL_BGR, crop_width, crop_height, new_width, new_height);
    for ( int index = 0; index < new_width * new_height * channels; index++ )
    {
        in[index] = in[index] - landmark_mean[index];
    }
    delete[] cropPixels;

    //ex.set_light_mode(true);

    Mat fc2;
    landmarkEngine.set_num_threads(THREAD_COUNT);
    landmarkEngine.input("data", in);
    landmarkEngine.extract("fc2", fc2);

    Mat fcPara;
    landmarkEngine.set_num_threads(THREAD_COUNT);
    landmarkEngine.input("fc2", fc2);
    landmarkEngine.extract("fc_para", fcPara);

    float* result_ptr = fcPara.floatData();
    objectLandMarkResult->left = objectRect->left;
    objectLandMarkResult->top = objectRect->top;
    objectLandMarkResult->right = objectRect->right;
    objectLandMarkResult->bottom = objectRect->bottom;
    objectLandMarkResult->score = objectRect->score;
    for ( int i = 0; i < 25; i++ )
    {
        objectLandMarkResult->landmarks[i * 2] = result_ptr[i] * crop_width + bbox_out[0];
        objectLandMarkResult->landmarks[i * 2 + 1] = result_ptr[i + 25] * crop_height + bbox_out[1];
    }

    int pointNum = 25;
    egp_NodePtrArr *nodes = new egp_NodePtrArr[pointNum];
    for ( int i = 0; i < pointNum; i++ )
    {
        nodes[i].x = objectLandMarkResult->landmarks[i * 2];
        nodes[i].y = objectLandMarkResult->landmarks[i * 2 + 1];
    }

    PoseEstimation(nodes, pointNum, CAFFE_LANDMARK_KPS, &objectLandMarkResult->roll, &objectLandMarkResult->yaw, &objectLandMarkResult->pitch);
    delete[] nodes;

    Mat fcq2;
    landmarkEngine.set_num_threads(THREAD_COUNT);
    landmarkEngine.input("fc2", fc2);
    landmarkEngine.extract("fcq2", fcq2);
    objectLandMarkResult->ptScore = fcq2[0];

    return 0;
}

int DetectNet::DetectObjects(unsigned char* imageData, int width, int height, int pitch, int bpp,
    float threshold, int maxCount, MlLandmarkResult* objectLandMarkResults)
{
    MlObjectRect* objectRects = new MlObjectRect[maxCount];
    int objectCount = DetectObjectRects(imageData, width, height, pitch, bpp, threshold, maxCount, objectRects);

    for ( int i = 0; i < objectCount; i++ )
    {
        DetectObjectByRect(imageData, width, height, pitch, bpp, &objectRects[i], &objectLandMarkResults[i]);
    }

    delete [] objectRects;

    return objectCount;
}

}
