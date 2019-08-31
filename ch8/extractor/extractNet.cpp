#include "extractNet.h"
#include "cropImage.h"
#include "Exception.h"

#include "resunpack/resunpack.hpp"

#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace learnml {

ExtractNet::ExtractNet(const char* modelPath)
{
    using learnml::resunpack::PackageManager;
    using learnml::resunpack::Package;
    using learnml::resunpack::ResunpackException;

    try {
        PackageManager& packageManager = PackageManager::GetInstance();

        std::string modelDirPath = modelPath;
        Package& extractPackage = packageManager.openPackage(modelDirPath + "/Extractor.pkg");
        std::istream& extractNetParamsFile = dynamic_cast<std::istream&>(extractPackage.openResource("Extractor.parambin"));
        std::istream& extractModelFile = dynamic_cast<std::istream&>(extractPackage.openResource("Extractor.bin"));

        extractNet = new Net();
        extractNet->loadBinaryParams(extractNetParamsFile);
        extractNet->loadModel(extractModelFile);

        extractPackage.closeResource(extractNetParamsFile);
        extractPackage.closeResource(extractModelFile);
        packageManager.closePackage(extractPackage);

        CropImageUtil::GetInstance();
    }
    catch ( const ResunpackException& ) {
        throw Exception(AMCORE_INVALID_MODEL, "Open model of extractor failed");
    }
}

int ExtractNet::ExtractFeature(unsigned char* imageData, int width, int height, int pitch, int bpp,
    MlLandmarkResult* faceLandMarkResults, unsigned char* feature)
{
    minicv::Mat image;
    if ( bpp == 8 )
    {
        image = minicv::Mat(height, width, CV_8UC1, imageData);
        minicv::cvtColor(image, image, CV_GRAY2BGR);
    }
    else if ( bpp == 24 )
    {
        image = minicv::Mat(height, width, CV_8UC3, imageData);
    }
    else
    {
        return -1;
    }

    int isMirror = 0;
    TLandmarks1 lms;
    lms.count = 25;

    for ( int k = 0; k < lms.count; k++ )
    {
        lms.pts[k].id = landmark_ids[k];
        lms.pts[k].x = faceLandMarkResults->landmarks[k * 2];
        lms.pts[k].y = faceLandMarkResults->landmarks[k * 2 + 1];
    }

    vector<minicv::Mat> patches = CropImageUtil::PreparePatches(image, &lms, isMirror);
    if ( patches.size() == 0 )
        return -2;

    minicv::Mat bgr = patches[0];

    Mat in =
        Mat::from_pixels(bgr.data, Mat::PIXEL_BGR, bgr.cols, bgr.rows);

    int total = in.total();
    for ( int i = 0; i < total; i++ ) {
        in[i] = (in[i] - 127.5f) * 0.0078125f;
    }

    Extractor extractEngine = extractNet->create_extractor();
    extractEngine.set_num_threads(THREAD_COUNT);

    Mat out;
	extractEngine.input("Data1", in);
	extractEngine.extract("L2", out);

    memcpy(feature, out.data, sizeof(float)* 256);
    
    return 0;
}

}
