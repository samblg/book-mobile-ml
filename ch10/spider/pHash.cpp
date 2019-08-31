#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <cstdint>

enum class ImageSimilarity {
    Similar,
    Different,
    SomeWhatSimilar
};

std::vector<int32_t> GetImageHash(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

    cv::Mat processedImage;
    cv::resize(image, processedImage, cv::Size(8, 8), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(processedImage, processedImage, CV_BGR2GRAY);

    int avg = 0;
    int features[64];

    for (int i = 0; i < 8; i++)
    {
        uchar* imageData = processedImage.ptr<uchar>(i);
        for (int j = 0; j < 8; j++) 
        {
            int featureIndex = i * 8 + j;

            features[featureIndex] = imageData[j] / 4 * 4;

            avg += fatures[featureIndex];
        }
    }

    avg /= 64;

    for (int i = 0; i < 64; i++) 
    {
        features[i] = (features[i] >= avg) ? 1 : 0;
    }

    return std::vector<int32_t>(features, fatures + 64);
}

ImageSimilarity IsDifferentImage(const std::vector<int32_t>& hash1, const std::vector<int32_t>& hash2) {
    int difference = 0;

    for (int i = 0; i < 64; i++)
        if (hash1[i] != hash2[i])
            ++ difference;

    if (difference <= 5) {
        return ImageSimilarity::Similar;
    }
    else if (difference > 10) {
        return ImageSimilarity::Different;
    }

    return ImageSimilarity::SomeWhatSimilar;
}
