#pragma once

#include "common/caffe-fast-rcnn/caffe/caffe.hpp"
#include "direction/DirectionDetector.h"

#include <string>
#include <memory>

class CaffeDirectionDetector : public DirectionDetector {
public:
    const static int DETECT_IMAGE_WIDTH;
    const static int DETECT_IMAGE_HEIGHT;

    CaffeDirectionDetector(const std::string& packagePath, int deviceId);
    ~CaffeDirectionDetector();

	virtual void detectDirection(const unsigned char* imageData, int imageWidth, int imageHeight,
            int pitch, AmFaceDirection* direction) override;

	virtual void setDeviceId(int deviceId);
	virtual int getDevicerId() const {
        return _deviceId;
    }

private:
    void readMeansFromStream(std::istream& stream);

private:
    int _deviceId;
    std::unique_ptr<caffe::Net<float>> _caffeNet;
    caffe::Blob<float> _means;
};
