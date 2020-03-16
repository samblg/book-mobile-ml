 #pragma once

#include "common/AmType.h"
#include "common/ObjectFactory.h"
#include <functional>

class DirectionDetector {
public:
    static DirectionDetector* CreateObject(const std::string& key, const std::string modelPath, int deviceId);
    virtual ~DirectionDetector() {}

	virtual void setDeviceId(int deviceId) = 0;
	virtual int getDevicerId() const = 0;
    virtual void detectDirection(const unsigned char* imageData, int imageWidth, int imageHeight,
            int pitch, AmFaceDirection* direction) = 0;
};

typedef std::function<DirectionDetector*(const std::string& modelPath, int deviceId)> DirectionDetectorCreator;

#define REGISTER_DIRECTION_DETECTOR(key, value) \
    REGISTER_OBJECT_CREATOR(DirectionDetector, DirectionDetectorCreator, key, value)
