#pragma once

#ifdef WIN32
#ifdef ITRACKER_EXPORTS
#define ITRACKER_API __declspec(dllexport)
#else
#define ITRACKER_API __declspec(dllimport)
#endif
#else
#define ITRACKER_API
#endif

#include "common/AmType.h"
#include "api/authen/Types.h"

#include <vector>
#include <functional>

class FaceTracker;

namespace authen {
namespace core {
namespace api {

class IFaceDet;

class TrackerInfo {
public:
    TrackerInfo() :
        faceID(0) {
    }

    TrackerInfo(int faceID, const Rect& rect) {
        this->faceID = faceID;
        this->faceRect = rect;
    }

    int faceID;
    Rect faceRect;
};

class TrackerResult {
public:
    TrackerResult() {}

    TrackerResult(const Image& image) {
        this->frame = image;
    }

    TrackerResult(const Image& image, const std::vector<TrackerInfo>& faceList) {
        this->frame = image;
        this->faceList = faceList;
    }

    void addTrackerInfo(const TrackerInfo& trackerInfo) {
        faceList.push_back(trackerInfo);
    }

    Image frame;
    std::vector<TrackerInfo> faceList;
};

//typedef std::function<void(TrackerResult)> TrackerCallback;
typedef void(*TrackerCallback)(TrackerResult);

class ITRACKER_API ITracker{
public:
    ITracker(
            authen::core::api::IFaceDet* faceDetImpl,
            int maxFrames,
            float faceDetectThr, int maxFaceNumDetect, int width, int height);
    ITracker(const ITracker&) = delete;
    ~ITracker();

    void setParams(const AmTrackerParams* params);
    int track(
            const unsigned char* image, int width, int height, int maxFaceNum,
            TrackerCallback callback, int milliseconds = 0);
	void stop(bool wait);
    void join();

private:
    FaceTracker* _tracker;
};

}
}
}
