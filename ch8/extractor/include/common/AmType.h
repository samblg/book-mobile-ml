#pragma once

#define MAX_LANDMARKS_NUM 27


typedef struct {
    double x;
    double y;
} DLPoint;

typedef enum {
    MouthStateOpen = 0,
    MouthStateClosed
} MouthState;

typedef enum {
    EyesStateOpen = 0,
    EyesStateClosed
} EyesState;

typedef struct {
    double x;
    double y;
    double width;
    double height;
} DLRectangle;

typedef struct {
    int width;
    int height;
    unsigned char* data;
} DLImage;

struct TPointF {
    float x;
    float y;
};

typedef struct {
    int id;
    float x;
    float y;
} TLandmark1;

typedef struct {
    int count;
    int view;		// 0: (-90, -30), 1: (-30, 0), 2: (0, 30), 3: (30, 90)
    TLandmark1 pts[MAX_LANDMARKS_NUM];
} TLandmarks1;

typedef void* AmDetectorHandle;
typedef void* AmDirectionHandle;
typedef void* AmWaterRemoverHandle;
typedef void* AmLandmarkerHandle;
typedef void* AmExtractorHandle;
typedef void* AmMatcherHandle;
typedef void* AmSlicerHandle;
typedef void* AmPropHandle;
typedef void* AmQualityHandle;

#define AM_VERIFIER_FEATURE_LENGTH 256 * 1 * 1 * 4
#define AM_VERIFIER_JFEATURE_LENGTH 502 * 4
#define AM_SDK_VERSION 3.2.0.0


#define LM_IMAGE_SIDE_LENGTH (300)
#define LM_IMAGE_WIDTH (LM_IMAGE_SIDE_LENGTH)
#define LM_IMAGE_HEIGHT (LM_IMAGE_SIDE_LENGTH)
#define LM_IMAGE_SIZE (LM_IMAGE_WIDTH * LM_IMAGE_HEIGHT * 3)
#define LM_POINT_COUNT 25

#define MAX_LANDMARKS_NUM 27
#define LANDMARK_CROP_STD_WIDTH 300
#define LANDMARK_CROP_STD_HEIGHT 300

typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} AmImage;

typedef struct face_RECT{
    int index;
    float left;
    float top;
    float right;
    float bottom;
    float score;
} AmFaceRect;

//
//typedef struct face_RECT_group{
//    int index;
//    float left;
//    float top;
//    float right;
//    float bottom;
//    float score;
//} AmFaceRect_group;

typedef struct {
    float up;
    float right;
    float down;
    float left;
} AmFaceDirection;

typedef struct tagAmDetectorParam
{
    int minObjSize;
    float scaleFactor;
} AmDetectorParam;

typedef struct tagAmLandMarkResult
{
    int index;
    float left;
    float right;
    float top;
    float bottom;
    float score;
    float roll;
    float yaw;
    float pitch;
    float ptScore;
    float landmarks[LM_POINT_COUNT * 2];
    int landmark_count;
} AmLandmarkResult;

typedef struct tagAmPropResult
{
    float age; 
    int gender; //0 -> MALE, 1 -> FEMALE
    int hasGlasses;	//0 -> NoGlasses, 1 -> HasGlasses
	int hasSunGlasses; //0 -> NoSunGlasses, 1 -> HasSunGlasses
	int hasMask; //0 -> NoMask, 1 -> HasMask
}AmPropResult;

typedef int AmQualityDetectTask;
#define AM_QUALITY_DETECT_GLASSES 0x01
#define AM_QUALITY_DETECT_SUN_GLASSES 0x02
#define AM_QUALITY_DETECT_MASK 0x04
#define AM_QUALITY_DETECT_BLUR 0x08

typedef struct {
    int hasGlasses;
    int hasSunGlasses;
	int hasMask;
    float blurScore;
    //bool isQualified;
} AmQualityResult;

typedef void* AmSlicerResult;

typedef struct tagAmLandmarkerParam
{
    int initialNumber;
} AmLandmarkerParam;

typedef struct tagAmExtractorParam
{

} AmExtractorParam;

typedef struct {
	int index;
	AmFaceRect faceRect;
	float landmarks[LM_POINT_COUNT * 2];
    float roll;                                 /*< the Euler angle around Z-axis */
    float yaw;                                  /*< the Euler angle around Y-axis */
    // !!! It is different from the pitch of image.
    float pitch;                                /*< the Euler angle around X-axis */
    float confidence;                           /*< confidence of face detector */
    unsigned char face[LM_IMAGE_SIZE];          /*< normalized face image, BGR format */
    float points[LM_POINT_COUNT * 2];
} AmCroppedResult;

typedef struct {
    int width;
    int height;
    int channels;
    int pitch;
    unsigned char* data;
} AmWaterRemovalResult;

typedef struct{



}AmPropParam;

typedef struct {
    int FrmNumBFullDetect;  //a full image detection per FrmNumBFullDetect frames

    int BoostThr;          //max number of boost state
    int OCCThr;            //max number of occluded state
    float OverlapRatio;    //overlap area ratio in tracking judge

    float VarThr;
    int ft_frameNum;
    int ft_frameTotalNum;
    int ft_IsStartTracking;

    float kcf_peak_thr;

    int Marginwidth;

    int zoom;
    int zoomWidthSize;
} AmTrackerParams;

#define RETURN_TOO_MANY_FRAMES 2

#ifdef BYTE
#undef BYTE
#endif

#ifdef BOOL
#undef BOOL
#endif

#ifdef TRUE
#undef TRUE
#endif

#ifdef FALSE
#undef FALSE
#endif

typedef unsigned char BYTE;

#ifdef BOOL
#undef BOOL
typedef bool BOOL;

const BOOL TRUE = true;
const BOOL FALSE = false;
#endif

typedef enum {
    // Internal error
    AMCORE_INTERNAL_ERROR = 0x0001,
    // Instruction error
    AMCORE_ILLEGAL_INSTRUCTION = 0x0002,
    // Invalid model
    AMCORE_INVALID_MODEL = 0x0003,
    // Load library error
    AMCORE_LOAD_LIBRARY_ERROR = 0x0004,
    // Can't find license
    AMCORE_LICENSE_NOT_FOUND = 0x0101,
    // Can't create license
    AMCORE_LICENSE_CREATE_ERROR = 0x0102,
    // Can't open license
    AMCORE_LICENSE_OPEN_ERROR = 0x0103,
    // Can't read license
    AMCORE_LICENSE_READ_ERROR = 0x0104,
    // Can't write license
    AMCORE_LICENSE_WRITE_ERROR = 0x0105,
    // The license is bad
    AMCORE_LICENSE_BAD = 0x0106,
    // The license is expired
    AMCORE_LICENSE_EXPIRED = 0x0107,
    // The license time is abnormal
    AMCORE_LICENSE_TIME_ABNORMAL = 0x0108,
    // Invaild Machine
    AMCORE_LICENSE_INVALID_MACHINE = 0x0109,
    // Can't detect camera in license verifier
    AMCORE_LICENSE_CAMERA_DETECTED_ERROR = 0x010a
} AmCoreErrorCode;
