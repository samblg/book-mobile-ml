#ifndef AuthenFA_API
#define AuthenFA_API

#include "common/externc.h"
#include "common/AmType.h"

typedef struct _AUTHENRECT
{
	float left;
	float top;
	float width;
	float height;
}AUTHENRECT;

typedef void *AmFaceLandmarkerHandle;
AuthenFA_API AmFaceLandmarkerHandle AmCreateFaceLandmarker(int type, const char* model_path);
AuthenFA_API int AmGetLandmark(AmFaceLandmarkerHandle handle, unsigned char* bgr, int width, int height, int pitch, AmFaceRect* rect, AmLandmarkResult* result);
AuthenFA_API void AmReleaseFaceLandmarker(AmFaceLandmarkerHandle handle);
#endif




