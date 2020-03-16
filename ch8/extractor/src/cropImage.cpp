#include "cropImage.h"
#include <iostream>

const int PATCH_COUNT = 25;
const int ADDED_PATCH_COUNT = 10;

bool isMirror = false;

const int PATCH_SIZE[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	55, 55, 47, 47, 47,
	47, 47, 47, 47, 47,
	47, 47, 47, 47, 47,
	47, 47, 47, 47, 47,
	47, 47, 47, 47, 47, 80, 55, 55, 60, 60, 112, 112, 80, 120, 120
};
const int PATCH_PAD_X[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	4, 4, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8,
	0, 0, 0
};
const int PATCH_PAD_Y[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	3, 0, 6, 0, 3,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0
};
const int PATCH_COLOR[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	1, 1, 1, 0, 1,
	0, 0, 1, 0, 0,
	1, 0, 0, 1, 0,
	0, 1, 0, 1, 0,
	1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1
};
const int PATCH_SCALE[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	12, 18, 33, 18, 27,
	33, 33, 30, 33, 30,
	12, 33, 15, 30, 15,
	24, 18, 30, 33, 21,
	33, 18, 33, 18, 33, 46, 70, 70, 70, 70, 36, 54,
	26, 39, 69
};
const int PATCH_KP[PATCH_COUNT + ADDED_PATCH_COUNT][2] = {
	{ 6, 6 }, { 6, 6 }, { 2, 5 }, { 10, 11 }, { 6, 6 },
	{ 9, 9 }, { 18, 19 }, { 13, 12 }, { 0, 1 }, { 10, 11 },
	{ 15, 15 }, { 15, 15 }, { 6, 6 }, { 13, 12 }, { 6, 6 },
	{ 15, 15 }, { 14, 14 }, { 3, 4 }, { 7, 7 }, { 6, 6 },
	{ 0, 1 }, { 22, 23 }, { 10, 11 }, { 15, 15 }, { 3, 4 }, { 6, 6 }, { 0, 0 }, { 1, 1 }, { 15, 15 }, { 8, 8 }, { 6, 6 }, { 6, 6 },
	{ 6, 6 }, { 6, 6 }, { 6, 6 }
};
const float PATCH_OFFSET[PATCH_COUNT + ADDED_PATCH_COUNT] = {
	0.0f, 0.0f, 0.0f, 0.0f, 0.3333f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.3333f, 0.0f, -1.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.3333f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3333f, -0.15f, -0.15f, -0.35f, 0.15f, 0.0f, 0.3333f,
	0.0f, 0.0f, 0.3333f
};

const float SDM_MEAN_SHAPE[25][3] = {
	{ 0, 55.1541, 54.3499 },
	{ 1, 104.824, 54.3484 },
	{ 2, 38.7477, 40.6755 },
	{ 3, 68.3927, 39.6999 },
	{ 4, 91.6609, 39.6145 },
	{ 5, 121.361, 40.6468 },
	{ 6, 79.8268, 51.0572 },
	{ 9, 79.754, 80.0781 },
	{ 10, 79.8685, 101.152 },
	{ 11, 79.8834, 116.29 },
	{ 12, 58.8725, 107.238 },
	{ 13, 101.025, 107.208 },
	{ 43, 91.4864, 88.3588 },
	{ 44, 68.2558, 88.4107 },
	{ 45, 79.8439, 107.688 },
	{ 200, 79.853, 90.0636 },
	{ 221, 65.0264, 55.2539 },
	{ 222, 55.1593, 57.2692 },
	{ 223, 44.9736, 54.7461 },
	{ 225, 115.094, 54.7284 },
	{ 226, 104.829, 57.2945 },
	{ 227, 94.9062, 55.2716 },
	{ 300, 53.4134, 33.4486 },
	{ 302, 107.088, 33.3567 },
	{ 314, 79.876, 139.761 }
};

float PATCH_MEAN_SHAPE[PATCH_COUNT + ADDED_PATCH_COUNT][MAX_LANDMARKS_NUM - 2][2] = { 0 };

const int PATCH_MAX_SIZE = 55;

#define KP1_INDEX 0
#define KP2_INDEX 1

#define KP_EYE_LEFT_INNER 16
#define KP_EYE_LEFT_OUTTER 18
#define KP_EYE_RIGHT_INNER 21
#define KP_EYE_RIGHT_OUTTER 19

#define KP3_INDEX 6
#define KP4_INDEX 15
#define KP5_INDEX 8

#define STD_WIDTH 160
#define STD_HEIGHT 160

static void EnryptLandmarks(unsigned char* data, int length)
{
	const unsigned char key = 0x7e;
	int i;

	for (i = 0; i < length; i++) {
		data[i] = data[i] ^ key;
	}
}



static int LandmarkCMP(const void *a, const void *b)
{
	const TLandmark1* item1 = (const TLandmark1*)a;
	const TLandmark1* item2 = (const TLandmark1*)b;

	if (item1->id > item2->id) {
		return 1;
	}
	else if (item1->id < item2->id) {
		return -1;
	}
	else {
		return 0;
	}
}

static void linsolve4(float* A, float* b, float* h)
{

	float det;
	float x, y, w, z;

	x = A[0]; y = A[1]; w = A[2]; z = A[3];
	det = w*z - x*x - y*y;

	h[0] = -x * b[0] - y * b[1] + w * b[2];
	h[1] = y * b[0] - x * b[1] + w * b[3];
	h[2] = z * b[0] - x * b[2] + y * b[3];
	h[3] = z * b[1] - y * b[2] - x * b[3];

	h[0] /= det;
	h[1] /= det;
	h[2] /= det;
	h[3] /= det;
}

static void sim_params_from_points(const TPointF dstKeyPoints[],
	const TPointF srcKeyPoints[], int count,
	float* a, float* b, float* tx, float* ty)
{
	int i;
	float X1, Y1, X2, Y2, Z, C1, C2;
	float A[4], c[4], h[4];

	X1 = 0.f; Y1 = 0.f;
	X2 = 0.f; Y2 = 0.f;
	Z = 0.f;
	C1 = 0.f; C2 = 0.f;
	for (i = 0; i < count; i++) {
		float x1, y1, x2, y2;

		x1 = dstKeyPoints[i].x;
		y1 = dstKeyPoints[i].y;
		x2 = srcKeyPoints[i].x;
		y2 = srcKeyPoints[i].y;

		X1 += x1;
		Y1 += y1;
		X2 += x2;
		Y2 += y2;
		Z += (x2 * x2 + y2 * y2);
		C1 += (x1 * x2 + y1 * y2);
		C2 += (y1 * x2 - x1 * y2);
	}

	A[0] = X2; A[1] = Y2; A[2] = (float)count;  A[3] = Z;
	c[0] = X1; c[1] = Y1; c[2] = C1; c[3] = C2;
	linsolve4(A, c, h);

	/* rotation, scaling, and translation parameters */
	*a = h[0];
	*b = h[1];
	*tx = h[2];
	*ty = h[3];
}

static void sim_transform_landmark(const TLandmark1* landmark, TPointF* dst,
	int count, float a, float b, float tx, float ty)
{
	int i;
	float x, y;

	// transform last shape to current shape
	for (i = 0; i < count; i++) {
		x = landmark[i].x;
		y = landmark[i].y;

		dst[i].x = a * x - b * y + tx;
		dst[i].y = a * y + b * x + ty;
	}
}

static void sim_transform_image(const unsigned char* gray, int width, int height, int pitch,
	unsigned char* dst, int width1, int height1,
	float a, float b, float tx, float ty)
{
	int i, j;
	float x, y;
	int ix, iy;
	float u, v;

	for (i = 0; i < height1; i++) {
		for (j = 0; j < width1; j++) {
			x = a * j - b * i + tx;
			y = a * i + b * j + ty;

			ix = (int)x;
			iy = (int)y;

			u = x - ix;
			v = y - iy;

			ix = MIN(width - 2, MAX(0, ix));
			iy = MIN(height - 2, MAX(0, iy));

			dst[i * width1 + j] = (unsigned char)((1.0f - u) * (1.0f - v) * gray[iy * pitch + ix] +
				u * (1.0f - v) * gray[iy * pitch + ix + 1] +
				(1.0f - u) * v * gray[(iy + 1) * pitch + ix] +
				u * v * gray[(iy + 1) * pitch + ix + 1] + 0.5f);
		}
	}
}

static void sim_transform_color_image(const unsigned char* color, int width, int height, int pitch,
	unsigned char* dst, int width1, int height1,
	float a, float b, float tx, float ty)
{
	int i, j;
	float x, y;
	int ix, iy;
	float u, v;

	for (i = 0; i < height1; i++) {
		for (j = 0; j < width1; j++) {
			x = a * j - b * i + tx;
			y = a * i + b * j + ty;

			ix = (int)x;
			iy = (int)y;

			u = x - ix;
			v = y - iy;

			ix = MIN(width - 2, MAX(0, ix));
			iy = MIN(height - 2, MAX(0, iy));

			dst[i * width1 * 3 + j * 3] = (unsigned char)((1.0f - u) * (1.0f - v) * color[iy * pitch + ix * 3] +
				u * (1.0f - v) * color[iy * pitch + (ix + 1) * 3] +
				(1.0f - u) * v * color[(iy + 1) * pitch + ix * 3] +
				u * v * color[(iy + 1) * pitch + (ix + 1) * 3] + 0.5f);
			dst[i * width1 * 3 + j * 3 + 1] = (unsigned char)((1.0f - u) * (1.0f - v) * color[iy * pitch + ix * 3 + 1] +
				u * (1.0f - v) * color[iy * pitch + (ix + 1) * 3 + 1] +
				(1.0f - u) * v * color[(iy + 1) * pitch + ix * 3 + 1] +
				u * v * color[(iy + 1) * pitch + (ix + 1) * 3 + 1] + 0.5f);
			dst[i * width1 * 3 + j * 3 + 2] = (unsigned char)((1.0f - u) * (1.0f - v) * color[iy * pitch + ix * 3 + 2] +
				u * (1.0f - v) * color[iy * pitch + (ix + 1) * 3 + 2] +
				(1.0f - u) * v * color[(iy + 1) * pitch + ix * 3 + 2] +
				u * v * color[(iy + 1) * pitch + (ix + 1) * 3 + 2] + 0.5f);
		}
	}
}

static void BGR2Gray(const unsigned char* bgr, int width, int height, int pitch, unsigned char* gray)
{
	int i, j;

	// bgr to gray
	const unsigned char* pSrc = bgr;
    unsigned char* dst = gray;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			dst[j] = (306 * pSrc[j * 3 + 2] + 601 * pSrc[j * 3 + 1] + 117 * pSrc[j * 3]) >> 10;
		}
		dst += width;
		pSrc += pitch;
	}
}

static void CropLeftPatch(const unsigned char* image, int width, int height, int pitch,
	const TPointF* lm, int k, float* mat)
{
	float* dst = mat;
    const unsigned char* src = image;

	int start = -PATCH_SIZE[k] / 2;
	int end = start + PATCH_SIZE[k];
	float offset = PATCH_OFFSET[k] * PATCH_SCALE[k];

	// y
	for (int i = start; i < end; i++) {
		// x
		for (int j = start; j < end; j++) {
			int pos = (j - start) * PATCH_SIZE[k] + (i - start);

			int ii = (int)(i + lm->y + offset + 0.5f);
			int jj = (int)(j + lm->x + 0.5f);

			ii = MIN(height - 1, MAX(0, ii));
			jj = MIN(width - 1, MAX(0, jj));

			dst[pos] = src[ii * pitch + jj];
		}
	}

	//ฒน0
	if (PATCH_PAD_X[k]>0)
	{
		for (int j = 0; j<PATCH_PAD_X[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
		for (int j = PATCH_SIZE[k] - PATCH_PAD_X[k]; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
	}
	else if (PATCH_PAD_Y[k]>0)
	{
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_PAD_Y[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = PATCH_SIZE[k] - PATCH_PAD_Y[k]; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
	}

}

static void CropRightPatch(const unsigned char* image, int width, int height, int pitch,
	const TPointF* lm, int k, float* mat)
{
	float* dst = mat;
    const unsigned char* src = image;

	int start = -PATCH_SIZE[k] / 2;
	int end = start + PATCH_SIZE[k];
	float offset = PATCH_OFFSET[k] * PATCH_SCALE[k];

	// y
	for (int i = start; i < end; i++) {
		// x
		for (int j = start; j < end; j++) {
			int pos = (j - start) * PATCH_SIZE[k] + (i - start);

			int ii = (int)(i + lm->y + offset + 0.5f);
			int jj = (int)(-j + lm->x + 0.5f);

			ii = MIN(height - 1, MAX(0, ii));
			jj = MIN(width - 1, MAX(0, jj));

			dst[pos] = src[ii * pitch + jj];
		}
	}

	//ฒน0
	if (PATCH_PAD_X[k]>0)
	{
		for (int j = 0; j<PATCH_PAD_X[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
		for (int j = PATCH_SIZE[k] - PATCH_PAD_X[k]; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
	}
	else if (PATCH_PAD_Y[k]>0)
	{
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_PAD_Y[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = PATCH_SIZE[k] - PATCH_PAD_Y[k]; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
			}
		}
	}

}

static void CropLeftColorPatch(const unsigned char* image, int width, int height, int pitch,
	const TPointF* lm, int k, float* mat)
{
	float* dst = mat;
    const unsigned char* src = image;

	int start = -PATCH_SIZE[k] / 2;
	int end = start + PATCH_SIZE[k];
	float offset = PATCH_OFFSET[k] * PATCH_SCALE[k];
	int channelStep = PATCH_SIZE[k] * PATCH_SIZE[k];

	// y
	for (int i = start; i < end; i++) {
		// x
		for (int j = start; j < end; j++) {
			int pos = (i - start) * PATCH_SIZE[k] + (j - start);

			int ii = (int)(i + lm->y + offset + 0.5f);
			int jj = (int)(j + lm->x + 0.5f);

			ii = MIN(height - 1, MAX(0, ii));
			jj = MIN(width - 1, MAX(0, jj));

			dst[pos * 3] = src[ii * pitch + jj * 3];
			dst[pos * 3 + 1] = src[ii * pitch + jj * 3 + 1];
			dst[pos * 3 + 2] = src[ii * pitch + jj * 3 + 2];
		}
	}

	//ฒน0
	if (PATCH_PAD_X[k]>0)
	{
		for (int j = 0; j<PATCH_PAD_X[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] * 3] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 1] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 2] = 0;
			}
		}
		for (int j = PATCH_SIZE[k] - PATCH_PAD_X[k]; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] * 3] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 1] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 2] = 0;
			}
		}
	}
	else if (PATCH_PAD_Y[k]>0)
	{
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_PAD_Y[k]; i++)
			{
				dst[j*PATCH_SIZE[k] * 3] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 1] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 2] = 0;
			}
		}
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = PATCH_SIZE[k] - PATCH_PAD_Y[k]; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] * 3] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 1] = 0;
				dst[j*PATCH_SIZE[k] * 3 + 2] = 0;
			}
		}
	}
}

static void CropRightColorPatch(const unsigned char* image, int width, int height, int pitch,
	const TPointF* lm, int k, float* mat)
{
	float* dst = mat;
    const unsigned char* src = image;

	int start = -PATCH_SIZE[k] / 2;
	int end = start + PATCH_SIZE[k];
	float offset = PATCH_OFFSET[k] * PATCH_SCALE[k];
	int channelStep = PATCH_SIZE[k] * PATCH_SIZE[k];

	// y
	for (int i = start; i < end; i++) {
		// x
		for (int j = start; j < end; j++) {
			int pos = (j - start) * PATCH_SIZE[k] + (i - start);

			int ii = (int)(i + lm->y + offset + 0.5f);
			int jj = (int)(-j + lm->x + 0.5f);

			ii = MIN(height - 1, MAX(0, ii));
			jj = MIN(width - 1, MAX(0, jj));

			// BGR
			dst[pos + channelStep * 2] = src[ii * pitch + jj * 3];
			dst[pos + channelStep] = src[ii * pitch + jj * 3 + 1];
			dst[pos] = src[ii * pitch + jj * 3 + 2];
		}
	}

	//ฒน0
	if (PATCH_PAD_X[k]>0)
	{
		for (int j = 0; j<PATCH_PAD_X[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep * 2] = 0;
			}
		}
		for (int j = PATCH_SIZE[k] - PATCH_PAD_X[k]; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep * 2] = 0;
			}
		}
	}
	else if (PATCH_PAD_Y[k]>0)
	{
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = 0; i<PATCH_PAD_Y[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep * 2] = 0;
			}
		}
		for (int j = 0; j<PATCH_SIZE[k]; j++)
		{
			for (int i = PATCH_SIZE[k] - PATCH_PAD_Y[k]; i<PATCH_SIZE[k]; i++)
			{
				dst[j*PATCH_SIZE[k] + i] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep] = 0;
				dst[j*PATCH_SIZE[k] + i + channelStep * 2] = 0;
			}
		}
	}
}

namespace learnml {

CropImageUtil::CropImageUtil()
{
    TLandmarks1 mean_points;
    mean_points.count = 25;
    for ( int k = 0; k < MAX_LANDMARKS_NUM - 2; k++ )
    {
        mean_points.pts[k].id = SDM_MEAN_SHAPE[k][0];
        mean_points.pts[k].x = SDM_MEAN_SHAPE[k][1];
        mean_points.pts[k].y = SDM_MEAN_SHAPE[k][2];
    }

    for ( int i = 0; i < PATCH_COUNT + ADDED_PATCH_COUNT; i++ )
    {
        //scale transform
        for ( int j = 0; j < MAX_LANDMARKS_NUM - 2; j++ )
        {
            float transform_x = (mean_points.pts[j].x - 80.0f) * (PATCH_SCALE[i] / 50.0f) + 80.0f;
            float transform_y = (mean_points.pts[j].y - 55.0f) * (PATCH_SCALE[i] / 50.0f) + 55.0f;
            PATCH_MEAN_SHAPE[i][j][0] = transform_x;
            PATCH_MEAN_SHAPE[i][j][1] = transform_y;

        }
        PATCH_MEAN_SHAPE[i][0][0] = (PATCH_MEAN_SHAPE[i][KP_EYE_LEFT_INNER][0] + PATCH_MEAN_SHAPE[i][KP_EYE_LEFT_OUTTER][0]) / 2;
        PATCH_MEAN_SHAPE[i][1][0] = (PATCH_MEAN_SHAPE[i][KP_EYE_RIGHT_INNER][0] + PATCH_MEAN_SHAPE[i][KP_EYE_RIGHT_OUTTER][0]) / 2;
    }
}

CropImageUtil::~CropImageUtil()
{

}

CropImageUtil& CropImageUtil::GetInstance()
{
    CropImageUtil instance;
    return instance;
}


vector<minicv::Mat> CropImageUtil::PreparePatches(minicv::Mat color, const TLandmarks1* lms, int isMirror)
{

    vector<minicv::Mat> matPatches;
    int i, k = 0;
    TLandmarks1 lm1;
    lm1.count = MAX_LANDMARKS_NUM - 2;
    for ( i = 0; i < lms->count; i++ ) {
        if ( lms->pts[i].id == 240 || lms->pts[i].id == 241 ||
            lms->pts[i].id == 315 || lms->pts[i].id == 316 ) {
            continue;
        }
        lm1.pts[k] = lms->pts[i];
        k++;
    }

    qsort(&lm1.pts, lm1.count, sizeof(TLandmark1), LandmarkCMP);

    float src_left_eye_X = (lm1.pts[KP_EYE_LEFT_INNER].x + lm1.pts[KP_EYE_LEFT_OUTTER].x) / 2;
    float src_left_eye_Y = (lm1.pts[KP_EYE_LEFT_INNER].y + lm1.pts[KP_EYE_LEFT_OUTTER].y) / 2;
    float src_right_eye_X = (lm1.pts[KP_EYE_RIGHT_INNER].x + lm1.pts[KP_EYE_RIGHT_OUTTER].x) / 2;
    float src_right_eye_Y = (lm1.pts[KP_EYE_RIGHT_INNER].y + lm1.pts[KP_EYE_RIGHT_OUTTER].y) / 2;

    // 25 patch pairs
    for ( i = 0; i < lm1.count + ADDED_PATCH_COUNT; i++ ) {
        if ( i != 33 )
        {
            continue;
        }

        float a, b, tx, ty;
        float a1, b1, tx1, ty1;
        TPointF landmark[MAX_LANDMARKS_NUM - 2];
        TPointF dstPoints[4] = {
            { PATCH_MEAN_SHAPE[i][KP1_INDEX][0], PATCH_MEAN_SHAPE[i][KP1_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP2_INDEX][0], PATCH_MEAN_SHAPE[i][KP2_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP4_INDEX][0], PATCH_MEAN_SHAPE[i][KP4_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP5_INDEX][0], PATCH_MEAN_SHAPE[i][KP5_INDEX][1] }
        };

        TPointF srcPoints[4] = {
            { src_left_eye_X, src_left_eye_Y },
            { src_right_eye_X, src_right_eye_Y },
            { lm1.pts[KP4_INDEX].x, lm1.pts[KP4_INDEX].y },
            { lm1.pts[KP5_INDEX].x, lm1.pts[KP5_INDEX].y } };

        sim_params_from_points(dstPoints, srcPoints, 4, &a, &b, &tx, &ty);
        sim_params_from_points(srcPoints, dstPoints, 4, &a1, &b1, &tx1, &ty1);

        sim_transform_landmark(lm1.pts, landmark, lm1.count, a, b, tx, ty);

        minicv::Mat dstMat = minicv::Mat(STD_HEIGHT, STD_WIDTH, CV_8UC3);
        sim_transform_color_image(color.data, color.cols, color.rows, color.step.p[0],
            dstMat.data, STD_WIDTH, STD_HEIGHT, a1, b1, tx1, ty1);

        float* colorStd = new float[2 * PATCH_SIZE[i] * PATCH_SIZE[i] * 3];
        TPointF temp_left_point;
        TPointF temp_right_point;
        temp_left_point.x = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][0];
        temp_left_point.y = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][1];
        temp_right_point.x = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][0];
        temp_right_point.y = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][1];

        CropLeftColorPatch(dstMat.data, STD_WIDTH, STD_HEIGHT, STD_WIDTH * 3, &temp_left_point, i, colorStd);

        minicv::Mat CropLeftColorMat = minicv::Mat(PATCH_SIZE[i], PATCH_SIZE[i], CV_8UC3);
        for ( int t = 0; t < PATCH_SIZE[i] * PATCH_SIZE[i] * 3; t++ )
        {
            CropLeftColorMat.data[t] = (unsigned char)colorStd[t];
        }
        delete[] colorStd;
        if ( !PATCH_COLOR[i] )
        {
            minicv::cvtColor(CropLeftColorMat, CropLeftColorMat, CV_BGR2GRAY);
        }

        matPatches.push_back(CropLeftColorMat);
        if ( isMirror == 1 )
        {
            minicv::Mat mirrorMat;
            minicv::flip(CropLeftColorMat, mirrorMat, 1);
            matPatches.push_back(mirrorMat);
        }
    }

    return matPatches;
}

vector<minicv::Mat> CropImageUtil::PrepareSinglePatch(minicv::Mat color, const TLandmarks1* lms, int isMirror)
{
    vector<minicv::Mat> matPatches;
    // remove 240, 241, 315, 316
    int i, k = 0;
    TLandmarks1 lm1;
    lm1.count = MAX_LANDMARKS_NUM - 2;
    for ( i = 0; i < lms->count; i++ ) {
        //printf("aaaaaa: %f, %f, %f\n", lms->pts[i].id, lms->pts[i].x,lms->pts[i].y);
        if ( lms->pts[i].id == 240 || lms->pts[i].id == 241 ||
            lms->pts[i].id == 315 || lms->pts[i].id == 316 ) {
            continue;
        }
        lm1.pts[k] = lms->pts[i];
        k++;
    }

    // sort landmarks by id  
    qsort(&lm1.pts, lm1.count, sizeof(TLandmark1), LandmarkCMP);

    float src_left_eye_X = (lm1.pts[KP_EYE_LEFT_INNER].x + lm1.pts[KP_EYE_LEFT_OUTTER].x) / 2;
    float src_left_eye_Y = (lm1.pts[KP_EYE_LEFT_INNER].y + lm1.pts[KP_EYE_LEFT_OUTTER].y) / 2;
    float src_right_eye_X = (lm1.pts[KP_EYE_RIGHT_INNER].x + lm1.pts[KP_EYE_RIGHT_OUTTER].x) / 2;
    float src_right_eye_Y = (lm1.pts[KP_EYE_RIGHT_INNER].y + lm1.pts[KP_EYE_RIGHT_OUTTER].y) / 2;

    // 25 patch pairs
    for ( i = 0; i < lm1.count + ADDED_PATCH_COUNT; i++ ) {
        if ( i != 1 && i != 4 && i != 15 && i != 19 && i != 23 )
        {
            minicv::Mat tempMat;
            matPatches.push_back(tempMat);
            continue;
        }

        float a, b, tx, ty;
        float a1, b1, tx1, ty1;
        TPointF landmark[MAX_LANDMARKS_NUM - 2];
        TPointF dstPoints[4] = {
            { PATCH_MEAN_SHAPE[i][KP1_INDEX][0], PATCH_MEAN_SHAPE[i][KP1_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP2_INDEX][0], PATCH_MEAN_SHAPE[i][KP2_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP4_INDEX][0], PATCH_MEAN_SHAPE[i][KP4_INDEX][1] },
            { PATCH_MEAN_SHAPE[i][KP5_INDEX][0], PATCH_MEAN_SHAPE[i][KP5_INDEX][1] }
        };

        TPointF srcPoints[4] = {
            { src_left_eye_X, src_left_eye_Y },
            { src_right_eye_X, src_right_eye_Y },
            { lm1.pts[KP4_INDEX].x, lm1.pts[KP4_INDEX].y },
            { lm1.pts[KP5_INDEX].x, lm1.pts[KP5_INDEX].y } };

        sim_params_from_points(dstPoints, srcPoints, 4, &a, &b, &tx, &ty);
        sim_params_from_points(srcPoints, dstPoints, 4, &a1, &b1, &tx1, &ty1);

        sim_transform_landmark(lm1.pts, landmark, lm1.count, a, b, tx, ty);

        minicv::Mat dstMat = minicv::Mat(STD_HEIGHT, STD_WIDTH, CV_8UC3);
        sim_transform_color_image(color.data, color.cols, color.rows, color.step.p[0],
            dstMat.data, STD_WIDTH, STD_HEIGHT, a1, b1, tx1, ty1);

        float* colorStd = new float[2 * PATCH_SIZE[i] * PATCH_SIZE[i] * 3];
        TPointF temp_left_point;
        TPointF temp_right_point;
        temp_left_point.x = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][0];
        temp_left_point.y = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][1];
        temp_right_point.x = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][0];
        temp_right_point.y = PATCH_MEAN_SHAPE[i][PATCH_KP[i][0]][1];

        CropLeftColorPatch(dstMat.data, STD_WIDTH, STD_HEIGHT, STD_WIDTH * 3, &temp_left_point, i, colorStd);

        minicv::Mat CropLeftColorMat = minicv::Mat(PATCH_SIZE[i], PATCH_SIZE[i], CV_8UC3);
        for ( int t = 0; t < PATCH_SIZE[i] * PATCH_SIZE[i] * 3; t++ )
        {
            CropLeftColorMat.data[t] = (unsigned char)colorStd[t];
        }
        delete[] colorStd;
        if ( !PATCH_COLOR[i] )
        {
            minicv::cvtColor(CropLeftColorMat, CropLeftColorMat, CV_BGR2GRAY);
        }
        matPatches.push_back(CropLeftColorMat);


    }

    return matPatches;
}

}
