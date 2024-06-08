#include <fstream>

inline void BGR2Gray(const unsigned char* bgr, int width, int height, int pitch, unsigned char* gray)
{
	int i, j;

	std::ofstream dst1("dst3.dat", std::ios::binary);
	float* grayFloat = new float[width * height];

	// bgr to gray
	const unsigned char* pSrc = bgr;
	unsigned char* dst = gray;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			//dst[j] = (306 * pSrc[j * 3 + 2] + 601 * pSrc[j * 3 + 1] + 117 * pSrc[j * 3]) >> 10;
		//	grayFloat[i * width + j] = 0.299f * pSrc[j * 3 + 2] + 0.587f * pSrc[j * 3 + 1] + 0.114f * pSrc[j * 3];
			
			//int tmp = (4899 * pSrc[j * 3 + 2] + 9617 * pSrc[j * 3 + 1] + 1868 * pSrc[j * 3]) >> 14;
			int tmp = (int)(0.299f * pSrc[j * 3 + 2] + 0.587f * pSrc[j * 3 + 1] + 0.114f * pSrc[j * 3] + 0.5f);
			if (tmp > 255) tmp = 255;
			dst[j] = (unsigned char)(tmp);
		}
		dst += width;
		pSrc += pitch;
	}

	dst1.write((char*)grayFloat, width * height * 4);
	dst1.close();


};