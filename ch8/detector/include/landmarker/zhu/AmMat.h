#ifndef AMMAT_H
#define AMMAT_H

#include <string>
#include <iostream>
#include <cstring>

using namespace std;

class AmMat
{
public:
	AmMat() : width(0), height(0), pitch(0), data(nullptr)
	{
	};

	AmMat(int height, int width, int pitch)
	{
		this->height = height;
		this->width = width;
		this->pitch = pitch;
		this->data = new unsigned char[pitch * height];
		memset(this->data, 0, pitch * height);
	};
	AmMat(int height, int width, int pitch, unsigned char* data)
	{
		this->height = height;
		this->width = width;
		this->pitch = pitch;
		this->data = new unsigned char[pitch * height];

		memcpy(this->data, data, pitch * height);
	};
	AmMat(const AmMat& amMat)
	{
		if (pitch * height != amMat.pitch * amMat.height)
		{
			if (NULL != this->data){
				delete[] this->data;
				data = nullptr;
			}
			this->data = new unsigned char[amMat.pitch * amMat.height];
		}
		this->width = amMat.width;
		this->height = amMat.height;
		this->pitch = amMat.pitch;
		memcpy(this->data, amMat.data, pitch * height);
	};
	
	AmMat(AmMat&& amMat) {
		this->width = amMat.width;
		this->height = amMat.height;
		this->pitch = amMat.pitch;
		this->data = amMat.data;

		amMat.width = 0;
		amMat.height = 0;
		amMat.pitch = 0;
		amMat.data = nullptr;
	}

	AmMat& operator = (const AmMat& amMat)
	{
		if (pitch * height != amMat.pitch * amMat.height)
		{
			if (NULL != this->data){
				delete[] this->data;
				data = nullptr;
			}
			this->data = new unsigned char[amMat.pitch * amMat.height];
		}

		this->width = amMat.width;
		this->height = amMat.height;
		this->pitch = amMat.pitch;

		memcpy(this->data, amMat.data, pitch * height);

		return *this;
	}

	AmMat Am_colRange(int left, int right);
	AmMat Am_rowRange(int top, int bottom);
	AmMat Am_spilt(int top, int bottom, int left, int right);
	AmMat Am_resize(int dstW, int dstH, int bpp);

	~AmMat()
	{
		if (this->data != NULL)
		{
			delete[] data;
			data = nullptr;
		}
	}

public:
	int width;
	int height;
	int pitch;
	unsigned char* data;
};

#endif
