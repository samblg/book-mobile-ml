#ifndef __CUDA_UTILITY__ 
#define __CUDA_UTILITY__ 10000
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include<iostream>
#include<memory>

using namespace std;
// Disable the copy and assignment operator for a class.

const char* cublasGetErrorString(cublasStatus_t error);
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
	if(error != cudaSuccess)\
			{cout<< cudaGetErrorString(error);cudaThreadExit();} \
      }while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    if(status != CUBLAS_STATUS_SUCCESS) \
			{cout<< cublasGetErrorString(status); cudaThreadExit();}\
    } while (0)


class Cuda {
public:
	~Cuda();
	static Cuda& Get();

	// This random number generator facade hides boost and CUDA rng
	// implementation from one another (for cross-platform compatibility).
	
	inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
	
	// Sets the device. Since we have cublas and curand stuff, set device also
	// requires us to reset those values.
	static void setDevice(const int device_id);
	// Prints the current GPU status.
	static void deviceQuery();

protected:
	cublasHandle_t cublas_handle_;
    static std::shared_ptr<Cuda> singleton_;

private:
	// The private constructor to avoid duplicate instantiation.
	Cuda();
	DISABLE_COPY_AND_ASSIGN(Cuda);
};
#endif
