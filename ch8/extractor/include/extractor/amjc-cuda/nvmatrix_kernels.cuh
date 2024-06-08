/*
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif
#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_
#include<iostream>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define uint unsigned int

#define NUM_BLOCKS_MAX                      65535

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Default grid/block sizes for the various functions.
 */
#define ADD_BLOCK_SIZE                      16

#define NUM_TILE_BLOCKS                     4096
#define NUM_TILE_THREADS_PER_BLOCK          512

#define ELTWISE_THREADS_X                   32
#define ELTWISE_THREADS_Y                   8

#define NUM_SUM_COLS_THREADS_PER_BLOCK      256

#define AGG_SHORT_ROWS_THREADS_X            32
#define AGG_SHORT_ROWS_THREADS_Y            8
#define AGG_SHORT_ROWS_LOOPS_Y              32

#define DP_BLOCKSIZE                        512
#define CPUSUM_MAX                          4096

#define ADD_VEC_THREADS_X                   64
#define ADD_VEC_THREADS_Y                   4

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define MYMAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef MUL24 // legacy
#define MUL24(x,y) ((x) * (y))
#endif
#define divideUpwards(a,b) ((a)+(b) - 1)/ (b)
#define AWR_NUM_THREADS           256
#define WARP_SIZE                 32
#define AWR_NUM_WARPS             AWR_NUM_THREADS / WARP_SIZE
#define AWR_LOG_NUM_THREADS       8
#define LOG_WARP_SIZE             5
#define AWR_LOG_NUM_WARPS         3

__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight);
__global__ void kDotProduct_r(float* a, float* b, float* target, const uint numCols, const uint numElements);
__global__ void kSetupCurand(curandState *state, unsigned long long seed);
__global__ void kNormCrossMap(float * src, float *dst, int sizeF, int imgLen, int imgPixels, int numFilters, int numImages, float addScale, float powScale);

/*
 * For now this is supported only for arrays with the same transposedness.
 */
template<class Op>
__global__ void kEltwiseTernaryOp(const float* a, const float* b, const float* c, float* const dest,
	const uint height, const uint width, uint strideA, const uint strideB, const uint strideC,
	const uint strideDest, Op op) {
	const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
	const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

	for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
		for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
			dest[y * strideDest + x] = op(a[y * strideA + x], b[y * strideB + x], c[y * strideC + x]);
		}
	}
}

/*
 * dest here is assumed to be "not transposed" -- height and width correspond to it.
 * b is assumed to be transposed.
 * a can be either transposed or not -- depending on parameter.
 *
 * Performs dest := op(a, b)
 */
template<class Op, bool checkBounds, bool aTrans, bool reverse>
__global__ void kEltwiseBinaryOpTrans(const float* a, const float* b, float* const dest,
	const uint height, const uint width,
	const uint strideA, const uint strideB, const uint strideDest, Op op) {

	__shared__ float shmem[ELTWISE_THREADS_X][ELTWISE_THREADS_X + 1];

	// x here because that's how much work we do
	for (uint by = ELTWISE_THREADS_X * blockIdx.y; by < height; by += ELTWISE_THREADS_X * gridDim.y) {
		for (uint bx = ELTWISE_THREADS_X * blockIdx.x; bx < width; bx += ELTWISE_THREADS_X * gridDim.x) {
			const uint readX = by + threadIdx.x;
			const uint readY = bx + threadIdx.y;

			for (uint y = 0; y < ELTWISE_THREADS_X; y += ELTWISE_THREADS_Y) {
				if (!checkBounds || (readX < height && readY + y < width)) {
					if (aTrans) {
						shmem[threadIdx.x][threadIdx.y + y] = reverse ? op(b[(readY + y) * strideB + readX], a[(readY + y) * strideA + readX])
							: op(a[(readY + y) * strideA + readX], b[(readY + y) * strideB + readX]);
					}
					else {
						shmem[threadIdx.x][threadIdx.y + y] = b[(readY + y) * strideB + readX];
					}
				}
			}
			__syncthreads();

			const uint writeX = bx + threadIdx.x;
			const uint writeY = by + threadIdx.y;

			for (uint y = 0; y < ELTWISE_THREADS_X; y += ELTWISE_THREADS_Y) {
				if (!checkBounds || (writeX < width && writeY + y < height)) {
					if (aTrans) {
						dest[(writeY + y) * strideDest + writeX] = shmem[threadIdx.y + y][threadIdx.x];
					}
					else {
						dest[(writeY + y) * strideDest + writeX] = reverse ? op(shmem[threadIdx.y + y][threadIdx.x], a[(writeY + y) * strideA + writeX])
							: op(a[(writeY + y) * strideA + writeX], shmem[threadIdx.y + y][threadIdx.x]);
					}
				}
			}
			__syncthreads();
		}
	}
}
template<class Op>
__global__ void kEltwiseBinaryOp(const float* a, const float* b, float* const dest, const uint height, const uint width,
	const uint strideA, const uint strideB, const uint strideDest, Op op) {
	const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
	const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

	for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
		for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
			dest[y * strideDest + x] = op(a[y * strideA + x], b[y * strideB + x]);
		}
	}
}

/*
 * dest here is assumed to be "not transposed" -- height and width correspond to it.
 */
template<class Op, bool checkBounds>
__global__ void kEltwiseUnaryOpTrans(const float* a, float* const dest,
	const uint height, const uint width,
	const uint strideA, const uint strideDest, Op op) {

	__shared__ float shmem[ELTWISE_THREADS_X][ELTWISE_THREADS_X + 1];

	for (uint by = ELTWISE_THREADS_X * blockIdx.y; by < height; by += ELTWISE_THREADS_X * gridDim.y) {
		for (uint bx = ELTWISE_THREADS_X * blockIdx.x; bx < width; bx += ELTWISE_THREADS_X * gridDim.x) {
			const uint readX = by + threadIdx.x;
			const uint readY = bx + threadIdx.y;
			for (uint y = 0; y < ELTWISE_THREADS_X; y += ELTWISE_THREADS_Y) {
				if (!checkBounds || (readX < height && readY + y < width)) {
					shmem[threadIdx.x][threadIdx.y + y] = op(a[(readY + y) * strideA + readX]);
				}
			}
			__syncthreads();

			const uint writeX = bx + threadIdx.x;
			const uint writeY = by + threadIdx.y;
			for (uint y = 0; y < ELTWISE_THREADS_X; y += ELTWISE_THREADS_Y) {
				if (!checkBounds || (writeX < width && writeY + y < height)) {
					dest[(writeY + y) * strideDest + writeX] = shmem[threadIdx.y + y][threadIdx.x];

				}
			}
			__syncthreads();
		}
	}
}

template<class Op>
__global__ void kEltwiseUnaryOp(const float* a, float* const dest, const uint height, const uint width,
	const uint strideA, const uint strideDest, Op op) {
	const uint idxX = blockIdx.x * ELTWISE_THREADS_X + threadIdx.x;
	const uint idxY = blockIdx.y * ELTWISE_THREADS_Y + threadIdx.y;

	for (uint y = idxY; y < height; y += gridDim.y * ELTWISE_THREADS_Y) {
		for (uint x = idxX; x < width; x += gridDim.x * ELTWISE_THREADS_X) {
			dest[y * strideDest + x] = op(a[y * strideA + x]);
		}
	}
}

/*
 * Matrix in ROW-MAJOR order!
 */
template <class Op>
__global__ void kRowVectorOp(const float* mat, const float* vec, float* const tgtMat, const uint width, const uint height,
	const uint matStride, const uint tgtStride, Op op) {
	__shared__ float shVec[ADD_VEC_THREADS_X];
	const uint bx = ADD_VEC_THREADS_X * blockIdx.x;
	const uint by = ADD_VEC_THREADS_Y * blockIdx.y;

	for (uint x = bx; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
		__syncthreads();
		if (x + threadIdx.x < width && threadIdx.y == 0) {
			shVec[threadIdx.x] = vec[x + threadIdx.x];
		}
		__syncthreads();

		if (x + threadIdx.x < width) {
			for (uint y = by + threadIdx.y; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
				tgtMat[y * tgtStride + x + threadIdx.x] = op(mat[y * matStride + x + threadIdx.x], shVec[threadIdx.x]);
			}
		}
	}
}

/*
 * Matrix in ROW-MAJOR order!
 */
template <class Op>
__global__ void kColVectorOp(const float* mat, const float* vec, float* const tgtMat,
	const uint width, const uint height,
	const uint matStride, const uint tgtStride, Op op) {
	__shared__ float shVec[ADD_VEC_THREADS_Y];
	const uint by = ADD_VEC_THREADS_Y * blockIdx.y;
	const uint bx = ADD_VEC_THREADS_X * blockIdx.x;
	//    const uint matIdx = (by + threadIdx.y) * matStride + bx + threadIdx.x;
	//    const uint tgtIdx = (by + threadIdx.y) * tgtStride + bx + threadIdx.x;
	const uint tidx = ADD_VEC_THREADS_X * threadIdx.y + threadIdx.x;

	for (uint y = by; y < height; y += gridDim.y * ADD_VEC_THREADS_Y) {
		__syncthreads();
		if (y + tidx < height && tidx < ADD_VEC_THREADS_Y) {
			shVec[tidx] = vec[y + tidx];
		}
		__syncthreads();

		if (y + threadIdx.y < height) {
			for (uint x = bx + threadIdx.x; x < width; x += gridDim.x * ADD_VEC_THREADS_X) {
				tgtMat[(y + threadIdx.y) * tgtStride + x] = op(mat[(y + threadIdx.y) * matStride + x], shVec[threadIdx.y]);
			}
		}
	}
}

/*
 * This one gets coalesced reads but computes only a partial sum which
 * must either be summed again (recursively) or summed on the host.
 */
template<class Agg, class BinaryOp, int blockSize>
__global__ void kAggRows(const float* mat, float* matSum, const uint width, const uint height, const uint sumWidth, Agg agg, BinaryOp op) {
	const int idxX = blockIdx.x * blockSize * 2 + threadIdx.x;

	__shared__ float accum[blockSize * 2];

	matSum += blockIdx.y * sumWidth + blockIdx.x;
	/*
	 * Here it's important to make sure that all threads in a block call __syncthreads,
	 * so I have even the redundant threads (for which idxX >= width) enter this loop
	 * just so that they may call __syncthreads at the appropriate times.
	 */
	mat += width * blockIdx.y + idxX;

	accum[threadIdx.x] = agg.getBaseValue();
	accum[threadIdx.x + blockSize] = agg.getBaseValue();
	for (uint idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
		if (idxX < width) {
			accum[threadIdx.x] = mat[0];
			if (idxX + blockSize < width)
				accum[threadIdx.x + blockSize] = mat[blockSize];
		}
		if (blockSize >= 512) {
			__syncthreads();
			if (threadIdx.x < 512)
				accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 512]);
		}
		if (blockSize >= 256) {
			__syncthreads();
			if (threadIdx.x < 256)
				accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 256]);
		}
		if (blockSize >= 128) {
			__syncthreads();
			if (threadIdx.x < 128)
				accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 128]);
		}
		if (blockSize >= 64) {
			__syncthreads();
			if (threadIdx.x < 64)
				accum[threadIdx.x] = agg(accum[threadIdx.x], accum[threadIdx.x + 64]);
		}

		__syncthreads();
		volatile float* myAccum = &accum[threadIdx.x];
		if (threadIdx.x < 32) { // executed only by first warp
			myAccum[0] = agg(myAccum[0], myAccum[32]);
			myAccum[0] = agg(myAccum[0], myAccum[16]);
			myAccum[0] = agg(myAccum[0], myAccum[8]);
			myAccum[0] = agg(myAccum[0], myAccum[4]);
			myAccum[0] = agg(myAccum[0], myAccum[2]);
			myAccum[0] = agg(myAccum[0], myAccum[1]);
		}

		if (threadIdx.x == 0) {
			matSum[0] = op(matSum[0], myAccum[0]);
			matSum += gridDim.y * sumWidth;
		}
		__syncthreads();
		mat += width * gridDim.y;
	}
}

template<class Agg, class BinaryOp>
__global__ void kAggRows_wholerow(const float* mat, float* matSum, const uint width, const uint height, Agg agg, BinaryOp op) {
	const int tidx = threadIdx.x;

	__shared__ float accum[AWR_NUM_THREADS];
	volatile float* vMyAccum = &accum[tidx];
	float* myAccum = &accum[tidx];

	matSum += blockIdx.y;
	mat += width * blockIdx.y;

	for (uint idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
		myAccum[0] = agg.getBaseValue();
		for (uint x = tidx; x < width; x += AWR_NUM_THREADS) {
			myAccum[0] = agg(myAccum[0], mat[x]);
		}
#pragma unroll
		for (uint i = AWR_LOG_NUM_THREADS - 1; i > LOG_WARP_SIZE; i--) {
			const uint d = 1 << i;
			__syncthreads();
			if (tidx < d) {
				myAccum[0] = agg(myAccum[0], myAccum[d]);
			}
		}
		__syncthreads();
		if (tidx < WARP_SIZE) {
#pragma unroll
			for (int i = LOG_WARP_SIZE; i >= 0; i--) {
				const uint d = 1 << i;
				vMyAccum[0] = agg(vMyAccum[0], vMyAccum[d]);
			}

			if (tidx == 0) {
				matSum[0] = op(matSum[0], vMyAccum[0]);
				matSum += gridDim.y;
			}
		}
		__syncthreads();
		mat += width * gridDim.y;
	}
}

/*
 * Implements multiscan idea from http://www.moderngpu.com
 * Not really useful for pure reductions but neat nonetheless.
 */
template<class Agg, class BinaryOp>
__global__ void kAggRows_wholerow_nosync(const float* mat, float* matSum, const uint width, const uint height,
	Agg agg, BinaryOp op) {
	const uint tidx = threadIdx.x;
	const uint warpIdx = tidx / WARP_SIZE;
	const uint tidxInWarp = tidx % WARP_SIZE;

	__shared__ float accum[(WARP_SIZE + 1) * AWR_NUM_WARPS + WARP_SIZE / 2];
	__shared__ float finalAccum[AWR_NUM_WARPS + AWR_NUM_WARPS / 2];

	float* myAccum = &accum[warpIdx * (WARP_SIZE + 1) + tidxInWarp];
	volatile float* vMyAccum = &accum[warpIdx * (WARP_SIZE + 1) + tidxInWarp];
	matSum += blockIdx.y;
	mat += width * blockIdx.y;

	for (uint idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
		float rAccum = agg.getBaseValue(); // cache in register, a bit faster than shmem
		for (uint x = tidx; x < width; x += AWR_NUM_THREADS) {
			rAccum = agg(rAccum, mat[x]);
		}
		myAccum[0] = rAccum;

		// Each warp does a reduction that doesn't require synchronizatoin
#pragma unroll
		for (uint i = 0; i < LOG_WARP_SIZE; i++) {
			const uint d = 1 << i;
			vMyAccum[0] = agg(vMyAccum[0], vMyAccum[d]);
		}
		__syncthreads();
		// The warps write their results
		if (tidx < AWR_NUM_WARPS) {
			volatile float* vMyFinalAccum = &finalAccum[tidx];
			vMyFinalAccum[0] = accum[tidx * (WARP_SIZE + 1)];
#pragma unroll
			for (uint i = 0; i < AWR_LOG_NUM_WARPS; i++) {
				const uint d = 1 << i;
				vMyFinalAccum[0] = agg(vMyFinalAccum[0], vMyFinalAccum[d]);
			}
			if (tidx == 0) {
				matSum[0] = op(matSum[0], vMyFinalAccum[0]);
				matSum += gridDim.y;
			}
		}
		__syncthreads();

		mat += width * gridDim.y;
	}
}

/*
 * To be used when the rows are <= 64.
 *
 * TODO: try to reduce reg usage. i think this can be made faster too.
 */
//#define AGG_SHORT_ROWS_LOOPS_X  4
template <class Agg, class BinaryOp, int LOOPS_X, int THREADS_X>
__global__ void kAggShortRows(const float* mat, float* matSum, const uint width, const uint height, Agg agg, BinaryOp op) {
	const uint shmemX = THREADS_X + 1;
	__shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];

	const uint tidx = threadIdx.y * THREADS_X + threadIdx.x;
	const uint ty = LOOPS_X == 1 ? tidx / width : threadIdx.y; // when loops==1, width is gonna be smaller than block x dim
	const uint tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
	const uint bidx = blockIdx.y * gridDim.x + blockIdx.x;
	const uint blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;
	float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
	matSum += blockRowIdx + tidx;
	//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
	mat += width * blockRowIdx + MUL24(ty, width) + tx;
	float* shmemWriteZeros = &shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x];

	bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y;

	if (blockRowIdx < height) {
#pragma unroll
		for (uint y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
			doAgg &= tidx + y + blockRowIdx < height;
			const bool heightIdxOK = ty < AGG_SHORT_ROWS_THREADS_Y && ty + y + blockRowIdx < height;

			shmemWriteZeros[0] = agg.getBaseValue();
			__syncthreads();
#pragma unroll
			for (uint x = 0; x < LOOPS_X * THREADS_X; x += THREADS_X) {
				//                __syncthreads();
				if (heightIdxOK && x + tx < width) {
					shmemWrite[0] = agg(mat[x], shmemWrite[0]);
				}
			}
			__syncthreads();
			if (doAgg) {
				/*
				 * I tried doing this final sum as a 4-step reduction, with 8 threads
				 * per warp participating. It was slightly slower.
				 */
				float accum = agg.getBaseValue();
				float* shmemRead = shmem + MUL24(tidx, shmemX);
				// this loops too much if the rows are really short :(
#pragma unroll
				for (uint i = 0; i < THREADS_X; i++) {
					accum = agg(accum, shmemRead[0]);
					shmemRead++;
				}
				matSum[0] = op(matSum[0], accum);
				matSum += AGG_SHORT_ROWS_THREADS_Y;
			}
			__syncthreads();
			mat += width * AGG_SHORT_ROWS_THREADS_Y;
		}
	}
}

template <class Agg, class BinaryOp>
__global__ void kAggShortRows2(const float* mat, float* matSum, const uint width, const uint height, Agg agg, BinaryOp op) {
	const uint shmemX = AGG_SHORT_ROWS_THREADS_X + 1;
	__shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];
	const uint LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);
	const uint tidx = threadIdx.y * AGG_SHORT_ROWS_THREADS_X + threadIdx.x;

	const uint bidx = blockIdx.y * gridDim.x + blockIdx.x;
	const uint blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;

	float* shmemWrite = shmem + MUL24(threadIdx.y, shmemX) + threadIdx.x;
	matSum += blockRowIdx + tidx;
	//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
	mat += width * blockRowIdx + MUL24(threadIdx.y, width) + threadIdx.x;

	bool doAgg = tidx < AGG_SHORT_ROWS_THREADS_Y;
	if (blockRowIdx < height) {
		for (uint y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
			doAgg &= tidx + y + blockRowIdx < height;
			const bool heightIdxOK = threadIdx.y + y + blockRowIdx < height;
			float accum = agg.getBaseValue();
			shmemWrite[0] = agg.getBaseValue();

			for (uint x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x += AGG_SHORT_ROWS_THREADS_X) {
				//                __syncthreads();
				if (heightIdxOK && x + threadIdx.x < width) {
					shmemWrite[0] = agg(mat[x], shmemWrite[0]);
				}
			}

			__syncthreads();
			if (doAgg) {
				float* shmemRead = shmem + MUL24(tidx, shmemX);

#pragma unroll
				for (uint i = 0; i < AGG_SHORT_ROWS_THREADS_X; i++) {
					accum = agg(accum, shmemRead[0]);
					shmemRead++;
				}

				matSum[0] = op(matSum[0], accum);
				matSum += AGG_SHORT_ROWS_THREADS_Y;
			}
			__syncthreads();
			mat += width * AGG_SHORT_ROWS_THREADS_Y;
		}
	}
}

/*
 * Bad when there are few columns.
 */
template <class Agg, class BinaryOp>
__global__ void kDumbAggCols(const float* mat, float* const vec, const uint width, const uint height, Agg agg, BinaryOp op) {
	const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	mat += idx;
	if (idx < width) {
		float mx = *mat;
		mat += width;
		for (uint j = 1; j < height; j++) {
			mx = agg(*mat, mx);
			mat += width;
		}
		vec[idx] = op(vec[idx], mx);
	}
}

template <class Agg>
__global__ void kTotalAgg(const float* a, float* const target, const uint numCols, const uint numElements, Agg agg) {
	__shared__ float shmem[DP_BLOCKSIZE];
	uint eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
	shmem[threadIdx.x] = agg.getBaseValue();
	if (eidx < numCols) {
		for (; eidx < numElements; eidx += numCols) {
			shmem[threadIdx.x] = agg(shmem[threadIdx.x], a[eidx]);
		}
	}
	__syncthreads();
	if (threadIdx.x < 256) {
		shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 256]);
	}
	__syncthreads();
	if (threadIdx.x < 128) {
		shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 128]);
	}
	__syncthreads();
	if (threadIdx.x < 64) {
		shmem[threadIdx.x] = agg(shmem[threadIdx.x], shmem[threadIdx.x + 64]);
	}
	__syncthreads();
	if (threadIdx.x < 32) {
		volatile float* mysh = &shmem[threadIdx.x];
		*mysh = agg(*mysh, mysh[32]);
		*mysh = agg(*mysh, mysh[16]);
		*mysh = agg(*mysh, mysh[8]);
		*mysh = agg(*mysh, mysh[4]);
		*mysh = agg(*mysh, mysh[2]);
		*mysh = agg(*mysh, mysh[1]);
		if (threadIdx.x == 0) {
			target[blockIdx.x] = *mysh;
		}
	}
}

class AddGaussianUnaryRandomizer {
private:
	const float stdev;
public:
	AddGaussianUnaryRandomizer(float _stdev) : stdev(_stdev) {
	}
	__device__ inline float operator ()(float data, curandState* state) {
		return data + stdev * curand_normal(state);
	}
};

class BinarizeUnaryRandomizer {
public:
	__device__ inline float operator ()(float data, curandState* state) {
		return data > curand_uniform(state);
	}
};

class UniformUnaryRandomizer {
public:
	__device__ inline float operator ()(float data, curandState* state) {
		return curand_uniform(state);
	}
};

class GaussianUnaryRandomizer {
private:
	const float mean, stdev;
public:
	GaussianUnaryRandomizer(float _mean, float _stdev) : mean(_mean), stdev(_stdev) {
	}
	__device__ inline float operator ()(float data, curandState* state) {
		return mean + stdev * curand_normal(state);
	}
};

template <bool var>
class AddGaussianBinaryRandomizer {
public:
	__device__ inline float operator ()(float data, float stdev, curandState* state) {
		return data + (var ? stdev : 1) * stdev * curand_normal(state);
	}
};

class GaussianBinaryRandomizer {
public:
	__device__ inline float operator ()(float data, float stdev, curandState* state) {
		return stdev * curand_normal(state);
	}
};

template<class Randomizer>
__global__ void kUnaryRandomize(float* data, float* targets, curandState* state, const uint numElements, Randomizer rnd) {
	const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
	curandState localState = state[tidx];

	for (uint i = tidx; i < numElements; i += NUM_RND_STREAMS) {
		targets[i] = rnd(data[i], &localState);
	}
	state[tidx] = localState;
}

template<class Randomizer>
__global__ void kBinaryRandomize(float* data, float* data2, float* targets, curandState* state, const uint numElements, Randomizer rnd) {
	const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
	curandState localState = state[tidx];

	for (uint i = tidx; i < numElements; i += NUM_RND_STREAMS) {
		targets[i] = rnd(data[i], data2[i], &localState);
	}
	state[tidx] = localState;
}
template<typename T>
__global__ void im2col_gpu_local_kernel(T*stacked, T const* data, const int numPatchesX,
	const int numPatchesY, const int numPatchSlices,
	const int width, const int height,
	const int windowWidth, const int windowHeight,
	const int strideX, const int strideY,
	const int padLeft, const int padTop,const int imgLen,const int stackCol )
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numPatchSlices)
	{
		int x = index % numPatchesX;
		int y = index / numPatchesX % numPatchesY;
		int z = index / numPatchesX / numPatchesY;
		int x_data = x * strideX - padLeft;
		int y_data = y * strideY - padTop;
		data += (z * height + y_data) * width + x_data;
		int patchSliceOffset =  (windowWidth*windowHeight) * z;
		stacked += (y * numPatchesX + x) * imgLen * stackCol + (numPatchesY * patchSliceOffset + y) * numPatchesX + x;
		for (int v = 0; v < windowHeight; ++v) {
			for (int u = 0; u < windowWidth; ++u) {
				if (y_data + v >= 0 &&
					y_data + v < height &&
					x_data + u >= 0 &&
					x_data + u < width) {
					*stacked = data[v * width + u];
				}
				else {
					*stacked = 0;
				}
				stacked += (numPatchesX*numPatchesY);
			}
		}
	}
}
template <typename T>
__global__ void
im2col_gpu_kernel(T* stacked, T const* data, const int numPatchesX,
const int numPatchesY, const int numPatchSlices,
const int width, const int height,
const int windowWidth, const int windowHeight,
const int strideX, const int strideY,
const int padLeft, const int padTop)
{
	/* each kernel copies the pixels in an image patch for one channel */
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numPatchSlices) {
		/*
		  get the patch slice (x,y,z) to copy
		  */
		int x = index;
		int y = x / numPatchesX;
		int z = y / numPatchesY;
		x %= numPatchesX;
		y %= numPatchesY;

		/*
		 pick the top-left corer of the patch slice in the input image
		 */
		int x_data = x * strideX - padLeft;
		int y_data = y * strideY - padTop;
		data += (z * height + y_data) * width + x_data;

		/*
		 pick the column of the stacked image which contains this patch,
		 and move down along the column at the beginning of the patch slice
		 */
		int patchSliceOffset = (windowWidth*windowHeight) * z;
		stacked += (numPatchesY * patchSliceOffset + y) * numPatchesX + x;

		/*
		 copy the patch slice
		 */
		for (int v = 0; v < windowHeight; ++v) {
			for (int u = 0; u < windowWidth; ++u) {
				if (y_data + v >= 0 &&
					y_data + v < height &&
					x_data + u >= 0 &&
					x_data + u < width) {
					*stacked = data[v * width + u];
				}
				else {
					*stacked = 0;
				}
				stacked += (numPatchesX*numPatchesY);
			}
		}
	}
}


template <typename T>
void im2col_gpu(T* stacked, T const* data, size_t width, size_t height,
	size_t depth, size_t windowWidth, size_t windowHeight,
	size_t strideX, size_t strideY, size_t padLeft,
	size_t padRight, size_t padTop, size_t padBottom)
{
	int numPatchesX = (width + (padLeft + padRight) - windowWidth) / strideX + 1;
	int numPatchesY = (height + (padTop + padBottom) - windowHeight) / strideY + 1;
	int numPatchSlices = numPatchesX * numPatchesY * depth;

	/*
	 Each kernel copies a feature dimension of a patch.
	 */
	im2col_gpu_kernel<T>
		<< < divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >> >
		(stacked, data, numPatchesX, numPatchesY, numPatchSlices, width, height,
		windowWidth, windowHeight, strideX, strideY, padLeft, padTop);
	if (cudaPeekAtLastError() != cudaSuccess) {
		std::cout << "im2col: CUDA kernel error ("
			<< cudaGetErrorString(cudaPeekAtLastError()) << ")" << std::endl;
	}
}

template <typename T>
void im2col_gpu_local(T* stacked, T const* data, size_t width, size_t height,
	size_t depth, size_t windowWidth, size_t windowHeight,
	size_t strideX, size_t strideY, size_t padLeft,
	size_t padRight, size_t padTop, size_t padBottom)
{
	int numPatchesX = (width + (padLeft + padRight) - windowWidth) / strideX + 1;
	int numPatchesY = (height + (padTop + padBottom) - windowHeight) / strideY + 1;
	int numPatchSlices = numPatchesX * numPatchesY * depth;
	int imgLen = windowWidth * windowHeight * depth;
	int stackCol = numPatchesY * numPatchesX;
	/*
	Each kernel copies a feature dimension of a patch.
	*/
	int threadsPerBlock = VL_CUDA_NUM_THREADS / 2;
	//threadsPerBlock = 1024;
	im2col_gpu_local_kernel<T>
		<< < divideUpwards(numPatchSlices, threadsPerBlock), threadsPerBlock >> >
		(stacked, data, numPatchesX, numPatchesY, numPatchSlices, width, height,
		windowWidth, windowHeight, strideX, strideY, padLeft, padTop, imgLen, stackCol);
	if (cudaPeekAtLastError() != cudaSuccess) {
		std::cout << "im2col: CUDA kernel error ("
			<< cudaGetErrorString(cudaPeekAtLastError()) << ")" << std::endl;
	}
}
// Explicit instantiation
template void im2col_gpu<float>(float* stacked,
	float const* data,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);
template void im2col_gpu<double>(double* stacked,
	double const* data,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);

template void im2col_gpu_local<float>(float* stacked,
	float const* data,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);
template void im2col_gpu_local<double>(double* stacked,
    double const* data,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);

/* ---------------------------------------------------------------- */
/*                                                     col2im (GPU) */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void col2im_gpu_kernel(T* data,
	T const* stacked,
	const int numPatchesX,
	const int numPatchesY,
	const int dataVolume,
	const int width,
	const int height,
	const int depth,
	const int windowWidth,
	const int windowHeight,
	const int strideX,
	const int strideY,
	const int padLeft,
	const int padTop)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < dataVolume)
	{
		T accumulator = 0;
		/*
		 This kernel accumulates on data[index] all elements in stacked
		 that receive copies of data[index] in im2col.

		 Consider coordinate (x_data,y_data) in the input image. Relative to patch
		 (x,y), this has offset

		 u = x_data - (x * strideX - padLeft)
		 v = y_data - (y * strideY - padRight)

		 In particular, (x_data,y_data) is contained (and hence contributes)
		 to patch (x,y) if, and only if,

		 0 <= u < windowWidth  <==>  1) x_data >= x * strideX - padLeft
		 2) x_data <  x * strideX - padLeft + windowWidth

		 and similar for y.

		 Hence, the patches that contribute to (x_data,y_data) are given
		 by indexes (x,y) such that

		 (x_data + padLeft - windowWidth)/stride < x
		 <= (x_data + padLeft)/stride

		 or, accounting for the boundaries,

		 x1 <= x <= x2, such that
		 x1 = max(0,  1 + floor(x_data + padLeft - windowWidth)/stride),
		 x2 = min(numPatchesX-1,  floor(x_data + padLeft)/stride),

		 and similar for y.

		 Note that (x_data + padLeft - windowWidth) may be negative. In this case,
		 the C convention for rounding division towards zero fails to compute
		 the floor() properly. Instead, we check this case explicitly and set
		 */

		int x_data = index;
		int y_data = x_data / width;
		int z = y_data / height;
		x_data %= width;
		y_data %= height;

		int dx = x_data + padLeft - windowWidth;
		int dy = y_data + padTop - windowHeight;
		int x1 = (dx >= 0) ? dx / strideX + 1 : 0;
		int y1 = (dy >= 0) ? dy / strideY + 1 : 0;
		int x2 = min((x_data + padLeft) / strideX, numPatchesX - 1);
		int y2 = min((y_data + padTop) / strideY, numPatchesY - 1);

		/*
		 Knowing which patches (x,y) contribute to (x_data,y_data) is not enough;
		 we need to determine the specific element within each patch. This
		 is given by the offset as given above:

		 u(x) = x_data - (x * strideX - padLeft)
		 v(y) = y_data - (y * strideY - padRight)

		 Now we can comptute the indeces of the elements of stacked[] to accumulate:

		 stackedIndex(x,y) =
		 (y * numPatchesX + x) +                 // column offset
		 ((z * windowHeight + v(y)) * windowWidth + u(x)) *  // within patch offset
		 (numPatchesX*numPatchesY)

		 Substituting the expression fo u(x), we find

		 stackedIndex(x,y) =
		 = (y * numPatchesX + x)
		 + ((z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
		 * (numPatchesX*numPatchesY)
		 - ((y * strideY) * windowWidth + x * strideX)
		 * (numPatchesX*numPatchesY)
		 = (z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
		 + x * (1 - strideX*numPatchesY*numPatchesX)
		 + y * (1 - strideY*numPatchesY*windowWidth)*numPatchesX ;

		 */

		int deltax = (1 - strideX * numPatchesY * numPatchesX);
		int deltay = (1 - strideY * numPatchesY * windowWidth) * numPatchesX;
		stacked += ((z * windowHeight + y_data + padTop) * windowWidth + (x_data + padLeft)) * (numPatchesX*numPatchesY);

		for (int y = y1; y <= y2; ++y) {
			for (int x = x1; x <= x2; ++x) {
				accumulator += stacked[y * deltay + x * deltax];
			}
		}
		data[index] = accumulator;
	}
}

template <typename T>
void col2im_gpu(T* data,
	T const* stacked,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom)
{
	/*
	 each kernel integrates all contributions to a particular element
	 of data.
	 */
	int numPatchesX = (width + (padLeft + padRight) - windowWidth) / strideX + 1;
	int numPatchesY = (height + (padTop + padBottom) - windowHeight) / strideY + 1;
	int dataVolume = width * height * depth;

	col2im_gpu_kernel<T>
		<< < divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >> >
		(data,
		stacked,
		numPatchesX,
		numPatchesY,
		dataVolume,
		width, height, depth,
		windowWidth, windowHeight,
		strideX, strideY,
		padLeft, padTop);

	if (cudaPeekAtLastError() != cudaSuccess) {
		std::cout
			<< "col2im: CUDA kernel error ("
			<< cudaGetErrorString(cudaPeekAtLastError())
			<< ")" << std::endl;
	}
}

template void col2im_gpu<float>(float* data,
	float const* stacked,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);

template void col2im_gpu<double>(double* data,
	double const* stacked,
	size_t width,
	size_t height,
	size_t depth,
	size_t windowWidth,
	size_t windowHeight,
	size_t strideX,
	size_t strideY,
	size_t padLeft,
	size_t padRight,
	size_t padTop,
	size_t padBottom);

/*bool hasGPU()//true if there exists cuda gpu,otherwise false
{
int devices = 0;
cudaError_t err = cudaGetDeviceCount(&devices);
if (devices > 0 && err == cudaSuccess)
return true;
else
return false;
}*/
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
	const int numImages, const int numFilters,
	const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
	const int moduleStride,
	const int numModulesY, const int numModulesX, const int imgStride,
	const float scaleTargets, const float scaleOutputs,
	const bool conv) {
	__shared__ float shFilters[B_Y*numColors][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
	__shared__ float shImages[B_Y*numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
	const int imgPixels = imgSizeY * imgSizeX;
	const int filterPixels = filterSize * filterSize;

	const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = blockIdx.y % blocksPerModule;

	const int tidx = threadIdx.y * B_X + threadIdx.x;

	const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
	const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

	const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
	images += myImgIdx;
	filters += filtersPerThread * B_Y * blockFilterIdx
		+ shFilterLoadY * numFilters + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numColors * filterPixels * numFilters;
	}

	targets += moduleIdx * numImages
		+ (blockFilterIdx * B_Y * filtersPerThread + threadIdx.y) * numImages * numModulesY * numModulesX
		+ myImgIdx;


	float prod[filtersPerThread][imgsPerThread];
#pragma unroll
	for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}

	for (int p = 0; p < filterPixels; p += B_Y) {
		/*
		* Load B_Y pixels from B_Y*filtersPerThread filters
		*/
		if (shFilterLoadY < B_Y) {
#pragma unroll
			for (int p2 = 0; p2 < B_Y; p2 += B_X / filtersPerThread) {
				if (p + p2 + shFilterLoadY < filterPixels) {
#pragma unroll
					for (int c = 0; c < numColors; c++) {
						shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[(c * filterPixels + p + p2) * numFilters];
					}
				}
				else {
#pragma unroll
					for (int c = 0; c < numColors; c++) {
						shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
					}
				}
			}
		}

		/*
		* Load B_Y pixels from B_X*imgsPerThread images
		*/
		const int pixIdx = p + threadIdx.y;
		if (pixIdx < filterPixels) {
			const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
			const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
			if (y >= 0 && y< imgSizeY && x >= 0 && x < imgSizeX) {
#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
					if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
#pragma unroll
						for (int c = 0; c < numColors; c++) {
							shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSizeX + x) + i * B_X];
						}
					}
					else {
#pragma unroll
						for (int c = 0; c < numColors; c++) {
							shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
						}
					}
				}
			}
			else { // Padding
#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int c = 0; c < numColors; c++) {
						shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
					}
				}
			}
		}
		__syncthreads();
#pragma unroll
		for (int i = 0; i < B_Y*numColors; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
				for (int g = 0; g < imgsPerThread; g++) {
					prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
				}
			}

		}
		__syncthreads();
	}

	if (scale) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
				}
			}
		}
	}
	else {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
				}
			}
		}
	}
}

/*
* Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
* threadIdx.x determines image
* threadIdx.y determines filter
*
* blockIdx.x determines image batch of B_X * imgsPerThread
* blockIdx.y determines filter batch of B_Y * filtersPerThread
*
* images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
* filters:     (numFilterColors, filterPixels, numFilters) if conv
*              (numModules, numFilterColors, filterPixels, numFilters) otherwise
*
* targets:     (numFilters, numModulesY, numModulesX, numImages)
*
* B_Y one of 4, 8, 16
* B_X one of 16, 32
* imgsPerThread one of 1, 2, 4
* filtersPerThread one of 1, 2, 4, 8
* colorCache: how many colors to put into shmem
*
* numFilters should be divisible by B_Y * filtersPerThread
* numImages be divisible by B_X * imgsPerThread
* numFilterColors should be divisible by colorCache.
* numImgColors must be even.
* numFilters must be divisible by numGroups.
*
* The imgSize here is the size of the actual image without the padding.
*
*/
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
	const int numImages, const int numFilters,
	const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
	const int moduleStride,
	const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
	const int numGroups,
	const float scaleTargets, const float scaleOutputs,
	const bool conv) {
	__shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
	__shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
	const int imgPixels = imgSizeY * imgSizeX;
	const int filterPixels = filterSize * filterSize;
	const int numFilterColors = numImgColors / numGroups;
	const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
	const int numFiltersPerGroup = numFilters / numGroups;
	const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

	const int numModules = numModulesX * numModulesY;
	const int blockColorIdx = numFilterColors * blockGroupIdx;

	const int tidx = threadIdx.y * B_X + threadIdx.x;

	const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

	const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

	images += blockColorIdx * imgPixels * imgStride + myImgIdx;
	filters += blockFilterIdx
		+ shFilterLoadY * numFilters + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numFilterColors * filterPixels * numFilters;
	}

	targets += moduleIdx * numImages
		+ (blockFilterIdx + threadIdx.y) * numImages * numModules
		+ myImgIdx;

	float prod[filtersPerThread][imgsPerThread];
#pragma unroll
	for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}
	//    __shared__ int imgPos[]
	for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
		for (int p = 0; p < filterPixels; p += B_Y) {
			/*
			* Load B_Y pixels from B_Y*filtersPerThread filters
			*/
			if (shFilterLoadY < B_Y) {
#pragma unroll
				for (int p2 = 0; p2 < B_Y; p2 += B_X / filtersPerThread) {
					if (p + p2 + shFilterLoadY < filterPixels) {
#pragma unroll
						for (int c = 0; c < colorCache; c++) {
							shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc + c) * filterPixels + p + p2) * numFilters];
						}
					}
					else {
#pragma unroll
						for (int c = 0; c < colorCache; c++) {
							shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
						}
					}
				}
			}

			/*
			* Load B_Y pixels from B_X*imgsPerThread images
			*/
			const int pixIdx = p + threadIdx.y;
			if (pixIdx < filterPixels) {
				const int x = imgLoadModPosX + pixIdx % filterSize;
				const int y = imgLoadModPosY + pixIdx / filterSize;
				if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
					float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
						if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
#pragma unroll
							for (int c = 0; c < colorCache; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
							}
						}
						else {
#pragma unroll
							for (int c = 0; c < colorCache; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
							}
						}
					}
				}
				else { // Padding
#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
						for (int c = 0; c < colorCache; c++) {
							shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
						}
					}
				}
			}
			__syncthreads();
#pragma unroll
			for (int i = 0; i < B_Y*colorCache; i++) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
					for (int g = 0; g < imgsPerThread; g++) {
						prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
					}
				}

			}
			__syncthreads();
		}
	}

	if (scale) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
				}
			}
		}
	}
	else {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
				}
			}
		}
	}
}

#endif /* NVMATRIX_KERNEL_H_ */
