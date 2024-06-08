#pragma once

#include "common/amblas/types.h"

namespace authen {
namespace blas {

extern void (*set_num_threads)(int num_threads);
extern int (*get_num_threads)(void);
extern int (*get_num_procs)(void);
extern char* (*get_config)(void);
extern char* (*get_corename)(void);
extern int (*get_parallel)(void);

extern void (*sgemm)(AM_CONST enum CBLAS_ORDER Order, AM_CONST enum CBLAS_TRANSPOSE TransA, AM_CONST enum CBLAS_TRANSPOSE TransB, AM_CONST blasint M, AM_CONST blasint N, AM_CONST blasint K,
          AM_CONST float alpha, AM_CONST float *A, AM_CONST blasint lda, AM_CONST float *B, AM_CONST blasint ldb, AM_CONST float beta, float *C, AM_CONST blasint ldc);
extern void (*dgemm)(AM_CONST enum CBLAS_ORDER Order, AM_CONST enum CBLAS_TRANSPOSE TransA, AM_CONST enum CBLAS_TRANSPOSE TransB, AM_CONST blasint M, AM_CONST blasint N, AM_CONST blasint K,
          AM_CONST double alpha, AM_CONST double *A, AM_CONST blasint lda, AM_CONST double *B, AM_CONST blasint ldb, AM_CONST double beta, double *C, AM_CONST blasint ldc);

extern void (*sgemv)(AM_CONST enum CBLAS_ORDER order,  AM_CONST enum CBLAS_TRANSPOSE trans,  AM_CONST blasint m, AM_CONST blasint n,
          AM_CONST float alpha, AM_CONST float  *a, AM_CONST blasint lda,  AM_CONST float  *x, AM_CONST blasint incx,  AM_CONST float beta,  float  *y, AM_CONST blasint incy);
extern void (*dgemv)(AM_CONST enum CBLAS_ORDER order,  AM_CONST enum CBLAS_TRANSPOSE trans,  AM_CONST blasint m, AM_CONST blasint n,
          AM_CONST double alpha, AM_CONST double  *a, AM_CONST blasint lda,  AM_CONST double  *x, AM_CONST blasint incx,  AM_CONST double beta,  double  *y, AM_CONST blasint incy);

extern void (*saxpy)(AM_CONST blasint n, AM_CONST float alpha, AM_CONST float *x, AM_CONST blasint incx, float *y, AM_CONST blasint incy);
extern void (*daxpy)(AM_CONST blasint n, AM_CONST double alpha, AM_CONST double *x, AM_CONST blasint incx, double *y, AM_CONST blasint incy);

extern void (*sscal)(AM_CONST blasint N, AM_CONST float alpha, float *X, AM_CONST blasint incX);
extern void (*dscal)(AM_CONST blasint N, AM_CONST double alpha, double *X, AM_CONST blasint incX);

extern float (*sdot)(AM_CONST blasint n, AM_CONST float  *x, AM_CONST blasint incx, AM_CONST float  *y, AM_CONST blasint incy);
extern double (*ddot)(AM_CONST blasint n, AM_CONST double *x, AM_CONST blasint incx, AM_CONST double *y, AM_CONST blasint incy);

extern float (*sasum)(AM_CONST blasint n, AM_CONST float  *x, AM_CONST blasint incx);
extern double (*dasum)(AM_CONST blasint n, AM_CONST double *x, AM_CONST blasint incx);

extern void (*scopy)(AM_CONST blasint n, AM_CONST float *x, AM_CONST blasint incx, float *y, AM_CONST blasint incy);
extern void (*dcopy)(AM_CONST blasint n, AM_CONST double *x, AM_CONST blasint incx, double *y, AM_CONST blasint incy);

extern void (*saxpby)(AM_CONST blasint n, AM_CONST float alpha, AM_CONST float *x, AM_CONST blasint incx,AM_CONST float beta, float *y, AM_CONST blasint incy);
extern void (*daxpby)(AM_CONST blasint n, AM_CONST double alpha, AM_CONST double *x, AM_CONST blasint incx,AM_CONST double beta, double *y, AM_CONST blasint incy);

}
}
