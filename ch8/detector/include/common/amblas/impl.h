#pragma once

#ifdef WIN32
#ifdef AMBLAS_EXPORTS
#define AMBLAS_API __declspec(dllexport)
#else
#define AMBLAS_API __declspec(dllimport)
#endif
#else // WIN32
#define AMBLAS_API
#endif // WIN32

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    typedef enum AMBLAS_ORDER {
        AmBlasRowMajor,
        AmBlasColMajor
    } AMBLAS_ORDER;

    typedef enum AMBLAS_TRANSPOSE {
        AmBlasNoTrans,
        AmBlasTrans,
        AmBlasConjTrans,
        AmBlasConjNoTrans
    } AMBLAS_TRANSPOSE;

    typedef int AM_INT;

    AMBLAS_API int amblas_init();
    AMBLAS_API const char* amblas_backend_type();

    AMBLAS_API void amblas_set_num_threads(int num_threads);
    AMBLAS_API int amblas_get_num_threads(void);
    AMBLAS_API int amblas_get_num_procs(void);
    AMBLAS_API char* amblas_get_config(void);
    AMBLAS_API char* amblas_get_corename(void);
    AMBLAS_API int amblas_get_parallel(void);

    AMBLAS_API void amblas_sgemm(const AMBLAS_ORDER Order, const  AMBLAS_TRANSPOSE TransA,
        const  AMBLAS_TRANSPOSE TransB, const AM_INT M, const AM_INT N,
        const AM_INT K, const float alpha, const float *A,
        const AM_INT lda, const float *B, const AM_INT ldb,
        const float beta, float *C, const AM_INT ldc);

    AMBLAS_API void amblas_dgemm(const AMBLAS_ORDER Order, const AMBLAS_TRANSPOSE TransA,
        const AMBLAS_TRANSPOSE TransB, const AM_INT M, const AM_INT N, const AM_INT K,
        const double alpha, const double* A, const AM_INT lda, const double* B, const AM_INT ldb, const double beta,
        double* C, const AM_INT ldc);

    AMBLAS_API void amblas_sgemv(const AMBLAS_ORDER order, const AMBLAS_TRANSPOSE trans, const AM_INT m, const AM_INT n,
        const float alpha, const float  *a, const AM_INT lda, const float  *x, const AM_INT incx, const float beta, float  *y, const AM_INT incy);
    AMBLAS_API void amblas_dgemv(const enum AMBLAS_ORDER order, const enum AMBLAS_TRANSPOSE trans, const AM_INT m, const AM_INT n,
        const double alpha, const double  *a, const AM_INT lda, const double  *x, const AM_INT incx, const double beta, double  *y, const AM_INT incy);

    AMBLAS_API void amblas_saxpy(const AM_INT n, const float alpha, const float *x, const AM_INT incx, float *y, const AM_INT incy);
    AMBLAS_API void amblas_daxpy(const AM_INT n, const double alpha, const double *x, const AM_INT incx, double *y, const AM_INT incy);
    
    AMBLAS_API void amblas_sscal(const AM_INT N, const float alpha, float *X, const AM_INT incX);
    AMBLAS_API void amblas_dscal(const AM_INT N, const double alpha, double *X, const AM_INT incX);
    
    AMBLAS_API float  amblas_sdot(const AM_INT n, const float  *x, const AM_INT incx, const float  *y, const AM_INT incy);
    AMBLAS_API double amblas_ddot(const AM_INT n, const double *x, const AM_INT incx, const double *y, const AM_INT incy);

    AMBLAS_API float  amblas_sasum(const AM_INT n, const float  *x, const AM_INT incx);
    AMBLAS_API double amblas_dasum(const AM_INT n, const double *x, const AM_INT incx);

    AMBLAS_API void amblas_scopy(const AM_INT n, const float *x, const AM_INT incx, float *y, const AM_INT incy);
    AMBLAS_API void amblas_dcopy(const AM_INT n, const double *x, const AM_INT incx, double *y, const AM_INT incy);

    AMBLAS_API void amblas_saxpby(const int N, const float alpha, const float* X,
        const int incX, const float beta, float* Y,
        const int incY);
    AMBLAS_API void amblas_daxpby(const int N, const double alpha, const double* X,
        const int incX, const double beta, double* Y,
        const int incY);

#ifdef __cplusplus
}
#endif // __cplusplus
