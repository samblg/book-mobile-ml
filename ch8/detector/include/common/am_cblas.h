#pragma once

#define CBLAS_ORDER AMBLAS_ORDER
#define CBLAS_TRANSPOSE AMBLAS_TRANSPOSE

#define CblasRowMajor AmBlasRowMajor
#define CblasColMajor AmBlasColMajor

#define CblasNoTrans AmBlasNoTrans
#define CblasTrans AmBlasTrans
#define CblasConjTrans AmBlasConjTrans
#define CblasConjNoTrans AmBlasConjNoTrans

#define cblas_sgemm amblas_sgemm
#define cblas_dgemm amblas_dgemm
#define cblas_sgemv amblas_sgemv
#define cblas_dgemv amblas_dgemv
#define cblas_saxpy amblas_saxpy
#define cblas_daxpy amblas_daxpy
#define cblas_sscal amblas_sscal
#define cblas_dscal amblas_dscal
#define cblas_sdot amblas_sdot
#define cblas_ddot amblas_ddot
#define cblas_sasum amblas_sasum
#define cblas_dasum amblas_dasum
#define cblas_scopy amblas_scopy
#define cblas_dcopy amblas_dcopy
#define cblas_saxpby amblas_saxpby
#define cblas_daxpby amblas_daxpby

#include "amblas/impl.h"
