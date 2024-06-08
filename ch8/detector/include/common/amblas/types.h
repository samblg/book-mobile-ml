#pragma once

#define AM_CONST const

typedef enum CBLAS_ORDER {
    CblasRowMajor=101,
    CblasColMajor=102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans=111,
    CblasTrans=112,
    CblasConjTrans=113,
    CblasConjNoTrans=114
} CBLAS_TRANSPOSE;

typedef int blasint;
