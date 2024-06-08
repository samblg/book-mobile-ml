#pragma once

#include "resunpack/externc.h"
#include "resunpack/api.h"

typedef void* ARUPackage;
typedef void* ARUResourceStream;

BEGIN_EXTERN_C

ARUPackage RESUNPACK_API ARULoadPackage(const char* fileName);
ARUPackage RESUNPACK_API ARULoadPackageWithMd5(const char* fileName, const char* md5Hex);
void RESUNPACK_API ARUDestroyPackage(ARUPackage packageHandle);

ARUResourceStream RESUNPACK_API ARUOpenResource(ARUPackage packageHandle, const char* fileName);
void RESUNPACK_API ARUCloseResource(ARUResourceStream resourceStreamHandle);

END_EXTERN_C
