#pragma once

#include "resunpack/platform.h"

#if defined(__OS_WINDOWS__)
#ifdef RESUNPACK_EXPORTS
#define RESUNPACK_API __declspec(dllexport)
#else
#define RESUNPACK_API __declspec(dllimport)
#endif
#else
#define RESUNPACK_API
#endif
