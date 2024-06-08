#pragma once

#include "resunpack/platform.h"
//#define __COMPILER_MSVC__

#ifdef __COMPILER_MSVC__

#if _MSC_VER <= 1800
#define STD_NO_EXCEPT
#else // _MSC_VER
#define STD_NO_EXCEPT noexcept
#endif // _MSC_VER

#else // __COMPILER_MSVC__

#define STD_NO_EXCEPT noexcept

#endif // __COMPILER_MSVC__
