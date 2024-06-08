#include "mtimes.h"

#define INFINITY1 100000000
#define MEASURETIME 1
#define COLORHOG 1
#define fastMode 1

#define USEDOUBLE 0

#if USEDOUBLE
    typedef double sType;
#else
    typedef float sType;
#endif

// Only need to define feature dimension
#define _biasNum 46
#define _binSizeRoot 31
#define _binSizePart 10

#define addbiasnum 3.5


