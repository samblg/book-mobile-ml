#include "types.h"
#include "net.h"

namespace learnml {

class ExtractNet
{
public:
    ExtractNet();
    ExtractNet(const char* modelPath);

    int ExtractFeature(unsigned char* imageData, int width, int height, int pitch, int bpp,
        MlLandmarkResult* objectLandMarkResults, unsigned char* feature);

private:
    Net* extractNet;
};

}
