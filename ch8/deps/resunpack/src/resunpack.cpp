#include "resunpack/resunpack.h"
#include "resunpack/package_manager.h"
#include "resunpack/package.h"
#include <sstream>

using authen::resunpack::Package;
using authen::resunpack::PackageManager;

ARUPackage ARULoadPackage(const char *fileName)
{
    PackageManager& manager = PackageManager::GetInstance();
    Package* package = manager.openPackagePointer(fileName);

    return reinterpret_cast<ARUPackage>(package);
}

void ARUDestroyPackage(ARUPackage packageHandle)
{
    if ( !packageHandle ) {
        return;
    }

    PackageManager& manager = PackageManager::GetInstance();
    Package* package = reinterpret_cast<Package*>(packageHandle);
    manager.closePackagePointer(&package);
}

ARUResourceStream ARUOpenResource(ARUPackage packageHandle, const char *fileName)
{
    if ( !packageHandle ) {
        return nullptr;
    }

    Package* package = reinterpret_cast<Package*>(packageHandle);
    std::istringstream* resourceStream = package->openResourcePointer(fileName);

    return reinterpret_cast<ARUResourceStream>(resourceStream);
}

void ARUCloseResource(ARUResourceStream resourceStreamHandle)
{
    if ( !resourceStreamHandle ) {
        return;
    }

    std::istringstream* resourceStream = reinterpret_cast<std::istringstream*>(resourceStreamHandle);
    delete resourceStream;
}

ARUPackage ARULoadPackageWithMd5(const char* fileName, const char* md5Hex)
{
    Package* package = new Package;
    if ( !package->load(fileName, md5Hex) ) {
        delete package;
        return nullptr;
    }

    return reinterpret_cast<ARUPackage>(package);
}
