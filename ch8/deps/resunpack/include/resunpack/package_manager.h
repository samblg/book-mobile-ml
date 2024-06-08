#pragma once

#include "resunpack/api.h"

#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace authen {
namespace resunpack {

class Package;

class RESUNPACK_API PackageManager {
public:
    static PackageManager& GetInstance();

    PackageManager() {}

    Package& openPackage(const std::string& packagePath);
    Package& openPackage(const std::string& packagePath, const std::string& md5);
    void closePackage(Package& package);

    Package* openPackagePointer(const std::string& packagePath);
    Package* openPackagePointer(const std::string& packagePath, const std::string& md5);
    void closePackagePointer(Package** package);

private:
    ~PackageManager() {}

private:
    std::mutex _mutex;
    std::map<std::string, Package*> _pathToPackages;
    std::map<Package*, std::string> _packageToPaths;
};

}
}
