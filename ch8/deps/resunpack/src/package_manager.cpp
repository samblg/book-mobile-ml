#include "resunpack/package_manager.h"
#include "resunpack/package.h"

namespace authen {
namespace resunpack {

PackageManager& PackageManager::GetInstance()
{
    static PackageManager packageManager;

    return packageManager;
}

Package& PackageManager::openPackage(const std::string& packagePath)
{
    Package* package = openPackagePointer(packagePath);
    if ( !package ) {
        throw ResunpackException("Package file not found: " + packagePath);
    }

    return *package;
}

Package& PackageManager::openPackage(const std::string& packagePath, const std::string& md5)
{
    Package* package = openPackagePointer(packagePath, md5);
    if ( !package ) {
        throw ResunpackException("Open package file error: " + packagePath);
    }

    return *package;
}

void PackageManager::closePackage(Package& package)
{
    Package* p = &package;
    closePackagePointer(&p);
}

Package* PackageManager::openPackagePointer(const std::string& packagePath)
{
    std::unique_lock<std::mutex> locker(_mutex);

    auto packageIt = _pathToPackages.find(packagePath);
    if ( packageIt != _pathToPackages.end() ) {
        Package* package = packageIt->second;
        package->retain();

        return package;
    }

    Package* package = new Package;

    package->load(packagePath);
    package->retain();
    _pathToPackages.insert({packagePath, package});
    _packageToPaths.insert({package, packagePath});

    return package;
}

Package* PackageManager::openPackagePointer(const std::string& packagePath, const std::string& md5)
{
    std::unique_lock<std::mutex> locker(_mutex);

    auto packageIt = _pathToPackages.find(packagePath);
    if ( packageIt != _pathToPackages.end() ) {
        Package* package = packageIt->second;
        package->retain();

        if ( !package->checkMd5(packagePath, md5) ) {
            return nullptr;
        }

        return package;
    }

    Package* package = new Package;
    if ( !package->load(packagePath, md5) ) {
        return nullptr;
    }

    package->retain();
    _pathToPackages.insert({packagePath, package});
    _packageToPaths.insert({package, packagePath});

    return package;
}

void PackageManager::closePackagePointer(Package** package)
{
   std::unique_lock<std::mutex> locker(_mutex);

   if ( !package ) {
       return;
   }

   if ( !(*package) ) {
       return;
   }

   (*package)->release();
   if ( !(*package)->refCount() ) {
       auto packageToPathIt = _packageToPaths.find(*package);
       if ( packageToPathIt != _packageToPaths.end() ) {
           std::string packagePath = packageToPathIt->second;
           _packageToPaths.erase(packageToPathIt);
           _pathToPackages.erase(packagePath);
       }

       delete *package;
       *package = nullptr;
   }
}

}
}
