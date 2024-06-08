#include "resunpack/package.h"
#include "resunpack/md5.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <locale>

namespace authen {
namespace resunpack {

Package::Package() : _refCount(0)
{
}

Package::~Package()
{}

void Package::load(const std::string &fileName)
{
    std::unique_lock<std::mutex> locker(_mutex);

    std::ifstream packageFile(fileName.c_str(), std::ios_base::binary);
    
    _fileName = fileName;
    _magicNumber = readString(packageFile);
    _name = readString(packageFile);

    int32_t filesCount = readInt32(packageFile);
    for ( int32_t fileIndex = 0; fileIndex != filesCount; ++ fileIndex ) {
        std::string fileName = readString(packageFile);
        int32_t fileSize = readInt32(packageFile);

        int32_t fileOffset = static_cast<int32_t>(packageFile.tellg());
        packageFile.seekg(fileSize, std::ios_base::cur);

        _resources.insert({ fileName, Resource(fileName, fileOffset, fileSize)});
    }
}

bool Package::load(const std::string &fileName, const std::string& md5Hex)
{
    std::unique_lock<std::mutex> locker(_mutex);

    if ( !checkMd5(fileName, md5Hex) ) {
        return false;
    }

    std::ifstream packageFile(fileName.c_str(), std::ios_base::binary);

    _fileName = fileName;
    _magicNumber = readString(packageFile);
    _name = readString(packageFile);

    int32_t filesCount = readInt32(packageFile);
    for ( int32_t fileIndex = 0; fileIndex != filesCount; ++ fileIndex ) {
        std::string fileName = readString(packageFile);
        int32_t fileSize = readInt32(packageFile);

        int32_t fileOffset = static_cast<int32_t>(packageFile.tellg());
        packageFile.seekg(fileSize, std::ios_base::cur);

        _resources.insert({ fileName, Resource(fileName, fileOffset, fileSize)});
    }

    return true;
}

std::istringstream& Package::openResource(const std::string& filePath) {
    std::istringstream* resourceStream = openResourcePointer(filePath);
    if ( !resourceStream ) {
        throw ResunpackException(std::string("Cannot open resource ") + filePath);
    }

    return *resourceStream;
}

void Package::closeResource(std::istream& modelStream) {
    std::istringstream* stringStream = dynamic_cast<std::istringstream*>(&modelStream);

    closeResourcePointer(&stringStream);
}

std::istringstream* Package::openResourcePointer(const std::string &resourceName)
{
    std::unique_lock<std::mutex> locker(_mutex);

	std::ifstream packageFile(_fileName.c_str(), std::ios_base::binary);

    if ( _resources.find(resourceName) == _resources.end() ) {
        return nullptr;
    }

    Resource resource = _resources.at(resourceName);
    int32_t fileOffset = resource.offset();

    packageFile.seekg(fileOffset - sizeof(int32_t), std::ios_base::beg);
    std::string resourceData = readString(packageFile);

    std::istringstream* resourceStream = new std::istringstream(resourceData);
    return resourceStream;
}

void Package::closeResourcePointer(std::istringstream** resourceStream)
{
    std::unique_lock<std::mutex> locker(_mutex);

    if ( resourceStream && *resourceStream ) {
        delete *resourceStream;
        *resourceStream = nullptr;
    }
}

bool Package::checkMd5(const std::string& fileName, const std::string& md5Hex)
{
    std::ifstream packageFile(fileName.c_str(), std::ios_base::binary);

    Md5Context md5Context;
    md5Context.addData(packageFile);
	std::string realMd5Hex = md5Context.hexResult();
    if ( md5Hex != realMd5Hex ) {
        return false;
    }

    return true;
}

int32_t Package::readInt32(std::ifstream &packageStream)
{
    int32_t number = 0;
    packageStream.read(reinterpret_cast<char*>(&number), sizeof(int32_t));

    return number;
}

std::string Package::readString(std::ifstream &packageStream)
{
    int32_t stringSize = readInt32(packageStream);

    char* stringBuffer = new char[stringSize];
    packageStream.read(stringBuffer, stringSize);

    std::string newString(stringBuffer, stringSize);
    delete [] stringBuffer;

    return newString;
}

}
}
