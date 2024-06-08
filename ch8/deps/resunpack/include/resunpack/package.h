#pragma once

#include "resunpack/resource.h"
#include "resunpack/std_compatible.h"
#include "resunpack/api.h"

#include <string>
#include <fstream>
#include <map>
#include <cstdint>
#include <mutex>

namespace authen {
namespace resunpack {

class RESUNPACK_API ResunpackException : public std::exception {
public:
    ResunpackException(const std::string& message) :
            _message(message){
    }

    const char* what() const STD_NO_EXCEPT {
        return _message.c_str();
    }

private:
    std::string _message;
};

class RESUNPACK_API Package {
public:
    Package();
    ~Package();

    void load(const std::string& fileName);
    bool load(const std::string& fileName, const std::string& md5Hex);
    std::istringstream* openResourcePointer(const std::string& resourceName);
    void closeResourcePointer(std::istringstream **resourceStream);

    std::istringstream& openResource(const std::string& filePath);
    void closeResource(std::istream& modelStream);

    int refCount() const {
        return _refCount;
    }

    void retain() {
        ++ _refCount;
    }

    void release() {
        -- _refCount;
    }

    bool checkMd5(const std::string& fileName, const std::string& md5Hex);

private:
    int32_t readInt32(std::ifstream& packageStream);
    std::string readString(std::ifstream& packageStream);

private:
    std::mutex _mutex;
    std::string _fileName;
    std::string _magicNumber;
    std::string _name;
    std::map<std::string, Resource> _resources;
    int _refCount;
};

}
}
