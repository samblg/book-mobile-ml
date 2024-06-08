#pragma once

#include <string>
#include <cstdint>

namespace authen {
namespace resunpack {

class Resource {
public:
    Resource() {}
    Resource(const std::string& fileName, int32_t offset, int32_t size) :
        _fileName(fileName), _offset(offset), _size(size) {}

    std::string fileName() const;
    void setFileName(const std::string &fileName);

    int32_t offset() const;
    void setOffset(const int32_t &offset);

    int32_t size() const;
    void setSize(const int32_t &size);

private:
    std::string _fileName;
    int32_t _offset;
    int32_t _size;
};

}
}
