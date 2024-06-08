#include "resunpack/resource.h"

namespace authen {
namespace resunpack {

std::string Resource::fileName() const
{
    return _fileName;
}

void Resource::setFileName(const std::string &fileName)
{
    _fileName = fileName;
}

int32_t Resource::offset() const
{
    return _offset;
}

void Resource::setOffset(const int32_t &offset)
{
    _offset = offset;
}

int32_t Resource::size() const
{
    return _size;
}

void Resource::setSize(const int32_t &size)
{
    _size = size;
}

}
}
