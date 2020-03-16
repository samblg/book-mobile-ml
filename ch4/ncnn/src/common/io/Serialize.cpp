#include "common/io/Serialize.h"

namespace authen{
namespace io {

template <>
void Serialize(FILE* fp, const std::string& str) {
    Serialize(fp, str.data(), str.size());
}

void Serialize(FILE* fp, const char* str, size_t size) {
    Serialize<uint32_t>(fp, size);
    fwrite(str, 1, size, fp);
}

template <>
void Serialize(std::ostream& os, const std::string& str) {
    Serialize(os, str.data(), str.size());
}

void Serialize(std::ostream& os, const char* str, size_t size) {
    Serialize<uint32_t>(os, size);
    os.write(str, size);
}

template <>
void Deserialize(std::istream& is, std::string& str) {
    
    uint32_t size = 0;
    Deserialize<uint32_t>(is, size);

    char* buffer = new char[size];
    Deserialize(is, buffer, size);

    str = std::string(buffer, size);

    delete[] buffer;
}

template <>
void Deserialize(FILE* fp, std::string& str) {

    uint32_t size = 0;
    Deserialize<uint32_t>(fp, size);

    char* buffer = new char[size];
    Deserialize(fp, buffer, size);

    str = std::string(buffer, size);

    delete[] buffer;
}

}
}
