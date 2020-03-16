#pragma once

#include <string>
#include <cstdint>
#include <cstdio>
#include <iostream>

namespace authen{
namespace io {

void Serialize(FILE* fp, const char* str, size_t size);
void Serialize(std::ostream& os, const char* str, size_t size);

template <class T>
void Serialize(FILE* fp, const T& value) {
    fwrite(&value, sizeof(T), 1, fp);
}

template <>
void Serialize(FILE* fp, const std::string& str);

template <class T>
void Serialize(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <>
void Serialize(std::ostream& os, const std::string& str);

template <class T>
void Deserialize(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <class T>
void Deserialize(std::istream& is, T* buffer, size_t size) {
    is.read(reinterpret_cast<char*>(buffer), sizeof(T) * size);
}

template <class T>
void Deserialize(FILE* fp, T& value) {
    fread(&value, sizeof(T), 1, fp);
}

template <class T>
void Deserialize(FILE* fp, T* buffer, size_t size) {
    fread(&buffer, sizeof(T), size, fp);
}

template <>
void Deserialize(std::istream& is, std::string& str);

template <>
void Deserialize(FILE* fp, std::string& str);

}
}
