#pragma once

#include "caffe/blob.hpp"
#include <ostream>
#include <iostream>

namespace caffe {

template <typename Dtype>
class BlobPersiter {
public:
    BlobPersiter() {}

    bool SaveToFile(const Blob<Dtype>* blob, const std::string& filePath);
    bool LoadFromFile(Blob<Dtype>* blob, const std::string& filePath);

    bool SaveToTextFile(const Blob<Dtype>* blob, const std::string& filePath);

private:
    template <typename Etype>
    void write(std::ostream& dataStream, Etype element) {
        char* dataBuffer = reinterpret_cast<char*>(&element);
        dataStream.write(dataBuffer, sizeof(Etype));
    }

    template <typename Etype>
    Etype read(std::istream& dataStream) {
        Etype element;
        char* dataBuffer = reinterpret_cast<char*>(&element);
        dataStream.read(dataBuffer, sizeof(Etype));

        return element;
    }
};

}
