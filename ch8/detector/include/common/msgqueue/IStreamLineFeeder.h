#pragma once

#include "common/msgqueue/BaseFeeder.h"

#include <istream>
#include <string>

namespace authen {
namespace message {

class IStreamLineFeeder : public BaseFeeder<std::string> {
public:
    IStreamLineFeeder(std::istream& stream) : _stream(stream) {
    }

    virtual bool nextMessage(std::string& message) override {
        std::string result;

        std::getline(_stream, result);
        if ( !_stream ) {
            return false;
        }

        message = result;
        return true;
    }

private:
    std::istream& _stream;
};

}
}
