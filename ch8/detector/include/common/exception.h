#pragma once

#include <exception>
#include <string>

#ifndef NO_EXCEPT
#ifdef WIN32
#define NO_EXCEPT
#else
#define NO_EXCEPT noexcept
#endif // WIN32
#endif // NO_EXCEPT

namespace authen {
namespace core {

namespace util {
class String;
}

class Exception : public std::exception {
public:
    Exception(int code) : _code(code) {
    }

    Exception(const std::string& message) : _message(message) {
    }

    Exception(int code, const std::string& message) : _code(code), _message(message) {
    }

    Exception(const util::String& message);
    Exception(int code, const util::String& message);

    const char* what() const NO_EXCEPT override {
        return _message.c_str();
    }

    int code() const {
        return _code;
    }

    const std::string& message() const {
        return _message;
    }

private:
    std::string _message;
    int _code;
};

};
};
