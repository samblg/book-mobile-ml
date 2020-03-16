#pragma once

#include <string>
#include "common/libutil.h"

namespace authen {
namespace blas {

class Backend {
public:
    static Backend& GetInstance();
    bool isIntialized() const;

    bool loadLibrary();
    void freeLibrary();
    const std::string& type() const;

    ~Backend();

private:
    Backend();

    LibraryHandle _library;
    bool _loadedSymbols;
    std::string _type;
};

}
}
