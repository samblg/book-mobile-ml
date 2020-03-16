#pragma once

#include <string>
#include <istream>
#include <initializer_list>

namespace authen {
namespace core {
namespace util {

class Configuration;

class Environment {
public:
    static Environment& InitializeOnce(const Configuration& conf);

    void initWithConfiguration(const Configuration& conf);
    void checkLicense(const std::string modelDir, std::string product_code);
    void checkModelFiles(const std::string& modelDir, const std::initializer_list<std::string>& modelFileNames);

private:
    Environment() = delete;
    Environment(const Environment&) = delete;
    Environment(const Configuration& conf);

    void checkModelFile(const std::string& modelDir, const std::string& modelFileName);

    int _threadPoolSize;
};

}
}
}
