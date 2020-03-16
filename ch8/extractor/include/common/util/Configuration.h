#pragma once

#include "common/jsonbox/JsonBox.h"
#include <string>
#include <istream>
#include <map>
#include <vector>
#include <memory>

namespace authen {
namespace core {
namespace util {

using jsonbox::Value;

class Configuration {
public:
    static Configuration CreateFromFile(const std::string& fileName);
    static Configuration CreateFromStream(std::istream& inputStream);

    Configuration();
    ~Configuration();

    void readFromFile(const std::string& fileName);
    void readFromStream(std::istream& inputStream);

    Value property(const std::string& key) const;
    void setProperty(const std::string& key, const Value& v);
    std::vector<std::string> keys() const;
    const std::shared_ptr<std::map<std::string, Value>>& properties() const;
    bool has(const std::string& key) const;

    bool booleanProperty(const std::string& key) const;
    int intProperty(const std::string& key) const;
    float floatProperty(const std::string& key) const;
    double doubleProperty(const std::string& key) const;
    std::string stringProperty(const std::string& key) const;

    bool booleanProperty(const std::string& key, bool defaultValue) const;
    int intProperty(const std::string& key, int defaultValue) const;
    float floatProperty(const std::string& key, float defaultValue) const;
    double doubleProperty(const std::string& key, double defaultValue) const;
    std::string stringProperty(const std::string& key, const std::string& defaultValue) const;

private:
    void parseObject(const jsonbox::Object& confObject, const std::string& prefix = "");

private:
    std::shared_ptr<std::map<std::string, Value>> _properties;
};

}
}
}
