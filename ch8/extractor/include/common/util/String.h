/**
 * licensed to the apache software foundation (asf) under one
 * or more contributor license agreements.  see the notice file
 * distributed with this work for additional information
 * regarding copyright ownership.  the asf licenses this file
 * to you under the apache license, version 2.0 (the
 * "license"); you may not use this file except in compliance
 * with the license.  you may obtain a copy of the license at
 *
 * http://www.apache.org/licenses/license-2.0
 *
 * unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "as is" basis,
 * without warranties or conditions of any kind, either express or implied.
 * see the license for the specific language governing permissions and
 * limitations under the license.
 */

#pragma once

#include "common/jsonbox/JsonBox.h"

#include <ostream>
#include <vector>
#include <string>
#include <cstdint>

namespace authen {
namespace core {
namespace util {

using jsonbox::Value;

class String {
public:
    static const char JOIN_STRING_DEFAULT_SPLITTER = ' ';

    static std::vector<std::string> Split(const std::string& value, char seperator);
    static std::string Trim(const std::string& value);
    static std::string Random(const std::string& candidate, int32_t length);
    static std::string Join(const std::vector<std::string>& words, char splitter = JOIN_STRING_DEFAULT_SPLITTER);
    static std::string Number(int32_t value);

    String(const std::string& s = "") : _argIndex(0), _data(s) {
    }

    const std::string& toStdString() const {
        return _data;
    }

    const char* toCString() const {
        return _data.c_str();
    }

    String& arg(int32_t number);
    String& arg(std::string str);
    String& format(std::initializer_list<Value> values);

private:
    int _argIndex;
    std::string _data;
};

}
}
}

std::ostream& operator<<(std::ostream& os, const authen::core::util::String& str);
