#include "SpiderTaskSpout.h"
#include "hurricane/util/StringUtil.h"
#include <thread>
#include <chrono>
#include <redox.hpp>
#include <fstream>

const std::string REDIS_HOST="localhost";
const int32_t REDIS_PORT = 6379;

void DataReaderSpout::Prepare(std::shared_ptr<hurricane::collector::OutputCollector> outputCollector) {
    _outputCollector = outputCollector;
    _subscriber = std::make_shared<redox::Subscriber>(REDIS_HOST, REDIS_PORT);
}

void DataReaderSpout::Cleanup() {
    if (!_subscriber.get()) {
        return;
    }

    _subscriber->disconnect();
    delete _redox;
}

std::vector<std::string> DataReaderSpout::DeclareFields() {
    return { "record" };
}

void DataReaderSpout::NextTuple() {
    sub.subscribe("data", [](const string& topic, const string& fileName) {
        std::ifstream& inputFile(fileName.c_str());
        std::vector<double> record;

        while (inputFile) {
            double element;

            inputFile >> element;
            record.push_back(element);
        }

        _outputCollector->Emit({ record });
    });
}
