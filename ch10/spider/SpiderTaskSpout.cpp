#include "SpiderTaskSpout.h"
#include "hurricane/util/StringUtil.h"
#include <thread>
#include <chrono>
#include <redox.hpp>

const std::string REDIS_HOST="localhost";
const int32_t REDIS_PORT = 6379;

void HelloWorldSpout::Prepare(std::shared_ptr<hurricane::collector::OutputCollector> outputCollector) {
    _outputCollector = outputCollector;
    _subscriber = std::make_shared<redox::Subscriber>(REDIS_HOST, REDIS_PORT);
}

void HelloWorldSpout::Cleanup() {
    if (!_subscriber.get()) {
        return;
    }

    _subscriber->disconnect();
    delete _redox;
}

std::vector<std::string> HelloWorldSpout::DeclareFields() {
    return { "url" };
}

void HelloWorldSpout::NextTuple() {
    sub.subscribe("spide", [](const string& topic, const string& msg) {
        _outputCollector->Emit({ msg });
    });
}
