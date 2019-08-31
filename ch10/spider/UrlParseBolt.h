#pragma once

#include "hurricane/bolt/IBolt.h"

#include <string>
#include <cstdint>

class UrlParseBolt : public hurricane::bolt::IBolt {
public:
    virtual hurricane::bolt::IBolt* Clone() override {
        return new UrlParseBolt(*this);
    }
    virtual void Prepare(std::shared_ptr<hurricane::collector::OutputCollector> outputCollector) override;
    virtual void Cleanup() override;
    virtual std::vector<std::string> DeclareFields() override;
    virtual void Execute(const hurricane::base::Tuple& tuple) override;

private:
    void PerConnect(const std::string& url);
    void PutImageToSet(
            std::vector<std::string>& photoUrls,
            std::vector<std::string>& comUrls)
    void StoreImage(const std::string& imageUrl);

    std::shared_ptr<hurricane::collector::OutputCollector> _outputCollector;
    int _socketFd;
};
