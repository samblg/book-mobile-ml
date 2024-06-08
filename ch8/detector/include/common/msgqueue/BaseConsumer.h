#pragma once

namespace authen {
namespace message {

template <class Message>
class BaseConsumer {
public:
    virtual void cleaup() = 0;
    virtual void execute(Message& message) = 0;
};

}
}
