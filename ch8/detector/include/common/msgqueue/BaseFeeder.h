#pragma once

namespace authen {
namespace message {

template <class Message>
class BaseFeeder {
public:
    virtual bool nextMessage(Message& message) = 0;
};

}
}
