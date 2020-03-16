#pragma once

#include <thread>
#include <memory>
#include <chrono>
#include "common/msgqueue/MessageQueue.h"
#include "common/msgqueue/BaseFeeder.h"

namespace authen {
namespace message {

template <class Message>
class FeederExecutor {
public:
    FeederExecutor(MessageQueue<Message>* queue, BaseFeeder<Message>* feeder) :
        _queue(queue), _feeder(feeder), _stopped(false){}

    void start() {
        _thread = std::thread(&FeederExecutor::threadMain, this);
        _thread.detach();
    }

    void join() {
        while ( !_stopped ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void threadMain() {
        Message message;

        while ( _feeder->nextMessage(message) ) {
            _queue->enqueue(message);
        }
        _stopped = true;
    }

private:
    std::thread _thread;
    MessageQueue<Message>* _queue;
    std::shared_ptr<BaseFeeder<Message>> _feeder;
    bool _stopped;
};

}
}
