#pragma once

#include <thread>
#include <memory>
#include "common/msgqueue/MessageQueue.h"
#include "common/msgqueue/BaseConsumer.h"

namespace authen {
namespace message {

template <class Message>
class ConsumerExecutor {
public:
    enum class StopStrategy {
        StopImmediately,
        StopWhenNoMessage
    };

    ConsumerExecutor(MessageQueue<Message>* queue, BaseConsumer<Message>* consumer) :
        _queue(queue), _consumer(consumer), _timeout(100), _needToStop(false), _stopped(false),
        _stopStrategy(StopStrategy::StopWhenNoMessage) {}

    void start() {
        _thread = std::thread(&ConsumerExecutor::threadMain, this);
        _thread.detach();
    }

    void join() {
        while ( !_stopped ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void stop() {
        _needToStop = true;
    }

    void threadMain() {
        Message message;

        while ( true ) {
            bool hasMessage = _queue->dequeue(message, _timeout);

            if ( _stopStrategy == StopStrategy::StopImmediately && _needToStop ) {
                break;
            }

            if ( !hasMessage ) {
                if ( _stopStrategy == StopStrategy::StopWhenNoMessage && _needToStop ) {
                    break;
                }

                continue;
            }

            _consumer->execute(message);
        }

        _consumer->cleaup();
        _stopped = true;
    }

private:
    std::thread _thread;
    MessageQueue<Message>* _queue;
    std::shared_ptr<BaseConsumer<Message>> _consumer;
    int _timeout;
    bool _needToStop;
    bool _stopped;
    StopStrategy _stopStrategy;
};

}
}
