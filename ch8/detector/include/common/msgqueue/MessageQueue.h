#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace authen {
namespace message {

template <class Message>
class MessageQueue {
public:
    MessageQueue(int maxSize = 0) : _maxSize(maxSize) {
    }

    bool enqueue(const Message& message) {
        std::unique_lock<std::mutex> locker(_mutex);

        while ( _maxSize && _queue.size() >= _maxSize ) {
            _fullCv.wait(locker);
        }

        _queue.push(message);
        _emptyCv.notify_one();

        return true;
    }

    bool enqueue(const Message& message, int milliseconds) {
        std::unique_lock<std::mutex> locker(_mutex);

        if ( _maxSize && _queue.size() >= _maxSize ) {
            std::cv_status status =_fullCv.wait_for(locker, std::chrono::milliseconds(milliseconds));
            if ( status == std::cv_status::timeout ) {
                return false;
            }
        }

        _queue.push(message);
        _emptyCv.notify_one();

        return true;
    }

    bool dequeue(Message& message) {
        std::unique_lock<std::mutex> locker(_mutex);

        while ( !_queue.size() ) {
            _emptyCv.wait(locker);
        }

        message = _queue.front();
        _queue.pop();

        if ( _maxSize ) {
            _fullCv.notify_one();
        }

        return true;
    }

    bool dequeue(Message& message, int milliseconds) {
        std::unique_lock<std::mutex> locker(_mutex);

        if ( !_queue.size() ) {
            std::cv_status status =_emptyCv.wait_for(locker, std::chrono::milliseconds(milliseconds));
            if ( status == std::cv_status::timeout ) {
                return false;
            }
        }

        message = _queue.front();
        _queue.pop();

        if ( _maxSize ) {
            _fullCv.notify_one();
        }

        return true;
    }

    bool peek(Message& message) const {
        std::unique_lock<std::mutex> locker(_mutex);

        if ( !_queue.size() ) {
            return false;
        }

        message = _queue.front();
        return true;
    }

    size_t size() const {
        std::unique_lock<std::mutex> locker(_mutex);

        return _queue.size();
    }

private:
    std::queue<Message> _queue;
    size_t _maxSize;
    mutable std::mutex _mutex;
    std::condition_variable _emptyCv;
    std::condition_variable _fullCv;
};

}
}
