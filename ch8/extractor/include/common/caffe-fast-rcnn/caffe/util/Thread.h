#pragma once

#include <thread>
#include <map>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <vector>
#include <functional>

#ifdef Yield
#undef Yield
#endif

namespace dataee {
	namespace thread {
		typedef std::thread::id ThreadId;
		class Thread;

		class ThreadManager {
		public:
			typedef std::function<void()> Cleaner;

			static ThreadManager& GetInstance();

			Thread& getThread(ThreadId id) {
				std::unique_lock<std::mutex> locker;

				return *(_threads.at(id));
			}

			void addThread(ThreadId id, Thread& thread) {
				std::unique_lock<std::mutex> locker;

				_threads.insert({ id, &thread });
			}

			void removeThread(ThreadId id) {
				std::unique_lock<std::mutex> locker;

				_threads.erase(id);
			}

			void registerCleaner(Cleaner cleaner) {
				std::unique_lock<std::mutex> locker;

				_cleaners.push_back(cleaner);
			}

			void executeCleaners() {
				std::unique_lock<std::mutex> locker;

				for (Cleaner cleaner : _cleaners) {
					cleaner();
				}
			}

		private:
			std::map<ThreadId, Thread*> _threads;
			std::mutex _mutex;
			std::vector<Cleaner> _cleaners;
		};

		class Thread {
		public:
			typedef ThreadId Id;
			typedef std::thread::native_handle_type NativeHandle;
			typedef std::function<void()> Cleaner;
			typedef std::function<void()> ThreadEntrance;

			static Thread& CurrentThread();

			static Id CurrentThreadId() {
				return std::this_thread::get_id();
			}

			static void Yield() {
				std::this_thread::yield();
			}

			static void Sleep(int64_t seconds) {
				std::this_thread::sleep_for(std::chrono::seconds(seconds));
			}

			static void MilliSleep(int64_t milliSeconds) {
				std::this_thread::sleep_for(std::chrono::milliseconds(milliSeconds));
			}

			static void MicroSleep(int64_t microSeconds) {
				std::this_thread::sleep_for(std::chrono::microseconds(microSeconds));
			}

			void NanoSleep(int64_t nanoSeconds) {
				std::this_thread::sleep_for(std::chrono::nanoseconds(nanoSeconds));
			}

			template<class _Fn,
			class... _Args>
				explicit Thread(_Fn&& _Fx, _Args&&... _Ax) : _thread(
						std::bind(&Thread::threadMain, this, std::placeholders::_1),
                        std::forward<ThreadEntrance>(std::bind(
                                                         std::forward<_Fn>(_Fx),
                                                         std::forward<_Args>(_Ax))...)
						) {
				ThreadManager::GetInstance().addThread(getId(), *this);
			}

			void threadMain(ThreadEntrance target) {
				target();
				ThreadManager::GetInstance().executeCleaners();
			}

			~Thread() {
				ThreadManager::GetInstance().removeThread(getId());
			}

			bool joinable() const
			{
				return _thread.joinable();
			}

			void join() {
				_thread.join();
			}

			void detach() {
				_thread.detach();
			}

			Id get_id() const {
				return getId();
			}

			Id getId() const {
				return _thread.get_id();
			}

			static unsigned int hardware_concurrency()
			{
				return HardwareConcurrency();
			}

			static unsigned int HardwareConcurrency()
			{
				return std::thread::hardware_concurrency();
			}

			NativeHandle native_handle()
			{
				return nativeHandle();
			}

			NativeHandle nativeHandle()
			{
				return _thread.native_handle();
			}

		private:
			std::thread _thread;
		};
	}
}
