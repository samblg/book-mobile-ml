#pragma once

#ifdef WIN32
#include <Windows.h>
#else //WIN32
#include <pthread.h>
#endif //WIN32

#include "Thread.h"
#include <iostream>

namespace dataee {
	namespace thread {
#ifdef WIN32
		typedef DWORD ThreadLocalKey;
		typedef PVOID ThreadLocalValue;
#else //WIN32
        typedef pthread_key_t ThreadLocalKey;
        typedef void* ThreadLocalValue;
#endif //WIN32

		ThreadLocalKey CreateThreadLocalKey();
		void FreeThreadLocalKey(ThreadLocalKey key);
		ThreadLocalValue GetThreadLocalValue(ThreadLocalKey key);
		void SetThreadLocalValue(ThreadLocalKey key, ThreadLocalValue value);

		template <class Data>
		class ThreadLocalPointer {
		public:
			ThreadLocalPointer() {
				_key = CreateThreadLocalKey();
				ThreadLocalKey key = _key;
				ThreadManager::GetInstance().registerCleaner([key]() -> void {
					Data* realPointer = reinterpret_cast<Data*>(GetThreadLocalValue(key));
					if (realPointer) {
						delete realPointer;
					}

					SetThreadLocalValue(key, nullptr);
				});
			}

			~ThreadLocalPointer() {
				FreeThreadLocalKey(_key);
			}

			void reset(Data* data) {
				release();
				assign(data);
			}

			void release() {
				Data* realPointer = get();
				if (realPointer) {
					delete realPointer;
				}

				assign(nullptr);
			}

			Data* get() {
				return reinterpret_cast<Data*>(GetThreadLocalValue(_key));
			}

			const Data* get() const {
				return const_cast<const Data*>(reinterpret_cast<Data*>(GetThreadLocalValue(_key)));
			}

			Data* operator->() {
				return get();
			}

			const Data* operator->() const {
				return get();
			}

			Data& operator*() {
				return *get();
			}

			const Data& operator*() const {
				return *get();
			}

		private:
			void assign(Data* data) {
				SetThreadLocalValue(_key, data);
			}

			ThreadLocalKey _key;
		};
	}
}
