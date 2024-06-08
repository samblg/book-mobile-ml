LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := resunpack

LOCAL_C_INCLUDES += ../include
LOCAL_CFLAGS += -O2 -std=c++11 -frtti -fexceptions -fvisibility=hidden -fPIC 
LOCAL_LDFLAGS += -lm 
LOCAL_SRC_FILES :=  ../src/md5.cpp \
                    ../src/package.cpp \
                    ../src/package_manager.cpp \
                    ../src/resource.cpp \
                    ../src/resunpack.cpp
                    
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
LOCAL_CFLAGS +=  -march=armv8-a
LOCAL_CPPFLAGS +=  -march=armv8-a
endif

include $(BUILD_STATIC_LIBRARY)