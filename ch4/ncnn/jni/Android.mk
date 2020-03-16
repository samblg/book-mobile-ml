LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE    := ncnn

LOCAL_C_INCLUDES += ../include
LOCAL_C_INCLUDES += ../include/layer
LOCAL_C_INCLUDES += ../include/layer/arm
LOCAL_CFLAGS = -fpic -Wno-psabi -funwind-tables -finline-limit=64 -fsigned-char -no-canonical-prefixes -fdata-sections -ffunction-sections -Wa,--noexecstack  -fopenmp -fno-rtti -fno-exceptions -fomit-frame-pointer -fno-strict-aliasing -O3 -DNDEBUG -Wall -Wextra -fPIC -Ofast -ffast-math -fvisibility=hidden -fvisibility-inlines-hidden
LOCAL_CXXFLAGS += -std=c++11

LOCAL_SRC_FILES := \
    ../src/blob.cpp \
    ../src/box_util.cpp \
    ../src/cpu.cpp \
    ../src/facebox.cpp \
    ../src/layer.cpp \
    ../src/mat.cpp \
    ../src/mat_pixel.cpp \
    ../src/net.cpp \
    ../src/opencv.cpp \
    ../src/common/io/Serialize.cpp \
    ../src/layer/absval.cpp \
    ../src/layer/axpy.cpp \
    ../src/layer/argmax.cpp \
    ../src/layer/batchnorm.cpp \
    ../src/layer/bias.cpp \
    ../src/layer/bnll.cpp \
    ../src/layer/concat.cpp \
    ../src/layer/convolution.cpp \
    ../src/layer/convolutiondepthwise.cpp \
    ../src/layer/crop.cpp \
    ../src/layer/deconvolution.cpp \
    ../src/layer/detectionout.cpp \
    ../src/layer/dropout.cpp \
    ../src/layer/eltwise.cpp \
    ../src/layer/elu.cpp \
    ../src/layer/embed.cpp \
    ../src/layer/exp.cpp \
    ../src/layer/flatten.cpp \
    ../src/layer/innerproduct.cpp \
    ../src/layer/input.cpp \
    ../src/layer/log.cpp \
    ../src/layer/lrn.cpp \
    ../src/layer/lstm.cpp \
    ../src/layer/memorydata.cpp \
    ../src/layer/mvn.cpp \
    ../src/layer/normalize.cpp \
    ../src/layer/padchannel.cpp \
    ../src/layer/permute.cpp \
    ../src/layer/pooling.cpp \
    ../src/layer/power.cpp \
    ../src/layer/prelu.cpp \
    ../src/layer/priorbox.cpp \
    ../src/layer/proposal.cpp \
    ../src/layer/reduction.cpp \
    ../src/layer/relu.cpp \
    ../src/layer/reshape.cpp \
    ../src/layer/rnn.cpp \
    ../src/layer/roipooling.cpp \
    ../src/layer/scale.cpp \
    ../src/layer/sigmoid.cpp \
    ../src/layer/slice.cpp \
    ../src/layer/softmax.cpp \
    ../src/layer/split.cpp \
    ../src/layer/spp.cpp \
    ../src/layer/tanh.cpp \
    ../src/layer/threshold.cpp \
    ../src/layer/tile.cpp \
    ../src/layer/arm/absval_arm.cpp \
    ../src/layer/arm/batchnorm_arm.cpp \
    ../src/layer/arm/bias_arm.cpp \
    ../src/layer/arm/convolution_arm.cpp \
    ../src/layer/arm/convolutiondepthwise_arm.cpp \
    ../src/layer/arm/eltwise_arm.cpp \
    ../src/layer/arm/innerproduct_arm.cpp \
    ../src/layer/arm/lrn_arm.cpp \
    ../src/layer/arm/pooling_arm.cpp \
    ../src/layer/arm/prelu_arm.cpp \
    ../src/layer/arm/relu_arm.cpp \
    ../src/layer/arm/scale_arm.cpp \
    ../src/layer/arm/sigmoid_arm.cpp \
    ../src/layer/arm/slice_arm.cpp \
    ../src/layer/arm/softmax_arm.cpp

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
LOCAL_CFLAGS +=  -mfloat-abi=softfp -mfpu=neon -mthumb
endif

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
LOCAL_CFLAGS +=  -march=armv8-a
endif

include $(BUILD_STATIC_LIBRARY)
