#include "model.h"

#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <string>
#include <unistd.h>

Model::Model(size_t size, int protect, int fd, size_t offset) :
        model_(nullptr),
        compilation_(nullptr),
        dimLength_(TENSOR_SIZE),
        modelDataFd_(fd),
        offset_(offset) {
    tensorSize_ = dimLength_;
    inputTensor1_.resize(tensorSize_);

    int32_t status = ANeuralNetworksMemory_createFromFd(size + offset, protect, fd, 0,
                                                        &memoryModel_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for trained weights");
        return;
    }

    inputTensor2Fd_ = ASharedMemory_create("input2", tensorSize_ * sizeof(float));
    outputTensorFd_ = ASharedMemory_create("output", tensorSize_ * sizeof(float));

    status = ANeuralNetworksMemory_createFromFd(tensorSize_ * sizeof(float),
                                                PROT_READ,
                                                inputTensor2Fd_, 0,
                                                &memoryInput2_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for Input2");
        return;
    }
    status = ANeuralNetworksMemory_createFromFd(tensorSize_ * sizeof(float),
                                                PROT_READ | PROT_WRITE,
                                                outputTensorFd_, 0,
                                                &memoryOutput_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksMemory_createFromFd failed for Output");
        return;
    }
}

bool Model::CreateCompiledModel() {
    int32_t status;

    status = ANeuralNetworksModel_create(&model_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_create failed");
        return false;
    }

    uint32_t dimensions[] = {dimLength_};
    ANeuralNetworksOperandType float32TensorType{
            .type = ANEURALNETWORKS_TENSOR_FLOAT32,
            .dimensionCount = sizeof(dimensions) / sizeof(dimensions[0]),
            .dimensions = dimensions,
            .scale = 0.0f,
            .zeroPoint = 0,
    };
    ANeuralNetworksOperandType scalarInt32Type{
            .type = ANEURALNETWORKS_INT32,
            .dimensionCount = 0,
            .dimensions = nullptr,
            .scale = 0.0f,
            .zeroPoint = 0,
    };

    uint32_t opIdx = 0;

    status = ANeuralNetworksModel_addOperand(model_, &scalarInt32Type);
    uint32_t fusedActivationFuncNone = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            fusedActivationFuncNone);
        return false;
    }

    FuseCode fusedActivationCodeValue = ANEURALNETWORKS_FUSED_NONE;
    status = ANeuralNetworksModel_setOperandValue(
            model_, fusedActivationFuncNone, &fusedActivationCodeValue,
            sizeof(fusedActivationCodeValue));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValue failed for operand (%d)",
                            fusedActivationFuncNone);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor0 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor0);
        return false;
    }
    status = ANeuralNetworksModel_setOperandValueFromMemory(model_,
                                                            tensor0,
                                                            memoryModel_,
                                                            offset_,
                                                            tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValueFromMemory failed for operand (%d)",
                            tensor0);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor1 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor1);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor2 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor2);
        return false;
    }
    status = ANeuralNetworksModel_setOperandValueFromMemory(
            model_, tensor2, memoryModel_, offset_ + tensorSize_ * sizeof(float),
            tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_setOperandValueFromMemory failed for operand (%d)",
                            tensor2);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t tensor3 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            tensor3);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t intermediateOutput0 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            intermediateOutput0);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t intermediateOutput1 = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            intermediateOutput1);
        return false;
    }

    status = ANeuralNetworksModel_addOperand(model_, &float32TensorType);
    uint32_t multiplierOutput = opIdx++;
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperand failed for operand (%d)",
                            multiplierOutput);
        return false;
    }

    std::vector<uint32_t> add1InputOperands = {
            tensor0,
            tensor1,
            fusedActivationFuncNone,
    };
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_ADD,
                                               add1InputOperands.size(), add1InputOperands.data(),
                                               1, &intermediateOutput0);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for ADD_1");
        return false;
    }

    std::vector<uint32_t> add2InputOperands = {
            tensor2,
            tensor3,
            fusedActivationFuncNone,
    };
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_ADD,
                                               add2InputOperands.size(),add2InputOperands.data(),
                                               1, &intermediateOutput1);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for ADD_2");
        return false;
    }

    std::vector<uint32_t> mulInputOperands = {
            intermediateOutput0,
            intermediateOutput1,
            fusedActivationFuncNone};
    status = ANeuralNetworksModel_addOperation(model_, ANEURALNETWORKS_MUL,
                                               mulInputOperands.size(),mulInputOperands.data(),
                                               1, &multiplierOutput);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_addOperation failed for MUL");
        return false;
    }

    std::vector<uint32_t> modelInputOperands = {
            tensor1, tensor3,
    };
    status = ANeuralNetworksModel_identifyInputsAndOutputs(model_,
                                                           modelInputOperands.size(),
                                                           modelInputOperands.data(),
                                                           1,
                                                           &multiplierOutput);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_identifyInputsAndOutputs failed");
        return false;
    }

    status = ANeuralNetworksModel_finish(model_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksModel_finish failed");
        return false;
    }

    status = ANeuralNetworksCompilation_create(model_, &compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_create failed");
        return false;
    }

    status = ANeuralNetworksCompilation_setPreference(compilation_,
                                                      ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_setPreference failed");
        return false;
    }

    status = ANeuralNetworksCompilation_finish(compilation_);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksCompilation_finish failed");
        return false;
    }

    return true;
}

bool Model::Compute(float inputValue1, float inputValue2,
                          float *result) {
    if (!result) {
        return false;
    }

    ANeuralNetworksExecution *execution;
    int32_t status = ANeuralNetworksExecution_create(compilation_, &execution);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_create failed");
        return false;
    }

    std::fill(inputTensor1_.data(), inputTensor1_.data() + tensorSize_,
              inputValue1);

    status = ANeuralNetworksExecution_setInput(execution, 0, nullptr,
                                               inputTensor1_.data(),
                                               tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setInput failed for input1");
        return false;
    }

    float *inputTensor2Ptr = reinterpret_cast<float *>(mmap(nullptr, tensorSize_ * sizeof(float),
                                                            PROT_READ | PROT_WRITE, MAP_SHARED,
                                                            inputTensor2Fd_, 0));
    for (int i = 0; i < tensorSize_; i++) {
        *inputTensor2Ptr = inputValue2;
        inputTensor2Ptr++;
    }
    munmap(inputTensor2Ptr, tensorSize_ * sizeof(float));

    status = ANeuralNetworksExecution_setInputFromMemory(execution, 1, nullptr,
                                                         memoryInput2_, 0,
                                                         tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setInputFromMemory failed for input2");
        return false;
    }

    status = ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                          memoryOutput_, 0,
                                                          tensorSize_ * sizeof(float));
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_setOutputFromMemory failed for output");
        return false;
    }

    ANeuralNetworksEvent *event = nullptr;
    status = ANeuralNetworksExecution_startCompute(execution, &event);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksExecution_startCompute failed");
        return false;
    }

    status = ANeuralNetworksEvent_wait(event);
    if (status != ANEURALNETWORKS_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "ANeuralNetworksEvent_wait failed");
        return false;
    }

    ANeuralNetworksEvent_free(event);
    ANeuralNetworksExecution_free(execution);

    const float goldenRef = (inputValue1 + 0.5f) * (inputValue2 + 0.5f);
    float *outputTensorPtr = reinterpret_cast<float *>(mmap(nullptr,
                                                            tensorSize_ * sizeof(float),
                                                            PROT_READ, MAP_SHARED,
                                                            outputTensorFd_, 0));
    for (int32_t idx = 0; idx < tensorSize_; idx++) {
        float delta = outputTensorPtr[idx] - goldenRef;
        delta = (delta < 0.0f) ? (-delta) : delta;
        if (delta > FLOAT_EPISILON) {
            __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                                "Output computation Error: output0(%f), delta(%f) @ idx(%d)",
                                outputTensorPtr[0], delta, idx);
        }
    }
    *result = outputTensorPtr[0];
    munmap(outputTensorPtr, tensorSize_ * sizeof(float));
    return result;
}

Model::~Model() {
    ANeuralNetworksCompilation_free(compilation_);
    ANeuralNetworksModel_free(model_);
    ANeuralNetworksMemory_free(memoryModel_);
    ANeuralNetworksMemory_free(memoryInput2_);
    ANeuralNetworksMemory_free(memoryOutput_);
    close(inputTensor2Fd_);
    close(outputTensorFd_);
    close(modelDataFd_);
}
