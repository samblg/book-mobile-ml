#include <jni.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <fcntl.h>

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/sharedmem.h>
#include <sys/mman.h>

#include "model.h"

extern "C"
JNIEXPORT jlong
JNICALL
Java_com_example_android_nnapidemo_MainActivity_initModel(
        JNIEnv *env,
        jobject /* this */,
        jobject _assetManager,
        jstring _assetName) {
    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    const char *assetName = env->GetStringUTFChars(_assetName, NULL);
    AAsset *asset = AAssetManager_open(assetManager, assetName, AASSET_MODE_BUFFER);
    if(asset == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to open the asset.");
        return 0;
    }
    env->ReleaseStringUTFChars(_assetName, assetName);
    off_t offset, length;
    int fd = AAsset_openFileDescriptor(asset, &offset, &length);
    AAsset_close(asset);
    if (fd < 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to open the model_data file descriptor.");
        return 0;
    }
    Model* nn_model = new Model(length, PROT_READ, fd, offset);
    if (!nn_model->CreateCompiledModel()) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,
                            "Failed to prepare the model.");
        return 0;
    }

    return (jlong)(uintptr_t)nn_model;
}

extern "C"
JNIEXPORT jfloat
JNICALL
Java_com_example_android_nnapidemo_MainActivity_startCompute(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel,
        jfloat inputValue1,
        jfloat inputValue2) {
    Model* nn_model = (Model*) _nnModel;
    float result = 0.0f;
    nn_model->Compute(inputValue1, inputValue2, &result);
    return result;
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_android_nnapidemo_MainActivity_destroyModel(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel) {
    Model* nn_model = (Model*) _nnModel;
    delete(nn_model);
}
