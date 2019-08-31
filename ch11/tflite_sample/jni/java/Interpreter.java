package org.tensorflow.lite;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;

public final class Interpreter implements AutoCloseable {
  public static class Options {
    public Options() {}

    public Options setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    public Options setUseNNAPI(boolean useNNAPI) {
      this.useNNAPI = useNNAPI;
      return this;
    }

    public Options setAllowFp16PrecisionForFp32(boolean allow) {
      this.allowFp16PrecisionForFp32 = allow;
      return this;
    }

    public Options addDelegate(Delegate delegate) {
      delegates.add(delegate);
      return this;
    }

    public Options setAllowBufferHandleOutput(boolean allow) {
      this.allowBufferHandleOutput = allow;
      return this;
    }

    int numThreads = -1;
    Boolean useNNAPI;
    Boolean allowFp16PrecisionForFp32;
    Boolean allowBufferHandleOutput;
    final List<Delegate> delegates = new ArrayList<>();
  }

  public Interpreter(@NonNull File modelFile) {
    this(modelFile, /*options = */ null);
  }

  public Interpreter(@NonNull File modelFile, Options options) {
    wrapper = new NativeInterpreterWrapper(modelFile.getAbsolutePath(), options);
  }

  public Interpreter(@NonNull ByteBuffer byteBuffer) {
    this(byteBuffer, /* options= */ null);
  }

  public Interpreter(@NonNull ByteBuffer byteBuffer, Options options) {
    wrapper = new NativeInterpreterWrapper(byteBuffer, options);
  }

  public void run(Object input, Object output) {
    Object[] inputs = {input};
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, output);
    runForMultipleInputsOutputs(inputs, outputs);
  }

  public void runForMultipleInputsOutputs(
      @NonNull Object[] inputs, @NonNull Map<Integer, Object> outputs) {
    checkNotClosed();
    wrapper.run(inputs, outputs);
  }

  public void resizeInput(int idx, @NonNull int[] dims) {
    checkNotClosed();
    wrapper.resizeInput(idx, dims);
  }

  public int getInputTensorCount() {
    checkNotClosed();
    return wrapper.getInputTensorCount();
  }

  public int getInputIndex(String opName) {
    checkNotClosed();
    return wrapper.getInputIndex(opName);
  }

  public Tensor getInputTensor(int inputIndex) {
    checkNotClosed();
    return wrapper.getInputTensor(inputIndex);
  }

  public int getOutputTensorCount() {
    checkNotClosed();
    return wrapper.getOutputTensorCount();
  }

  public int getOutputIndex(String opName) {
    checkNotClosed();
    return wrapper.getOutputIndex(opName);
  }

  public Tensor getOutputTensor(int outputIndex) {
    checkNotClosed();
    return wrapper.getOutputTensor(outputIndex);
  }

  public Long getLastNativeInferenceDurationNanoseconds() {
    checkNotClosed();
    return wrapper.getLastNativeInferenceDurationNanoseconds();
  }

  public void modifyGraphWithDelegate(Delegate delegate) {
    checkNotClosed();
    wrapper.modifyGraphWithDelegate(delegate);
  }

  @Override
  public void close() {
    if (wrapper != null) {
      wrapper.close();
      wrapper = null;
    }
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      close();
    } finally {
      super.finalize();
    }
  }

  private void checkNotClosed() {
    if (wrapper == null) {
      throw new IllegalStateException("Internal error: The Interpreter has already been closed.");
    }
  }

  NativeInterpreterWrapper wrapper;
}
