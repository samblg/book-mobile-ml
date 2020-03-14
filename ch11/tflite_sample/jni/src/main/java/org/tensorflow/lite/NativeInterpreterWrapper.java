package org.tensorflow.lite;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class NativeInterpreterWrapper implements AutoCloseable {

  NativeInterpreterWrapper(String modelPath) {
    this(modelPath, /* options= */ null);
  }

  NativeInterpreterWrapper(String modelPath, Interpreter.Options options) {
    long errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    long modelHandle = createModel(modelPath, errorHandle);
    init(errorHandle, modelHandle, options);
  }

  NativeInterpreterWrapper(ByteBuffer byteBuffer) {
    this(byteBuffer, /* options= */ null);
  }

  NativeInterpreterWrapper(ByteBuffer buffer, Interpreter.Options options) {
    if (buffer == null
        || (!(buffer instanceof MappedByteBuffer)
            && (!buffer.isDirect() || buffer.order() != ByteOrder.nativeOrder()))) {
      throw new IllegalArgumentException(
          "Model ByteBuffer should be either a MappedByteBuffer of the model file, or a direct "
              + "ByteBuffer using ByteOrder.nativeOrder() which contains bytes of model content.");
    }
    this.modelByteBuffer = buffer;
    long errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    long modelHandle = createModelWithBuffer(modelByteBuffer, errorHandle);
    init(errorHandle, modelHandle, options);
  }

  private void init(long errorHandle, long modelHandle, Interpreter.Options options) {
    if (options == null) {
      options = new Interpreter.Options();
    }
    this.errorHandle = errorHandle;
    this.modelHandle = modelHandle;
    this.interpreterHandle = createInterpreter(modelHandle, errorHandle, options.numThreads);
    this.inputTensors = new Tensor[getInputCount(interpreterHandle)];
    this.outputTensors = new Tensor[getOutputCount(interpreterHandle)];
    if (options.useNNAPI != null) {
      setUseNNAPI(options.useNNAPI.booleanValue());
    }
    if (options.allowFp16PrecisionForFp32 != null) {
      allowFp16PrecisionForFp32(
          interpreterHandle, options.allowFp16PrecisionForFp32.booleanValue());
    }
    if (options.allowBufferHandleOutput != null) {
      allowBufferHandleOutput(interpreterHandle, options.allowBufferHandleOutput.booleanValue());
    }
    for (Delegate delegate : options.delegates) {
      applyDelegate(interpreterHandle, errorHandle, delegate.getNativeHandle());
      delegates.add(delegate);
    }
    allocateTensors(interpreterHandle, errorHandle);
    this.isMemoryAllocated = true;
  }

  @Override
  public void close() {
    for (int i = 0; i < inputTensors.length; ++i) {
      if (inputTensors[i] != null) {
        inputTensors[i].close();
        inputTensors[i] = null;
      }
    }
    for (int i = 0; i < outputTensors.length; ++i) {
      if (outputTensors[i] != null) {
        outputTensors[i].close();
        outputTensors[i] = null;
      }
    }
    delete(errorHandle, modelHandle, interpreterHandle);
    errorHandle = 0;
    modelHandle = 0;
    interpreterHandle = 0;
    modelByteBuffer = null;
    inputsIndexes = null;
    outputsIndexes = null;
    isMemoryAllocated = false;
    delegates.clear();
  }

  void run(Object[] inputs, Map<Integer, Object> outputs) {
    inferenceDurationNanoseconds = -1;
    if (inputs == null || inputs.length == 0) {
      throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
    }
    if (outputs == null || outputs.isEmpty()) {
      throw new IllegalArgumentException("Input error: Outputs should not be null or empty.");
    }

    for (int i = 0; i < inputs.length; ++i) {
      Tensor tensor = getInputTensor(i);
      int[] newShape = tensor.getInputShapeIfDifferent(inputs[i]);
      if (newShape != null) {
        resizeInput(i, newShape);
      }
    }

    boolean needsAllocation = !isMemoryAllocated;
    if (needsAllocation) {
      allocateTensors(interpreterHandle, errorHandle);
      isMemoryAllocated = true;
    }

    for (int i = 0; i < inputs.length; ++i) {
      getInputTensor(i).setTo(inputs[i]);
    }

    long inferenceStartNanos = System.nanoTime();
    run(interpreterHandle, errorHandle);
    long inferenceDurationNanoseconds = System.nanoTime() - inferenceStartNanos;

    // Allocation can trigger dynamic resizing of output tensors, so refresh all output shapes.
    if (needsAllocation) {
      for (int i = 0; i < outputTensors.length; ++i) {
        if (outputTensors[i] != null) {
          outputTensors[i].refreshShape();
        }
      }
    }
    for (Map.Entry<Integer, Object> output : outputs.entrySet()) {
      getOutputTensor(output.getKey()).copyTo(output.getValue());
    }

    this.inferenceDurationNanoseconds = inferenceDurationNanoseconds;
  }

  private static native boolean run(long interpreterHandle, long errorHandle);

  void resizeInput(int idx, int[] dims) {
    if (resizeInput(interpreterHandle, errorHandle, idx, dims)) {
      isMemoryAllocated = false;
      if (inputTensors[idx] != null) {
        inputTensors[idx].refreshShape();
      }
    }
  }

  private static native boolean resizeInput(
      long interpreterHandle, long errorHandle, int inputIdx, int[] dims);

  void setUseNNAPI(boolean useNNAPI) {
    useNNAPI(interpreterHandle, useNNAPI);
  }

  void setNumThreads(int numThreads) {
    numThreads(interpreterHandle, numThreads);
  }

  void modifyGraphWithDelegate(Delegate delegate) {
    applyDelegate(interpreterHandle, errorHandle, delegate.getNativeHandle());
    delegates.add(delegate);
  }

  int getInputIndex(String name) {
    if (inputsIndexes == null) {
      String[] names = getInputNames(interpreterHandle);
      inputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          inputsIndexes.put(names[i], i);
        }
      }
    }
    if (inputsIndexes.containsKey(name)) {
      return inputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Input error: '%s' is not a valid name for any input. Names of inputs and their "
                  + "indexes are %s",
              name, inputsIndexes.toString()));
    }
  }

  int getOutputIndex(String name) {
    if (outputsIndexes == null) {
      String[] names = getOutputNames(interpreterHandle);
      outputsIndexes = new HashMap<>();
      if (names != null) {
        for (int i = 0; i < names.length; ++i) {
          outputsIndexes.put(names[i], i);
        }
      }
    }
    if (outputsIndexes.containsKey(name)) {
      return outputsIndexes.get(name);
    } else {
      throw new IllegalArgumentException(
          String.format(
              "Input error: '%s' is not a valid name for any output. Names of outputs and their "
                  + "indexes are %s",
              name, outputsIndexes.toString()));
    }
  }

  Long getLastNativeInferenceDurationNanoseconds() {
    return (inferenceDurationNanoseconds < 0) ? null : inferenceDurationNanoseconds;
  }

  int getOutputQuantizationZeroPoint(int index) {
    return getOutputQuantizationZeroPoint(interpreterHandle, index);
  }

  float getOutputQuantizationScale(int index) {
    return getOutputQuantizationScale(interpreterHandle, index);
  }

  int getInputTensorCount() {
    return inputTensors.length;
  }

  Tensor getInputTensor(int index) {
    if (index < 0 || index >= inputTensors.length) {
      throw new IllegalArgumentException("Invalid input Tensor index: " + index);
    }
    Tensor inputTensor = inputTensors[index];
    if (inputTensor == null) {
      inputTensor =
          inputTensors[index] =
              Tensor.fromIndex(interpreterHandle, getInputTensorIndex(interpreterHandle, index));
    }
    return inputTensor;
  }

  int getOutputTensorCount() {
    return outputTensors.length;
  }

  Tensor getOutputTensor(int index) {
    if (index < 0 || index >= outputTensors.length) {
      throw new IllegalArgumentException("Invalid output Tensor index: " + index);
    }
    Tensor outputTensor = outputTensors[index];
    if (outputTensor == null) {
      outputTensor =
          outputTensors[index] =
              Tensor.fromIndex(interpreterHandle, getOutputTensorIndex(interpreterHandle, index));
    }
    return outputTensor;
  }

  private static native int getOutputDataType(long interpreterHandle, int outputIdx);
  private static native int getOutputQuantizationZeroPoint(long interpreterHandle, int outputIdx);
  private static native float getOutputQuantizationScale(long interpreterHandle, int outputIdx);

  private static final int ERROR_BUFFER_SIZE = 512;

  private long errorHandle;
  private long interpreterHandle;
  private long modelHandle;
  private long inferenceDurationNanoseconds = -1;
  private ByteBuffer modelByteBuffer;
  private Map<String, Integer> inputsIndexes;
  private Map<String, Integer> outputsIndexes;
  private Tensor[] inputTensors;
  private Tensor[] outputTensors;
  private boolean isMemoryAllocated = false;
  private final List<Delegate> delegates = new ArrayList<>();

  private static native long allocateTensors(long interpreterHandle, long errorHandle);
  private static native int getInputTensorIndex(long interpreterHandle, int inputIdx);
  private static native int getOutputTensorIndex(long interpreterHandle, int outputIdx);
  private static native int getInputCount(long interpreterHandle);
  private static native int getOutputCount(long interpreterHandle);
  private static native String[] getInputNames(long interpreterHandle);
  private static native String[] getOutputNames(long interpreterHandle);
  private static native void useNNAPI(long interpreterHandle, boolean state);
  private static native void numThreads(long interpreterHandle, int numThreads);
  private static native void allowFp16PrecisionForFp32(long interpreterHandle, boolean allow);
  private static native void allowBufferHandleOutput(long interpreterHandle, boolean allow);
  private static native long createErrorReporter(int size);
  private static native long createModel(String modelPathOrBuffer, long errorHandle);
  private static native long createModelWithBuffer(ByteBuffer modelBuffer, long errorHandle);
  private static native long createInterpreter(long modelHandle, long errorHandle, int numThreads);
  private static native void applyDelegate(
      long interpreterHandle, long errorHandle, long delegateHandle);
  private static native void delete(long errorHandle, long modelHandle, long interpreterHandle);

  static {
    TensorFlowLite.init();
  }
}
