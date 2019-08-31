package org.tensorflow.lite;

public enum DataType {
  FLOAT32(1),
  INT32(2),
  UINT8(3),
  INT64(4),
  STRING(5);

  private final int value;

  DataType(int value) {
    this.value = value;
  }

  public int byteSize() {
    switch (this) {
      case FLOAT32:
        return 4;
      case INT32:
        return 4;
      case UINT8:
        return 1;
      case INT64:
        return 8;
      case STRING:
        return -1;
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + this + " is not supported yet");
  }

  int c() {
    return value;
  }

  static DataType fromC(int c) {
    for (DataType t : values) {
      if (t.value == c) {
        return t;
      }
    }
    throw new IllegalArgumentException(
        "DataType error: DataType "
            + c
            + " is not recognized in Java (version "
            + TensorFlowLite.version()
            + ")");
  }

  String toStringName() {
    switch (this) {
      case FLOAT32:
        return "float";
      case INT32:
        return "int";
      case UINT8:
        return "byte";
      case INT64:
        return "long";
      case STRING:
        return "string";
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + this + " is not supported yet");
  }

  private static final DataType[] values = values();
}
