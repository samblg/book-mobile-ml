package org.tensorflow.demo;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;

public class TFLiteImageClassifier implements Classifier {
  private static final String TAG = "TFLiteImageClassifier";

  private static final int MAX_RESULTS = 3;

  private Interpreter tfLite;

  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  private static final int DIM_IMG_SIZE_X = 224;
  private static final int DIM_IMG_SIZE_Y = 224;

  byte[][] labelProb;

  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private ByteBuffer imgData = null;

  private TFLiteImageClassifier() {}

  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public static Classifier create(
      AssetManager assetManager, String modelFilename, String labelFilename, int inputSize) {
    TFLiteImageClassifier c = new TFLiteImageClassifier();

    Log.i(TAG, "Reading labels from: " + labelFilename);
    BufferedReader br = null;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(labelFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        c.labels.add(line);
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!" , e);
    }

    c.imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);

    c.imgData.order(ByteOrder.nativeOrder());
    try {
      c.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    Log.i(TAG, "Read " + c.labels.size() + " labels");

    c.intValues = new int[inputSize * inputSize];

    c.labelProb = new byte[1][c.labels.size()];

    return c;
  }

  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");

    long startTime;
    long endTime;
    startTime = SystemClock.uptimeMillis();

    convertBitmapToByteBuffer(bitmap);

    Trace.beginSection("run");
    startTime = SystemClock.uptimeMillis();
    tfLite.run(imgData, labelProb);
    endTime = SystemClock.uptimeMillis();
    Log.i(TAG, "Inf time: " + (endTime - startTime));
    Trace.endSection();

    PriorityQueue<Recognition> pq =
        new PriorityQueue<Recognition>(
            3,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    for (int i = 0; i < labels.size(); ++i) {
      pq.add(
          new Recognition(
              "" + i,
              labels.size() > i ? labels.get(i) : "unknown",
              (float) labelProb[0][i],
              null));
    }
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    Trace.endSection();
    return recognitions;
  }

  @Override
  public void enableStatLogging(boolean logStats) {
  }

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
  }
}
