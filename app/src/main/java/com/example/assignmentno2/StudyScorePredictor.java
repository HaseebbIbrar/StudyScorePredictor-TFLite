package com.example.assignmentno2;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * StudyScorePredictor
 *
 * Wraps a TensorFlow Lite {@link Interpreter} that predicts an expected exam score
 * (0-100) from the number of hours a student plans to study.
 *
 * The model ("study_score.tflite") is a small Keras regression network
 * (1 input -> Dense(8, relu) -> Dense(1) output) trained on synthetic data and
 * converted to TFLite. It is bundled as an Android asset, so every prediction runs
 * fully on-device: no network call, no server, and it keeps working offline.
 *
 * This mirrors the model-loading / inference pattern used by the reference
 * "NumberGuessingAppTflite" repository's MainActivity.java.
 */
public class StudyScorePredictor {

    private static final String MODEL_FILE_NAME = "study_score.tflite";

    // The Interpreter is TFLite's runtime object - it loads the model graph (weights +
    // operations) from the memory-mapped buffer below and knows how to execute it.
    private final Interpreter interpreter;

    public StudyScorePredictor(Context context) throws IOException {
        interpreter = new Interpreter(loadModelFile(context));
    }

    /**
     * Memory-maps the .tflite file directly out of the APK's assets folder.
     * Memory-mapping (instead of reading the whole file into a byte array) avoids an
     * extra copy of the model into the Java heap - it keeps load time fast and memory
     * usage low, which matters on resource-constrained mobile devices.
     */
    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor assetFileDescriptor = context.getAssets().openFd(MODEL_FILE_NAME);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Runs one forward pass ("inference") of the model.
     *
     * Input tensor shape  : [1, 1]  -> a single float, hours studied
     * Output tensor shape : [1, 1]  -> a single float, predicted score (0-100)
     *
     * TFLite's Interpreter.run() expects the input/output as nested arrays matching the
     * model's tensor shapes exactly, which is why both are wrapped one level deep here.
     */
    public float predictScore(float hoursStudied) {
        float[][] input = new float[][]{{hoursStudied}};   // shape [1][1]
        float[][] output = new float[1][1];                // shape [1][1]
        interpreter.run(input, output);

        float score = output[0][0];
        if (score < 0f) score = 0f;
        if (score > 100f) score = 100f;
        return score;
    }

    /** Releases the native TFLite resources. Call this when the activity is destroyed. */
    public void close() {
        interpreter.close();
    }
}
