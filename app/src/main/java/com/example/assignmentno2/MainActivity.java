package com.example.assignmentno2;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.IOException;
import java.util.Locale;

/**
 * StudyScore Predictor - AI-based mini application (Assignment No. 2, Task 3).
 *
 * The user enters how many hours they plan to study for an exam, taps "Predict My
 * Score", and an on-device TensorFlow Lite model predicts their likely score.
 *
 * This reuses the exact TFLite Interpreter pattern analysed from the reference
 * "NumberGuessingAppTflite" repository (Task 1/2 of this assignment):
 *   model loaded from assets -> Interpreter.run() -> result shown in the UI.
 */
public class MainActivity extends Activity {

    private EditText hoursInput;
    private Button predictButton;
    private TextView resultText;
    private StudyScorePredictor predictor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        hoursInput = findViewById(R.id.hoursInput);
        predictButton = findViewById(R.id.predictButton);
        resultText = findViewById(R.id.resultText);

        try {
            predictor = new StudyScorePredictor(this);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load study_score.tflite", e);
        }

        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = hoursInput.getText().toString().trim();
                if (text.isEmpty()) {
                    resultText.setText("Please enter the hours you plan to study");
                    return;
                }
                try {
                    float hours = Float.parseFloat(text);
                    // This is the actual on-device inference call - no network round-trip.
                    float predictedScore = predictor.predictScore(hours);
                    resultText.setText(String.format(Locale.US, "Predicted score: %.1f%%", predictedScore));
                } catch (NumberFormatException e) {
                    resultText.setText("Please enter a valid number of hours");
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (predictor != null) {
            predictor.close();
        }
    }
}
