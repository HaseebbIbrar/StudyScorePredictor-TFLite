# StudyScore Predictor (TensorFlow Lite, on-device)

AI-based mini Android application built AI-Based Mini Application
Development, Integrating ML/DL Models into Android with TensorFlow Lite.

The app predicts a student's likely exam score from the number of hours they plan to
study, using a small regression model that runs entirely on-device with TensorFlow
Lite (no server, no network call at inference time).

Reference repository studied for Task 1/2 of this assignment:
https://github.com/Uzair390-Del/NumberGuessingAppTflite

## How it works

1. A tiny Keras model (Input(1) -> Dense(8, relu) -> Dense(1)) was trained on synthetic
   data (score approx 7.5 times hours plus 10, clipped to 0-100) and converted to TensorFlow Lite.
2. The tflite file is bundled as an Android asset (app/src/main/assets/study_score.tflite).
3. StudyScorePredictor.java memory-maps the asset and loads it into a TFLite
   Interpreter (the same pattern used by the reference repo's MainActivity.java).
4. MainActivity.java reads the hours typed into an EditText, calls
   interpreter.run(input, output), and shows the predicted score in a TextView.

## Project structure

app/src/main/java/com/example/assignmentno2/MainActivity.java - UI and button handler
app/src/main/java/com/example/assignmentno2/StudyScorePredictor.java - TFLite loading and inference
app/src/main/res/layout/activity_main.xml - UI layout
app/src/main/assets/study_score.tflite - the on-device model

## Prerequisites

- Android Studio with Android SDK 37 installed
- A device or emulator running Android 11 (API 30) or newer (minSdk 30)

## Running the app

1. Clone this repository.
2. Open the project folder in Android Studio and let Gradle sync (this downloads the
   org.tensorflow:tensorflow-lite dependency automatically).
3. Select a device or emulator and click Run.
4. Enter a number of hours (for example 6) and tap Predict My Score.

## Dependencies

- org.tensorflow:tensorflow-lite:2.17.0
- AndroidX core-ktx, lifecycle-runtime-ktx (default Android Studio template deps)

## Notes

- study_score.tflite is intentionally a very small model (about 1.6 KB), more than enough
  to demonstrate the full on-device ML integration pipeline required for this
  assignment, while staying well within mini assignment scope.
- The prediction is for demonstration purposes only; the relationship between study
  hours and score is a synthetic approximation, not real exam data.
