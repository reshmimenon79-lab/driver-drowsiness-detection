# Drowsiness Detection Project Documentation

## 1. Project Overview

This project is a real-time driver drowsiness detection system built with Python, OpenCV, TensorFlow/Keras, and optional `dlib`.

The project supports two execution paths:

- `detect.py`
  Main entry point. Uses facial landmarks with `dlib` when available.
- `detect_no_dlib.py`
  Automatic fallback path when `dlib` is not installed. Uses OpenCV Haar cascades and CNN prediction.

The goal of the system is to detect unsafe driver states such as:

- drowsiness from eye closure
- yawning from sustained mouth opening

and then alert the user with an audible alarm.

## 2. Current Project Files

Main files in the project:

- [detect.py](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/detect.py)
  Main detector with `dlib`-based landmark pipeline.
- [detect_no_dlib.py](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/detect_no_dlib.py)
  Fallback detector without `dlib`.
- [detect_simple.py](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/detect_simple.py)
  Older/simpler detection implementation.
- [model.json](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/model.json)
  Saved Keras model architecture.
- [model.weights.h5](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/model.weights.h5)
  Saved model weights.
- [shape_predictor_68_face_landmarks.dat](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/shape_predictor_68_face_landmarks.dat)
  `dlib` facial landmark predictor used by `detect.py`.
- [alarm.wav](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/alarm.wav)
  Alarm sound file.
- [test_camera.py](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/test_camera.py)
  Camera testing utility.
- [QUICKSTART.md](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/QUICKSTART.md)
  Quick start instructions.
- [SETUP_REQUIREMENTS.md](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/SETUP_REQUIREMENTS.md)
  Setup notes and required assets.
- [CAMERA_CRASH_FIXES.md](c:/Users/LENOVO/Downloads/Drowsiness_Project/Drowsiness_Project/CAMERA_CRASH_FIXES.md)
  Notes about camera robustness and crash handling.

## 3. Main Entry Flow

The normal command to run the project is:

```powershell
python detect.py
```

Execution flow:

1. `detect.py` starts.
2. It tries to import `dlib`.
3. If `dlib` is available:
   - it uses landmark-based detection.
4. If `dlib` is not available:
   - it launches `detect_no_dlib.py` automatically.

This means `detect.py` is always the correct file to run.

## 4. Core Features

### 4.1 Drowsiness Detection

The system detects drowsiness using:

- Eye Aspect Ratio (EAR)
- CNN-based face classification
- Frame-based confirmation counters

### 4.2 Yawn Detection

In the `dlib` path, yawning is detected using:

- Mouth Aspect Ratio (MAR)
- Adaptive mouth thresholding
- Consecutive frame counting

### 4.3 Alarm System

The system includes:

- `alarm.wav` sound playback
- cooldown logic
- re-arm logic after returning to normal state

### 4.4 Camera Handling

The project tries multiple camera backends and indexes to improve compatibility on Windows.

Supported camera behaviors:

- default OpenCV backend
- `CAP_DSHOW`
- `CAP_MSMF`
- `CAP_VFW`
- `CAP_ANY`

### 4.5 Fallback Modes

The no-`dlib` detector supports:

- webcam input
- optional video-file input
- demo mode when no live source is available

## 5. Detector Architecture

### 5.1 `detect.py` Architecture

`detect.py` uses:

- `dlib` frontal face detector
- 68-point facial landmark predictor
- EAR calculation from eye landmarks
- MAR calculation from mouth landmarks
- CNN prediction on face ROI
- adaptive thresholds for EAR and MAR
- HUD overlays for current metrics and status

Main internal components:

- `Cfg`
  Central configuration values.
- `Alarm`
  Alarm playback and cooldown control.
- `build_model()`
  Loads Keras model from saved files.
- `calc_ear()`
  Calculates eye aspect ratio.
- `calc_mar()`
  Calculates mouth aspect ratio.
- `open_camera()`
  Tries camera indexes and backends.
- `draw_hud()`
  Draws status badge, thresholds, and counters.
- `main()`
  Main runtime loop.

### 5.2 `detect_no_dlib.py` Architecture

`detect_no_dlib.py` uses:

- OpenCV Haar cascade face detection
- OpenCV Haar eye detection
- CNN face ROI prediction
- missing-eye heuristic for drowsiness
- fallback camera/video/demo modes

Important behaviors:

- Loads and sanitizes Keras JSON config to avoid version incompatibility issues.
- Detects actual model input size dynamically.
- Tracks the largest face only.
- Uses upper-face eye search.
- Holds the last detected face for a few frames to reduce `No face` flicker.

## 6. Detection Logic

### 6.1 EAR

EAR is used to infer eye openness.

In the `dlib` path:

- EAR is computed from landmark geometry.
- A rolling buffer smooths noise.
- An adaptive threshold is learned from the user’s normal face.

In the no-`dlib` path:

- EAR is approximated using Haar eye rectangles.
- If eyes disappear for several frames, that is also treated as a drowsiness signal.

### 6.2 MAR

MAR is used to infer mouth opening and yawning.

In the `dlib` path:

- MAR is calculated from mouth landmarks.
- A dynamic mouth threshold is derived from the user’s normal mouth shape.
- Yawn is confirmed only after sustained mouth opening across multiple frames.

### 6.3 CNN Prediction

The CNN model predicts a drowsiness score from the face ROI.

Current model files:

- `model.json`
- `model.weights.h5`

The code now automatically detects the model’s expected input size.

### 6.4 Status States

The main visible states are:

- `NORMAL`
- `DROWSY`
- `YAWNING`

Status is displayed in the HUD and is tied to counters and thresholds.

## 7. Model Loading

The project loads a saved Keras model in two ways:

1. Try direct `model_from_json(...)`
2. If that fails, manually rebuild the architecture layer-by-layer

The code includes compatibility handling for:

- unsupported `quantization_config`
- JSON config cleanup
- dynamic input-size detection

This was added because the saved model and the installed Keras version were not fully compatible by default.

## 8. Camera and Input Sources

### 8.1 Webcam

The system tries common indexes:

- `0`
- `1`
- `2`

and multiple backends.

### 8.2 Manual Camera Selection

You can manually force a webcam index with:

```powershell
$env:DROWSY_CAMERA_INDEX="0"; python detect.py
```

or:

```powershell
$env:DROWSY_CAMERA_INDEX="1"; python detect.py
```

### 8.3 Video File Input

The no-`dlib` path can use a video file:

```powershell
$env:DROWSY_VIDEO_PATH="C:\path\to\video.mp4"; python detect.py
```

### 8.4 Demo Mode

If no webcam and no video file are available, `detect_no_dlib.py` falls back to demo mode.

Demo mode:

- creates a synthetic frame
- alternates between normal and drowsy simulation
- lets the UI and alarm path keep running

## 9. Alarm Behavior

Alarm behavior is controlled through the `Alarm` class in `detect.py` and simple flags in `detect_no_dlib.py`.

Alarm features:

- one-shot alarm trigger
- cooldown timer
- re-arming after return to normal
- `alarm.wav` playback through `pygame`

## 10. User Interface / HUD

The detector displays:

- current state label
- EAR and threshold
- MAR and threshold
- CNN score and threshold
- progress bars for eye closure, yawn, CNN frames, and normal frames
- cooldown status

The fallback detector displays:

- status label
- EAR
- CNN score
- missing-eye counter
- drowsy frame count
- optional face-hold counter

## 11. Environment Variables

Supported environment variables:

- `DROWSY_CAMERA_INDEX`
  Forces a specific webcam index.
- `DROWSY_VIDEO_PATH`
  Uses a local video file instead of webcam in the fallback path.

## 12. Dependencies

Main Python dependencies used in the code:

- `python`
- `opencv-python`
- `numpy`
- `pygame`
- `tensorflow`
- `scipy`
- `imutils`
- `dlib` (optional but required for `detect.py` landmark mode)

## 13. How To Run

### Standard run

```powershell
python detect.py
```

### Force camera index

```powershell
$env:DROWSY_CAMERA_INDEX="0"; python detect.py
```

### Use video file

```powershell
$env:DROWSY_VIDEO_PATH="C:\path\to\video.mp4"; python detect.py
```

### Test camera only

```powershell
python test_camera.py
```

## 14. Typical Runtime Flow

### When everything works

1. Program starts
2. Model loads
3. Camera opens
4. Face is detected
5. EAR/MAR/CNN are calculated
6. Status is updated
7. Alarm plays when alert conditions are met

### When `dlib` is missing

1. `detect.py` starts
2. `dlib` import fails
3. `detect.py` launches `detect_no_dlib.py`
4. Fallback detector uses OpenCV cascades instead

### When no webcam exists

1. Camera probing fails
2. If `DROWSY_VIDEO_PATH` exists:
   - video mode starts
3. Otherwise:
   - demo mode starts

## 15. Current Strengths

- Main entry point is stable.
- `dlib` missing no longer crashes startup.
- Model-loading compatibility was improved.
- Camera probing is more flexible.
- No-camera situations no longer hard-fail in fallback mode.
- Face tracking in fallback mode is more stable due to face-hold behavior.

## 16. Current Limitations

- Yawn detection is strongest in the `dlib` path; it is not fully implemented in the fallback path.
- Haar cascade eye detection is less accurate than landmark-based eye geometry.
- Webcam availability still depends on the local machine and driver support.
- CNN thresholding may still need tuning for different lighting, faces, and cameras.
- The repository currently contains multiple detector variants, which can make maintenance harder.

## 17. Suggested Improvements

Recommended future improvements:

- merge both detector paths into one unified codebase
- add a proper configuration file
- add a logging mode instead of raw console prints
- add recording/replay of test videos
- add a calibration step for each user
- improve no-`dlib` yawning detection
- add requirements file with exact package versions
- create one polished README and one technical design document

## 18. Summary

This project is a real-time drowsiness detection system with:

- a primary `dlib`-based landmark detector
- an OpenCV cascade fallback detector
- CNN-based face-state prediction
- alarm playback
- camera probing and fallback modes

The recommended file to run is:

```powershell
python detect.py
```

The system will automatically choose the best available detection path based on whether `dlib` is installed and whether a webcam is available.
