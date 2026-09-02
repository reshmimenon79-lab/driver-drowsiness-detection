# Driver Drowsiness Detection System

## Overview

The Driver Drowsiness Detection System is a real-time computer vision and deep learning application designed to detect signs of driver fatigue.

The system captures live video from a webcam and monitors the driver's facial features. It combines Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and CNN-based drowsiness prediction to identify prolonged eye closure, yawning, and possible drowsiness.

When drowsiness or yawning is confirmed for the required duration, the system triggers an audio alarm to alert the driver.

## Features

- Real-time webcam monitoring
- Face detection using Dlib
- 68-point facial landmark detection
- Eye closure detection using Eye Aspect Ratio (EAR)
- Yawning detection using Mouth Aspect Ratio (MAR)
- CNN-based drowsiness prediction
- Adaptive EAR and MAR thresholds
- Consecutive-frame monitoring to reduce false alarms
- Audio alarm for drowsiness and yawning
- Alarm cooldown and re-arming mechanism
- Automatic camera detection
- Alternative detection mode when Dlib is unavailable
- Camera testing utility

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Dlib
- NumPy
- SciPy
- Imutils
- Pygame

## Project Structure

```text
driver-drowsiness-detection/
│
├── detect.py
├── detect_no_dlib.py
├── detect_simple.py
├── test_camera.py
├── model.json
├── model.weights.h5
├── alarm.wav
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_DOCUMENTATION.md
├── QUICKSTART.md
├── SETUP_REQUIREMENTS.md
└── CAMERA_CRASH_FIXES.md
```

The Dlib facial landmark file `shape_predictor_68_face_landmarks.dat`
is required for the main detection program but is not included in this
repository because of its large file size.

## Important Files

### `detect.py`

Main real-time drowsiness detection program.

It combines:

- Dlib facial landmarks
- EAR-based eye closure detection
- MAR-based yawning detection
- CNN-based drowsiness prediction
- Adaptive thresholds
- Audio alarm logic

### `detect_no_dlib.py`

Alternative detection implementation used when Dlib is not available.

### `detect_simple.py`

Simplified implementation of the drowsiness detection system.

### `test_camera.py`

Utility program used to check whether the webcam is detected and working correctly.

### `model.json`

Contains the architecture of the trained CNN model.

### `model.weights.h5`

Contains the trained weights of the CNN model.

### `alarm.wav`

Audio warning played when drowsiness or yawning is detected.

## How the System Works

The basic processing flow is:

```text
Webcam
   ↓
Capture Video Frame
   ↓
Detect Driver's Face
   ↓
Extract Facial Landmarks
   ↓
Eyes and Mouth
   ↓
Calculate EAR and MAR
   ↓
CNN Drowsiness Prediction
   ↓
Apply Adaptive Thresholds
   ↓
Monitor Consecutive Frames
   ↓
Classify Driver State
   ↓
NORMAL / YAWNING / DROWSY
   ↓
Trigger Alarm if Required
```

## Eye Aspect Ratio (EAR)

Eye Aspect Ratio is used to determine whether the driver's eyes are open or closed.

EAR is calculated using distances between the facial landmark points around each eye.

When the eyes are open, the EAR value is generally higher. When the eyes close, the EAR value decreases.

The system monitors the EAR value across consecutive frames instead of triggering an alarm from a single frame. This helps distinguish prolonged eye closure from a normal blink.

The default EAR threshold in the current implementation is:

```text
0.26
```

The system also maintains an adaptive EAR threshold based on the driver's observed open-eye measurements.

## Mouth Aspect Ratio (MAR)

Mouth Aspect Ratio is used to detect prolonged mouth opening associated with yawning.

The MAR value is calculated using distances between facial landmark points around the mouth.

A high MAR value maintained for the required duration can indicate a yawn.

The default MAR threshold is:

```text
0.72
```

The system also adapts the MAR threshold based on the driver's observed facial measurements.

## CNN-Based Drowsiness Detection

In addition to EAR and MAR, the system uses a trained CNN model to calculate a drowsiness score from the detected face.

The repository contains:

```text
model.json
model.weights.h5
```

`model.json` contains the model architecture and `model.weights.h5` contains the trained model weights.

The default CNN drowsiness threshold is:

```text
0.60
```

The program can also attempt to load `mobilenetv2_drowsy.h5` if that model file is available.

If no trained model can be loaded, the program contains a MobileNetV2 fallback. However, the fallback is not trained specifically for this drowsiness-detection task and therefore should not be expected to provide reliable drowsiness classification.

## Detection Logic

The system uses multiple signals instead of depending on only one measurement.

Drowsiness can be confirmed when:

- Eye closure continues for the required number of frames, or
- CNN predictions indicate drowsiness for the required number of frames, or
- Both eye-closure and CNN evidence reach their combined confirmation conditions.

Yawning is monitored separately using MAR and duration.

The final driver status displayed by the application is:

```text
NORMAL
YAWNING
DROWSY
```

## Current Detection Parameters

The main implementation currently uses approximately:

| Parameter | Value |
|---|---:|
| EAR threshold | 0.26 |
| EAR confirmation | 4 frames |
| MAR threshold | 0.72 |
| CNN threshold | 0.60 |
| CNN confirmation | 8 frames |
| Yawn duration | 1.3 seconds |
| Target FPS | 15 |
| Alarm cooldown | 10 seconds |
| Normal-state re-arm duration | 2 seconds |

These values can be modified in the configuration section of `detect.py`.

## Facial Landmark Model

The main `detect.py` program requires Dlib's 68-point facial landmark predictor:

```text
shape_predictor_68_face_landmarks.dat
```

Download the predictor separately and place it in the root directory of the project.

The expected location is:

```text
driver-drowsiness-detection/
│
├── detect.py
├── shape_predictor_68_face_landmarks.dat
├── model.json
├── model.weights.h5
└── alarm.wav
```

Without this file, the Dlib-based facial landmark detection in `detect.py` cannot run.

## Dlib Fallback

If Dlib is not installed, `detect.py` automatically attempts to run:

```text
detect_no_dlib.py
```

For full EAR and MAR facial-landmark detection, Dlib and the 68-point facial landmark predictor are required.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd driver-drowsiness-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

- TensorFlow
- OpenCV
- NumPy
- SciPy
- Dlib
- Imutils
- Pygame

### 3. Add the facial landmark predictor

Place:

```text
shape_predictor_68_face_landmarks.dat
```

inside the project root directory.

## Run the Project

First, you can test the webcam using:

```bash
python test_camera.py
```

Then run the main application:

```bash
python detect.py
```

Make sure a working webcam is connected or available.

Press:

```text
ESC
```

to close the drowsiness monitor.

## Alarm System

The application uses `alarm.wav` through Pygame.

When the driver is classified as `DROWSY` or `YAWNING`, the alarm is triggered.

The alarm includes a cooldown mechanism to prevent it from continuously restarting. After the system observes a sufficient normal period, the alarm is re-armed for future drowsiness events.

## Future Improvements

- Improve detection accuracy under low-light conditions
- Further reduce false positives caused by speaking
- Improve detection for drivers wearing glasses
- Add PERCLOS-based fatigue analysis
- Evaluate the model on a larger and more diverse dataset
- Optimize the system for embedded or edge devices
- Add driver monitoring statistics and reports

## Author

**Reshmi Menon**

B.E. Computer Science and Engineering
