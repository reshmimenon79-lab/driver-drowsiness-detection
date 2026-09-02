# Driver Drowsiness Detection System

## Overview

The Driver Drowsiness Detection System is a real-time application designed to detect signs of driver fatigue using computer vision and deep learning.

The system captures live video from a webcam and monitors facial features such as the eyes and mouth. Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) are used to identify prolonged eye closure and yawning. When signs of drowsiness are detected, the system triggers an audio alarm to alert the driver.

## Features

- Real-time webcam monitoring
- Face and facial landmark detection
- Eye closure detection using Eye Aspect Ratio (EAR)
- Yawning detection using Mouth Aspect Ratio (MAR)
- Deep learning based driver-state classification
- Consecutive frame monitoring to reduce false alarms
- Audio alarm when drowsiness is detected
- Camera testing functionality

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
├── PROJECT_DOCUMENTATION.md
├── QUICKSTART.md
├── SETUP_REQUIREMENTS.md
└── CAMERA_CRASH_FIXES.md
```

## Important Files

### detect.py
Main program for real-time driver drowsiness detection.

### detect_no_dlib.py
Alternative implementation that can run without Dlib.

### detect_simple.py
Simplified version of the detection system.

### test_camera.py
Used to test whether the webcam is detected and working correctly.

### model.json
Contains the architecture of the trained deep learning model.

### model.weights.h5
Contains the trained weights of the deep learning model.

### alarm.wav
Audio file used to alert the driver when drowsiness is detected.

## How the System Works

The basic workflow is:

```text
Webcam
   ↓
Capture Video Frame
   ↓
Detect Face
   ↓
Detect Eyes and Mouth
   ↓
Calculate EAR and MAR
   ↓
Check Thresholds
   ↓
Monitor Consecutive Frames
   ↓
Detect Drowsiness
   ↓
Trigger Alarm
```

## Eye Aspect Ratio (EAR)

Eye Aspect Ratio is used to determine whether the driver's eyes are open or closed.

When the eyes are open, the EAR value is generally higher.

When the eyes close, the EAR value decreases.

If the EAR remains below the configured threshold for several consecutive frames, the system considers it a possible sign of drowsiness.

## Mouth Aspect Ratio (MAR)

Mouth Aspect Ratio is used to monitor mouth opening.

A significant increase in MAR for a period of time may indicate yawning, which can be another sign of driver fatigue.

## Deep Learning Model

The project also uses a trained deep learning model for driver-state classification.

The model architecture is stored in:

`model.json`

The trained model weights are stored in:

`model.weights.h5`

## Installation

Install the required Python libraries before running the project.

```bash
pip install tensorflow opencv-python numpy scipy dlib imutils pygame
```

## Run the Project

Run the main detection program using:

```bash
python detect.py
```

Make sure a working webcam is connected to the computer.

## Future Improvements

- Improve accuracy under low-light conditions
- Reduce false alarms while the driver is speaking
- Improve detection for drivers wearing glasses
- Add PERCLOS-based fatigue analysis
- Optimize the system for embedded devices
- Add driver monitoring statistics and reports

## Author

**Reshmi Menon**

B.E. Computer Science and Engineering
