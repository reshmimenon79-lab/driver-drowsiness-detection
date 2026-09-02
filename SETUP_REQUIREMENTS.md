# Drowsiness Detection Project - Setup Requirements

## ✅ Files You Have
- `detect.py` - Main detection script
- `model.json` - CNN model architecture
- `model.weights.h5` - Pre-trained weights

## ❌ Missing Files Required

### 1. **shape_predictor_68_face_landmarks.dat** (Required)
   - **Purpose**: Detects 68 facial landmarks (eyes, mouth, nose, etc.)
   - **Size**: ~100 MB
   - **Where to get**:
     ```
     Download from: https://github.com/davisking/dlib-models/releases
     Look for: shape_predictor_68_face_landmarks.dat.bz2
     ```
   - **How to extract**:
     - Extract the `.bz2` file using 7-Zip or another archive tool
     - Place `shape_predictor_68_face_landmarks.dat` in your project folder

### 2. **alarm.wav** (Optional but recommended)
   - **Purpose**: Audio alarm when drowsiness is detected
   - **How to create**:
     - Option 1: Record a WAV file using Audacity (free audio editor)
     - Option 2: Download from free audio sites (freesound.org, zapsplat.com)
     - Option 3: Use a built-in Windows sound file
   - **Placement**: Save as `alarm.wav` in your project folder
   - **Note**: If missing, the program still works but won't play audio

## 📁 Project Structure
```
Drowsiness_Project/
├── detect.py
├── model.json
├── model.weights.h5
├── shape_predictor_68_face_landmarks.dat  ← ADD THIS
├── alarm.wav                               ← ADD THIS (optional)
└── venv/
```

## 🚀 Quick Start
1. Place `shape_predictor_68_face_landmarks.dat` in the project folder
2. (Optional) Add `alarm.wav` file
3. Run: `python detect.py`
4. Press ESC to exit

## ⚠️ Troubleshooting
- **"shape_predictor_68_face_landmarks.dat not found"**: Download and extract from link above
- **"alarm.wav not found"**: Program will still work without audio
- **"Cannot access camera"**: Check if camera is already in use by another app
- **Model loading errors**: Verify `model.json` and `model.weights.h5` exist and are valid
