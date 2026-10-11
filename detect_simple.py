import cv2
import numpy as np
import pygame
import os
import sys
import time
import random

# =========================
# INIT PYGAME
# =========================
try:
    pygame.mixer.init()
    alarm_available = os.path.exists("alarm.wav")
    if alarm_available:
        print("✅ Alarm sound file found")
    else:
        print("⚠️  alarm.wav not found - using system beep")
except Exception as e:
    print(f"⚠️  Pygame initialization error: {e}")
    alarm_available = False

alarm_playing = False

def play_alarm():
    global alarm_playing
    try:
        if not alarm_playing:
            alarm_playing = True
            if alarm_available:
                pygame.mixer.music.load("alarm.wav")
                pygame.mixer.music.play(-1)
            else:
                # Fallback: beep
                import winsound
                winsound.Beep(1000, 500)
            print("🔔 ALARM ON - Drowsiness detected!")
    except Exception as e:
        print(f"⚠️  Alarm error: {e}")
        alarm_playing = False

def stop_alarm():
    global alarm_playing
    try:
        if alarm_playing:
            if alarm_available and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            alarm_playing = False
            print("🔔 ALARM OFF - Alert cleared")
    except Exception as e:
        print(f"⚠️  Stop alarm error: {e}")

# =========================
# LOAD CASCADE CLASSIFIERS
# =========================
print("✅ Loading face and eye cascade classifiers...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# =========================
# PARAMETERS
# =========================
EAR_THRESHOLD = 0.3
MAR_THRESHOLD = 0.6  # Mouth aspect ratio for yawning
DROWSY_FRAMES = 5
YAWN_FRAMES = 3  # Frames for yawn detection
COOLDOWN_FRAMES = 30

# =========================
# CAMERA SETUP
# =========================
print("🔍 Initializing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found - running DEMO mode")
    demo_mode = True
else:
    demo_mode = False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("✅ Camera initialized")

print("Press ESC to exit\n")

# =========================
# MAIN LOOP
# =========================
alarm_on = False
drowsy_count = 0
yawn_count = 0
cooldown = 0
frame_num = 0

try:
    while True:
        if demo_mode:
            # Create demo frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (150, 100), (490, 400), (200, 200, 200), -1)
            cv2.circle(frame, (220, 150), 40, (100, 100, 100), -1)
            cv2.circle(frame, (420, 150), 40, (100, 100, 100), -1)
            cv2.ellipse(frame, (320, 300), (80, 50), 0, 0, 180, (100, 100, 100), -1)
            cv2.putText(frame, "DEMO MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Randomly trigger drowsiness in demo
            if random.random() < 0.08:
                if not alarm_on:
                    alarm_on = True
                    play_alarm()
                cv2.putText(frame, "DROWSY!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if alarm_on:
                    alarm_on = False
                    stop_alarm()
                cv2.putText(frame, "ALERT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If no face detected
        if len(faces) == 0:
            drowsy_count = 0
            if alarm_on:
                alarm_on = False
                stop_alarm()
        else:
            # Process first face
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Detect eyes in face
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Detect mouth (yawning)
            mouth = mouth_cascade.detectMultiScale(roi_gray)
            
            # Calculate eye aspect ratio (approximate)
            if len(eyes) >= 2:
                eye_heights = [eye[3] for eye in eyes[:2]]  # Get eye heights
                avg_eye_height = np.mean(eye_heights)
                
                # Simple drowsiness detection
                if avg_eye_height < 15:  # Eyes too small = closed/drowsy
                    drowsy_count += 1
                    yawn_count = 0  # Reset yawn if drowsy
                    cv2.putText(frame, f"DROWSY ({drowsy_count}/{DROWSY_FRAMES})", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    drowsy_count = 0
                    cv2.putText(frame, "ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                drowsy_count = 0
                cv2.putText(frame, "ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detect yawning (mouth opening)
            if len(mouth) > 0:
                mouth_height = mouth[0][3]  # Get mouth height
                if mouth_height > 20:  # Mouth too open = yawning
                    yawn_count += 1
                    cv2.putText(frame, f"YAWNING ({yawn_count}/{YAWN_FRAMES})", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    yawn_count = 0
            else:
                yawn_count = 0
        
        # Alarm control
        if (drowsy_count >= DROWSY_FRAMES or yawn_count >= YAWN_FRAMES) and cooldown == 0:
            if not alarm_on:
                alarm_on = True
                play_alarm()
                cooldown = COOLDOWN_FRAMES
                if drowsy_count >= DROWSY_FRAMES:
                    print("🔔 ALARM TRIGGERED: DROWSINESS DETECTED")
                else:
                    print("🔔 ALARM TRIGGERED: YAWNING DETECTED")
        
        if cooldown > 0:
            cooldown -= 1
        
        if drowsy_count == 0 and yawn_count == 0 and alarm_on:
            alarm_on = False
            stop_alarm()
        
        # Display status
        status = "🔔 ALARM ON" if alarm_on else "✅ NORMAL"
        cv2.putText(frame, status, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if alarm_on else (0, 255, 0), 2)
        
        cv2.imshow("Drowsiness Detection", frame)
        
        if frame_num % 30 == 0:
            print(f"Frame {frame_num} | Drowsy: {drowsy_count} | Yawn: {yawn_count} | Alarm: {'ON' if alarm_on else 'OFF'}")
        
        frame_num += 1
        
        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n✅ Exiting...")
            break

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n🔧 Cleaning up...")
    if not demo_mode and cap:
        cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    print("✅ Done!")
