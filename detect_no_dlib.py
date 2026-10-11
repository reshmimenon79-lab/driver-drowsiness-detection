import json
import os
import sys
import time

import cv2
import numpy as np
import pygame
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from tensorflow.keras.models import Sequential, model_from_json


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def clean_keras_config(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "quantization_config":
                continue
            cleaned[key] = clean_keras_config(item)
        return cleaned
    if isinstance(value, list):
        return [clean_keras_config(item) for item in value]
    return value


def rebuild_model_from_json(json_string, weights_path):
    try:
        config = clean_keras_config(json.loads(json_string))
        clean_json = json.dumps(config)

        try:
            model = model_from_json(clean_json)
            model.load_weights(weights_path)
            return model
        except Exception as exc:
            print(f"[WARN] Direct JSON loading failed: {str(exc)[:120]}")
            print("Attempting alternative loading method...")

        layers_config = config.get("config", {}).get("layers", [])
        model = Sequential(name=config.get("config", {}).get("name", "model"))

        for layer_config in layers_config:
            class_name = layer_config.get("class_name")
            cfg = clean_keras_config(layer_config.get("config", {}))
            if class_name == "InputLayer":
                shape = cfg.get("batch_input_shape", cfg.get("batch_shape"))[1:]
                model.add(InputLayer(input_shape=shape, name=cfg.get("name")))
            elif class_name == "Conv2D":
                model.add(
                    Conv2D(
                        filters=cfg.get("filters"),
                        kernel_size=cfg.get("kernel_size"),
                        strides=cfg.get("strides", [1, 1]),
                        padding=cfg.get("padding", "valid"),
                        activation=cfg.get("activation"),
                        name=cfg.get("name"),
                    )
                )
            elif class_name == "MaxPooling2D":
                model.add(
                    MaxPooling2D(
                        pool_size=cfg.get("pool_size", [2, 2]),
                        strides=cfg.get("strides", [2, 2]),
                        padding=cfg.get("padding", "valid"),
                        name=cfg.get("name"),
                    )
                )
            elif class_name == "Flatten":
                model.add(Flatten(name=cfg.get("name")))
            elif class_name == "Dense":
                model.add(
                    Dense(
                        units=cfg.get("units"),
                        activation=cfg.get("activation"),
                        name=cfg.get("name"),
                    )
                )
            elif class_name == "Dropout":
                model.add(Dropout(rate=cfg.get("rate"), name=cfg.get("name")))

        input_shape = config.get("config", {}).get("build_input_shape", [1, 64, 64, 3])
        model.build(input_shape)
        model.load_weights(weights_path)
        return model
    except Exception as exc:
        print(f"[ERROR] Model rebuilding failed: {exc}")
        return None


def resolve_input_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 4 and shape[1] and shape[2]:
            return (int(shape[2]), int(shape[1]))
    except Exception:
        pass
    return (224, 224)


try:
    with open("model.json", "r", encoding="utf-8") as json_file:
        model_json = json_file.read()
    model = rebuild_model_from_json(model_json, "model.weights.h5")
    if model is None:
        print("[ERROR] Failed to load model.")
        sys.exit(1)
    MODEL_INPUT_SIZE = resolve_input_size(model)
    print(f"Model input size: {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}")
except FileNotFoundError as exc:
    print(f"[ERROR] Missing model files: {exc}")
    sys.exit(1)


try:
    pygame.mixer.init()
    alarm_available = os.path.exists("alarm.wav")
except Exception as exc:
    print(f"[WARN] Pygame mixer failed to initialize: {exc}")
    alarm_available = False


alarm_playing = False


def play_alarm():
    global alarm_playing
    try:
        if alarm_available and not alarm_playing:
            alarm_playing = True
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.play(-1)
            print("Alarm ON - drowsiness detected")
    except Exception as exc:
        print(f"[WARN] Error playing alarm: {exc}")
        alarm_playing = False


def stop_alarm():
    global alarm_playing
    try:
        if alarm_playing:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            alarm_playing = False
            print("Alarm OFF - alert cleared")
    except Exception as exc:
        print(f"[WARN] Error stopping alarm: {exc}")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


EAR_THRESHOLD = 0.22
DROWSY_FRAMES_THRESHOLD = 4
ALERT_COOLDOWN_FRAMES = 30
MODEL_DROWSY_THRESHOLD = 0.65
MISSING_EYES_THRESHOLD = 4
FACE_HOLD_FRAMES = 10


def calculate_ear_from_eyes(left_eye, right_eye):
    if left_eye is None or right_eye is None:
        return 0.50
    lx, ly, lw, lh = left_eye
    rx, ry, rw, rh = right_eye
    left_ear = lh / (lw + 1e-3)
    right_ear = rh / (rw + 1e-3)
    return (left_ear + right_ear) / 2.0


def is_frame_valid(frame):
    if frame is None or frame.size == 0:
        return False
    if frame.mean() < 10 or frame.mean() > 245:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= 30


def camera_candidates(max_index=3):
    forced_index = os.environ.get("DROWSY_CAMERA_INDEX")
    indexes = []
    if forced_index is not None:
        try:
            indexes.append(int(forced_index))
        except ValueError:
            print(f"Ignoring invalid DROWSY_CAMERA_INDEX={forced_index!r}")
    indexes.extend(idx for idx in range(max_index) if idx not in indexes)

    backends = [None]
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_VFW", "CAP_ANY"):
        backend = getattr(cv2, name, None)
        if backend is not None and backend not in backends:
            backends.append(backend)
    return indexes, backends


def open_camera():
    indexes, backends = camera_candidates()
    backend_names = {
        None: "default",
        getattr(cv2, "CAP_DSHOW", None): "CAP_DSHOW",
        getattr(cv2, "CAP_MSMF", None): "CAP_MSMF",
        getattr(cv2, "CAP_VFW", None): "CAP_VFW",
        getattr(cv2, "CAP_ANY", None): "CAP_ANY",
    }
    for index in indexes:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.5)
                for _ in range(5):
                    cap.read()
                ret, frame = cap.read()
                if ret and frame is not None and is_frame_valid(frame):
                    backend_name = backend_names.get(backend, str(backend))
                    print(f"Camera opened at index {index} via {backend_name}")
                    return cap
                cap.release()
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
                continue
    return None


def open_video_source():
    video_path = os.environ.get("DROWSY_VIDEO_PATH")
    if not video_path:
        return None, None
    if not os.path.exists(video_path):
        print(f"[WARN] DROWSY_VIDEO_PATH not found: {video_path}")
        return None, None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video file: {video_path}")
        return None, None
    print(f"Using video file: {video_path}")
    return cap, "video"


def make_demo_frame(loop_count, alarm_on):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (170, 90), (470, 390), (195, 195, 195), -1)
    cv2.circle(frame, (255, 200), 24, (90, 90, 90), -1)
    cv2.circle(frame, (385, 200), 24, (90, 90, 90), -1)
    mouth_height = 20 if not alarm_on else 45
    cv2.ellipse(frame, (320, 305), (70, mouth_height), 0, 0, 180, (80, 80, 80), -1)
    cv2.putText(frame, "DEMO MODE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    cv2.putText(frame, f"Loop: {loop_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return frame


print("Searching for available camera...")
cap = open_camera()
source_kind = "camera" if cap is not None else None
if cap is None:
    cap, source_kind = open_video_source()

demo_mode = False
if cap is None:
    print("[WARN] No camera found.")
    print("Tip: set DROWSY_CAMERA_INDEX to 0, 1, or 2 before running if your webcam uses a non-default index.")
    print("Tip: set DROWSY_VIDEO_PATH to a local video file to test detection without a webcam.")
    print("Falling back to demo mode.")
    demo_mode = True
else:
    print(f"{source_kind.capitalize()} initialized. Press ESC to exit.")

alarm_on = False
drowsy_frame_count = 0
alarm_cooldown_count = 0
missing_eyes_count = 0
loop_count = 0
last_face = None
face_hold_count = 0


try:
    while True:
        loop_count += 1
        if demo_mode:
            frame = make_demo_frame(loop_count, alarm_on)
            demo_drowsy = (loop_count // 45) % 2 == 1
            if demo_drowsy:
                drowsy_frame_count = min(DROWSY_FRAMES_THRESHOLD, drowsy_frame_count + 1)
            else:
                drowsy_frame_count = 0
            status = "DROWSY" if drowsy_frame_count >= DROWSY_FRAMES_THRESHOLD else "NORMAL"
            color = (0, 0, 255) if status == "DROWSY" else (0, 200, 0)
            if status == "DROWSY" and alarm_cooldown_count == 0:
                if not alarm_on:
                    alarm_on = True
                    play_alarm()
                    alarm_cooldown_count = ALERT_COOLDOWN_FRAMES
            elif status == "NORMAL" and alarm_on:
                alarm_on = False
                stop_alarm()
            if alarm_cooldown_count > 0:
                alarm_cooldown_count -= 1
            cv2.putText(frame, status, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, "No live camera/video source available", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.imshow("Drowsiness Detection (NO DLIB)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            if source_kind == "video":
                print("Video ended.")
                break
            continue
        if not is_frame_valid(frame):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(50, 50))

        if len(faces) == 0:
            if last_face is not None and face_hold_count < FACE_HOLD_FRAMES:
                x, y, w, h = last_face
                face_hold_count += 1
            else:
                last_face = None
                face_hold_count = 0
                missing_eyes_count = 0
                drowsy_frame_count = 0
                if alarm_on:
                    alarm_on = False
                    stop_alarm()
                cv2.putText(frame, "No face", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                cv2.imshow("Drowsiness Detection (NO DLIB)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
        else:
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            last_face = (x, y, w, h)
            face_hold_count = 0

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        upper_gray = roi_gray[: max(1, h // 2), :]
        eyes = eye_cascade.detectMultiScale(upper_gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))

        left_eye = None
        right_eye = None
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[-1]
            missing_eyes_count = 0
        else:
            missing_eyes_count += 1

        ear = calculate_ear_from_eyes(left_eye, right_eye)

        face_roi = frame[y:y + h, x:x + w]
        img = cv2.resize(face_roi, MODEL_INPUT_SIZE).astype("float32") / 255.0
        img = np.reshape(img, (1, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3))
        pred = float(model.predict(img, verbose=0)[0][0])

        eyes_missing_alert = missing_eyes_count >= MISSING_EYES_THRESHOLD
        drowsy = eyes_missing_alert or ear < EAR_THRESHOLD or pred > MODEL_DROWSY_THRESHOLD

        if drowsy:
            drowsy_frame_count += 1
        else:
            drowsy_frame_count = 0

        if drowsy_frame_count >= DROWSY_FRAMES_THRESHOLD and alarm_cooldown_count == 0:
            if not alarm_on:
                alarm_on = True
                play_alarm()
                alarm_cooldown_count = ALERT_COOLDOWN_FRAMES
        elif drowsy_frame_count == 0 and alarm_on:
            alarm_on = False
            stop_alarm()

        if alarm_cooldown_count > 0:
            alarm_cooldown_count -= 1

        status = "DROWSY" if drowsy_frame_count >= DROWSY_FRAMES_THRESHOLD else "NORMAL"
        color = (0, 0, 255) if status == "DROWSY" else (0, 200, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if left_eye is not None:
            ex, ey, ew, eh = left_eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        if right_eye is not None:
            ex, ey, ew, eh = right_eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"EAR: {ear:.2f} thr={EAR_THRESHOLD:.2f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(frame, f"CNN: {pred:.2f} thr={MODEL_DROWSY_THRESHOLD:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(frame, f"Eyes lost: {missing_eyes_count}/{MISSING_EYES_THRESHOLD}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(frame, f"Drowsy frames: {drowsy_frame_count}/{DROWSY_FRAMES_THRESHOLD}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        if face_hold_count > 0:
            cv2.putText(frame, f"Face hold: {face_hold_count}/{FACE_HOLD_FRAMES}", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

        print(
            f"EAR: {ear:.2f}, EyesFound: {2 if left_eye is not None and right_eye is not None else len(eyes)}, "
            f"MissingEyes: {missing_eyes_count}, FaceHold: {face_hold_count}, Pred: {pred:.2f}, "
            f"DrowsyFrames: {drowsy_frame_count}, Status: {status}, Alarm: {'ON' if alarm_on else 'OFF'}"
        )

        cv2.imshow("Drowsiness Detection (NO DLIB)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    print("Program closed successfully.")
