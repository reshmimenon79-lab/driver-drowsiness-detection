"""
Real-time drowsiness detection with dlib landmarks and a trained CNN.
"""

import json
import os
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, InputLayer, MaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json

try:
    import dlib
except ModuleNotFoundError:
    dlib = None


class Cfg:
    MODEL_PATH = "mobilenetv2_drowsy.h5"
    MODEL_JSON = "model.json"
    MODEL_WEIGHTS = "model.weights.h5"
    IMG_SIZE = (64, 64)
    CNN_THRESH = 0.60
    CNN_FRAMES = 8

    EAR_THRESH = 0.26
    EAR_FRAMES = 4
    EAR_SMOOTH = 5
    EAR_BASELINE_ALPHA = 0.02
    EAR_DYNAMIC_RATIO = 0.88
    EAR_DYNAMIC_MIN = 0.22
    EAR_DYNAMIC_MAX = 0.34

    MAR_THRESH = 0.72
    MAR_SMOOTH = 3
    MAR_BASELINE_ALPHA = 0.03
    MAR_DYNAMIC_DELTA = 0.18
    MAR_DYNAMIC_MIN = 0.68
    MAR_DYNAMIC_MAX = 0.95
    YAWN_SECONDS = 1.3
    FPS = 15
    YAWN_FRAMES = int(YAWN_SECONDS * FPS)
    MIN_FACE_SIZE = 40

    ALARM_FILE = "alarm.wav"
    COOLDOWN_SEC = 10.0
    NORMAL_SEC = 2.0
    NORMAL_FRAMES = int(NORMAL_SEC * FPS)


class Alarm:
    def __init__(self):
        self._ok = False
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._ok = os.path.exists(Cfg.ALARM_FILE)
            if self._ok:
                pygame.mixer.music.load(Cfg.ALARM_FILE)
            print(f"Alarm ready. File found: {self._ok}")
        except Exception as exc:
            print(f"Pygame init failed: {exc}")

        self._cooling_until = 0.0
        self._fired = False

    def trigger(self):
        now = time.time()
        if self._fired or now < self._cooling_until:
            return
        self._fired = True
        self._cooling_until = now + Cfg.COOLDOWN_SEC
        if self._ok:
            try:
                pygame.mixer.music.play()
                print(f"[{self._ts()}] ALARM fired, cooldown {Cfg.COOLDOWN_SEC:.0f}s")
            except Exception as exc:
                print(f"Alarm play error: {exc}")

    def notify_normal(self, normal_frame_count):
        if self._fired and time.time() >= self._cooling_until and normal_frame_count >= Cfg.NORMAL_FRAMES:
            self._fired = False
            print(f"[{self._ts()}] Alarm re-armed")

    def stop(self):
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def cooldown_remaining(self):
        return max(0.0, self._cooling_until - time.time())

    def is_fired(self):
        return self._fired

    def quit(self):
        try:
            self.stop()
            pygame.mixer.quit()
        except Exception:
            pass

    @staticmethod
    def _ts():
        return time.strftime("%H:%M:%S")


def rebuild_model_from_json(json_string, weights_path):
    try:
        try:
            model = model_from_json(json_string)
            model.load_weights(weights_path)
            return model
        except Exception:
            pass

        config = json.loads(json_string)
        layers_config = config.get("config", {}).get("layers", [])
        model = Sequential(name=config.get("config", {}).get("name", "model"))

        for layer_config in layers_config:
            class_name = layer_config.get("class_name")
            cfg = layer_config.get("config", {})
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
                model.add(Dense(units=cfg.get("units"), activation=cfg.get("activation"), name=cfg.get("name")))
            elif class_name == "Dropout":
                model.add(Dropout(rate=cfg.get("rate"), name=cfg.get("name")))

        input_shape = config.get("config", {}).get("build_input_shape", [1, 64, 64, 3])
        model.build(input_shape)
        model.load_weights(weights_path)
        return model
    except Exception as exc:
        print(f"Could not rebuild model from JSON: {exc}")
        return None


def resolve_input_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 3 and shape[1] and shape[2]:
            return (int(shape[2]), int(shape[1]))
    except Exception:
        pass
    return Cfg.IMG_SIZE


def build_model():
    if os.path.exists(Cfg.MODEL_PATH):
        try:
            model = load_model(Cfg.MODEL_PATH)
            print(f"Loaded model from {Cfg.MODEL_PATH}")
            Cfg.IMG_SIZE = resolve_input_size(model)
            return model
        except Exception as exc:
            print(f"Could not load {Cfg.MODEL_PATH}: {exc}")

    if os.path.exists(Cfg.MODEL_JSON) and os.path.exists(Cfg.MODEL_WEIGHTS):
        with open(Cfg.MODEL_JSON, "r", encoding="utf-8") as handle:
            json_string = handle.read()
        model = rebuild_model_from_json(json_string, Cfg.MODEL_WEIGHTS)
        if model is not None:
            Cfg.IMG_SIZE = resolve_input_size(model)
            print(f"Loaded model from {Cfg.MODEL_JSON} + {Cfg.MODEL_WEIGHTS}")
            return model

    print("Falling back to an untrained MobileNetV2 model. Accuracy will be poor.")
    base = MobileNetV2(input_shape=(*Cfg.IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out)


def preprocess(bgr_roi):
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, Cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    return np.expand_dims(norm, 0)


def predict_drowsy_score(model, roi):
    pred = model.predict(preprocess(roi), verbose=0)[0]
    if np.isscalar(pred):
        return float(pred)
    pred = np.asarray(pred).reshape(-1)
    if pred.size == 1:
        return float(pred[0])
    return float(pred[-1])


def calc_ear(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.3


def calc_mar(mouth):
    v1 = distance.euclidean(mouth[13], mouth[19])
    v2 = distance.euclidean(mouth[14], mouth[18])
    v3 = distance.euclidean(mouth[15], mouth[17])
    vert = (v1 + v2 + v3) / 3.0
    horiz = distance.euclidean(mouth[0], mouth[6])
    return float(np.clip(vert / horiz, 0.0, 1.5)) if horiz > 0 else 0.0


def _camera_candidates(max_index=3):
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
    indexes, backends = _camera_candidates()
    backend_names = {
        None: "default",
        getattr(cv2, "CAP_DSHOW", None): "CAP_DSHOW",
        getattr(cv2, "CAP_MSMF", None): "CAP_MSMF",
        getattr(cv2, "CAP_VFW", None): "CAP_VFW",
        getattr(cv2, "CAP_ANY", None): "CAP_ANY",
    }

    for idx in indexes:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, Cfg.FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.6)
                for _ in range(5):
                    cap.read()
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0 and frame.mean() > 8:
                    backend_name = backend_names.get(backend, str(backend))
                    print(f"Camera opened at index {idx} via {backend_name}")
                    return cap
                cap.release()
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
    return None


def run_no_dlib_fallback():
    fallback_path = os.path.join(os.path.dirname(__file__), "detect_no_dlib.py")
    if not os.path.exists(fallback_path):
        print("dlib is not installed, and detect_no_dlib.py was not found.")
        print("Install dlib or keep detect_no_dlib.py in the same folder.")
        sys.exit(1)

    print("dlib is not installed. Switching to detect_no_dlib.py...")
    try:
        subprocess.run([sys.executable, fallback_path], check=False)
    finally:
        sys.exit(0)


FONT = cv2.FONT_HERSHEY_SIMPLEX


def bar(img, x, y, w, h, ratio, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.putText(img, label, (x + w + 6, y + h - 1), FONT, 0.42, (180, 180, 180), 1)


def draw_hud(frame, ear, ear_thresh, mar, mar_thresh, cnn, eye_cnt, cnn_cnt, yawn_cnt, normal_cnt, status, alarm):
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color = (0, 0, 220) if status == "DROWSY" else (0, 130, 255) if status == "YAWNING" else (0, 200, 60)
    cv2.rectangle(frame, (10, 10), (300, 56), color, -1)
    cv2.putText(frame, status, (18, 46), FONT, 1.1, (255, 255, 255), 2)

    if alarm.is_fired():
        text = f"COOLDOWN {alarm.cooldown_remaining():.1f}s"
        text_color = (0, 100, 200)
    else:
        text = "ARMED - ready"
        text_color = (0, 190, 60)
    cv2.putText(frame, text, (14, 74), FONT, 0.50, text_color, 1)

    cv2.putText(frame, f"EAR: {ear:.3f} thr={ear_thresh:.3f}", (14, 102), FONT, 0.52, (220, 220, 220), 1)
    cv2.putText(frame, f"MAR: {mar:.3f} thr={mar_thresh:.3f}", (14, 122), FONT, 0.52, (220, 220, 220), 1)
    cv2.putText(frame, f"CNN: {cnn:.3f} thr={Cfg.CNN_THRESH}", (14, 142), FONT, 0.52, (220, 220, 220), 1)

    bar(frame, 14, 158, 170, 11, min(1.0, eye_cnt / Cfg.EAR_FRAMES), (0, 0, 200), f"Eyes {eye_cnt}/{Cfg.EAR_FRAMES}")
    bar(frame, 14, 176, 170, 11, min(1.0, yawn_cnt / Cfg.YAWN_FRAMES), (0, 130, 255), f"Yawn {yawn_cnt}/{Cfg.YAWN_FRAMES}")
    bar(frame, 14, 194, 170, 11, min(1.0, cnn_cnt / Cfg.CNN_FRAMES), (180, 0, 180), f"CNN {cnn_cnt}/{Cfg.CNN_FRAMES}")
    bar(frame, 14, 212, 170, 11, min(1.0, normal_cnt / Cfg.NORMAL_FRAMES), (0, 200, 60), f"Normal {normal_cnt}/{Cfg.NORMAL_FRAMES}")
    cv2.putText(frame, "ESC = quit", (14, h - 12), FONT, 0.40, (120, 120, 120), 1)


def main():
    print("\n=== Drowsiness Detection System ===")

    if dlib is None:
        run_no_dlib_fallback()

    model = build_model()

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("dlib loaded.")
    except Exception as exc:
        print(f"dlib error: {exc}")
        sys.exit(1)

    alarm = Alarm()
    cap = open_camera()
    if cap is None:
        print("No camera found.")
        alarm.quit()
        sys.exit(1)

    print("Warming camera...")
    for _ in range(20):
        cap.read()
        time.sleep(0.05)
    print("Ready. Press ESC to quit.\n")

    ear_buf = deque(maxlen=Cfg.EAR_SMOOTH)
    mar_buf = deque(maxlen=Cfg.MAR_SMOOTH)
    eye_closed_cnt = 0
    cnn_drowsy_cnt = 0
    yawn_cnt = 0
    normal_cnt = 0
    status = "NORMAL"
    errors = 0
    adaptive_ear_thresh = Cfg.EAR_THRESH
    adaptive_mar_thresh = Cfg.MAR_THRESH
    ear_baseline = None
    mar_baseline = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            errors += 1
            if errors > 10:
                break
            time.sleep(0.1)
            continue
        errors = 0

        display = frame.copy()
        gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8))

        try:
            faces = detector(gray, 0)
        except Exception:
            faces = []

        ear_val = 0.30
        mar_val = 0.00
        cnn_val = 0.00

        if not faces:
            eye_closed_cnt = max(0, eye_closed_cnt - 1)
            cnn_drowsy_cnt = max(0, cnn_drowsy_cnt - 1)
            yawn_cnt = max(0, yawn_cnt - 2)
            if eye_closed_cnt == 0 and cnn_drowsy_cnt == 0 and yawn_cnt == 0:
                normal_cnt += 1
                status = "NORMAL"
                alarm.stop()
                alarm.notify_normal(normal_cnt)
            cv2.putText(display, "No face", (320, 30), FONT, 0.8, (140, 140, 140), 1)
        else:
            face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
            fw = face.right() - face.left()
            fh = face.bottom() - face.top()

            if fw >= Cfg.MIN_FACE_SIZE and fh >= Cfg.MIN_FACE_SIZE:
                shape = face_utils.shape_to_np(predictor(gray, face))
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                mouth = shape[48:68]

                raw_ear = (calc_ear(left_eye) + calc_ear(right_eye)) / 2.0
                ear_buf.append(float(np.clip(raw_ear, 0.05, 0.50)))
                ear_val = float(np.mean(ear_buf))

                raw_mar = calc_mar(mouth)
                mar_buf.append(raw_mar)
                mar_val = float(np.mean(mar_buf))

                x1 = max(0, face.left())
                y1 = max(0, face.top())
                x2 = min(frame.shape[1], face.right())
                y2 = min(frame.shape[0], face.bottom())
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    try:
                        cnn_val = predict_drowsy_score(model, roi)
                    except Exception as exc:
                        print(f"Model prediction error: {exc}")

                if ear_baseline is None:
                    ear_baseline = ear_val
                elif ear_val > adaptive_ear_thresh:
                    # Learn the user's natural open-eye EAR slowly so detection
                    # stays usable across different faces and camera angles.
                    ear_baseline = (
                        (1.0 - Cfg.EAR_BASELINE_ALPHA) * ear_baseline
                        + Cfg.EAR_BASELINE_ALPHA * ear_val
                    )

                adaptive_ear_thresh = float(
                    np.clip(
                        max(Cfg.EAR_THRESH, ear_baseline * Cfg.EAR_DYNAMIC_RATIO),
                        Cfg.EAR_DYNAMIC_MIN,
                        Cfg.EAR_DYNAMIC_MAX,
                    )
                )

                if mar_baseline is None:
                    mar_baseline = mar_val
                elif mar_val < adaptive_mar_thresh:
                    mar_baseline = (
                        (1.0 - Cfg.MAR_BASELINE_ALPHA) * mar_baseline
                        + Cfg.MAR_BASELINE_ALPHA * mar_val
                    )

                adaptive_mar_thresh = float(
                    np.clip(
                        max(Cfg.MAR_THRESH, mar_baseline + Cfg.MAR_DYNAMIC_DELTA),
                        Cfg.MAR_DYNAMIC_MIN,
                        Cfg.MAR_DYNAMIC_MAX,
                    )
                )

                eye_alert = ear_val < adaptive_ear_thresh
                cnn_alert = cnn_val >= Cfg.CNN_THRESH
                mouth_alert = mar_val >= adaptive_mar_thresh and ear_val > adaptive_ear_thresh * 0.95

                if eye_alert:
                    eye_closed_cnt += 1
                    normal_cnt = 0
                else:
                    eye_closed_cnt = max(0, eye_closed_cnt - 1)

                if cnn_alert:
                    cnn_drowsy_cnt += 1
                else:
                    cnn_drowsy_cnt = max(0, cnn_drowsy_cnt - 1)

                if mouth_alert:
                    yawn_cnt += 1
                    normal_cnt = 0
                else:
                    yawn_cnt = max(0, yawn_cnt - 2)

                drowsy_confirmed = (
                    eye_closed_cnt >= Cfg.EAR_FRAMES
                    or cnn_drowsy_cnt >= Cfg.CNN_FRAMES
                    or (eye_closed_cnt >= max(2, Cfg.EAR_FRAMES // 2) and cnn_drowsy_cnt >= max(2, Cfg.CNN_FRAMES // 2))
                )
                yawn_confirmed = yawn_cnt >= Cfg.YAWN_FRAMES and eye_closed_cnt < max(2, Cfg.EAR_FRAMES // 2) and cnn_drowsy_cnt < max(2, Cfg.CNN_FRAMES // 2)

                if drowsy_confirmed:
                    status = "DROWSY"
                elif yawn_confirmed:
                    status = "YAWNING"
                else:
                    status = "NORMAL"
                    normal_cnt += 1
                    alarm.stop()
                    alarm.notify_normal(normal_cnt)

                if status in ("DROWSY", "YAWNING"):
                    alarm.trigger()
                else:
                    alarm.stop()
                    alarm.notify_normal(normal_cnt)

                box_color = (0, 0, 220) if status == "DROWSY" else (0, 130, 255) if status == "YAWNING" else (0, 200, 60)
                cv2.rectangle(display, (face.left(), face.top()), (face.right(), face.bottom()), box_color, 2)
                for (x, y) in np.concatenate([left_eye, right_eye]):
                    cv2.circle(display, (x, y), 2, (0, 230, 230), -1)
                for (x, y) in mouth:
                    cv2.circle(display, (x, y), 2, (255, 180, 0), -1)
            else:
                eye_closed_cnt = max(0, eye_closed_cnt - 1)
                cnn_drowsy_cnt = max(0, cnn_drowsy_cnt - 1)
                yawn_cnt = max(0, yawn_cnt - 1)
                if eye_closed_cnt == 0 and cnn_drowsy_cnt == 0 and yawn_cnt == 0:
                    normal_cnt += 1
                    status = "NORMAL"
                    alarm.stop()
                    alarm.notify_normal(normal_cnt)
                cv2.putText(display, "Face too small", (300, 30), FONT, 0.7, (140, 140, 140), 1)

        draw_hud(
            display,
            ear_val,
            adaptive_ear_thresh,
            mar_val,
            adaptive_mar_thresh,
            cnn_val,
            eye_closed_cnt,
            cnn_drowsy_cnt,
            yawn_cnt,
            normal_cnt,
            status,
            alarm,
        )
        cv2.imshow("Drowsiness Monitor", display)

        if int(time.time() * Cfg.FPS) % 15 == 0:
            print(
                f"EAR:{ear_val:.3f} MAR:{mar_val:.3f} CNN:{cnn_val:.3f} | "
                f"E:{eye_closed_cnt}/{Cfg.EAR_FRAMES} "
                f"Y:{yawn_cnt}/{Cfg.YAWN_FRAMES} "
                f"C:{cnn_drowsy_cnt}/{Cfg.CNN_FRAMES} "
                f"N:{normal_cnt} | {status} | "
                f"fired={alarm.is_fired()} cd={alarm.cooldown_remaining():.1f}s"
            )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    alarm.quit()
    print("Done.")


if __name__ == "__main__":
    main()
