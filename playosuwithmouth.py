#Cool vibecoded stuff, you play osu by opening your mouth. 
#F6/F7 to change sensitivity, F8 to pause, F9 to quit.
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pynput import keyboard

# ---------------- settings ----------------
KEYS = ["z", "x"] # alternates on each mouth-open. Use ["z"] for just one key.
OPEN_TH = 0.25    # oppening gap
CLOSE_GAP = 0.2   # closing gap
CAM_INDEX = 0
# ------------------------------------------

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

LM_UP_LIP, LM_LOW_LIP = 13, 14
LM_MOUTH_L, LM_MOUTH_R = 78, 308


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print("Downloading face model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        sys.exit(f"Model download failed ({e}).\nDownload it manually from:\n{MODEL_URL}\n"
                 f"and save it as:\n{MODEL_PATH}")


class State:
    def __init__(self):
        self.enabled = True
        self.quit = False
        self.open_th = OPEN_TH


def main():
    ensure_model()
    kb = keyboard.Controller()
    st = State()

    def on_press(key):
        if key == keyboard.Key.f6:
            st.open_th = max(0.12, st.open_th - 0.02)
        elif key == keyboard.Key.f7:
            st.open_th = min(0.80, st.open_th + 0.02)
        elif key == keyboard.Key.f8:
            st.enabled = not st.enabled
        elif key == keyboard.Key.f9:
            st.quit = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1))
    t_start = time.perf_counter()
    last_ts = -1

    mouth_open = False
    held = None          # key currently held
    key_i = 0
    ratio = 0.0

    def release():
        nonlocal held
        if held is not None:
            kb.release(held)
            held = None

    print("F6/F7 sensitivity | F8 pause | F9 quit")

    try:
        while not st.quit:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int((time.perf_counter() - t_start) * 1000), last_ts + 1)
            last_ts = ts
            res = face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            face_ok = bool(res.face_landmarks)

            if face_ok:
                lm = res.face_landmarks[0]
                mw = math.hypot(lm[LM_MOUTH_L].x - lm[LM_MOUTH_R].x, lm[LM_MOUTH_L].y - lm[LM_MOUTH_R].y)
                mh = math.hypot(lm[LM_UP_LIP].x - lm[LM_LOW_LIP].x, lm[LM_UP_LIP].y - lm[LM_LOW_LIP].y)
                ratio = mh / mw if mw > 1e-6 else 0.0

                if not mouth_open and ratio > st.open_th:
                    mouth_open = True
                    if st.enabled:
                        held = KEYS[key_i % len(KEYS)]
                        key_i += 1
                        kb.press(held)
                elif mouth_open and ratio < st.open_th - CLOSE_GAP:
                    mouth_open = False
                    release()
            else:
                ratio = 0.0
                mouth_open = False
                release()

            if not st.enabled:
                release()

            # ---- small preview ----
            view = frame.copy()
            h, w = view.shape[:2]
            status = "ACTIVE" if st.enabled else "PAUSED (F8)"
            col = (120, 255, 140) if st.enabled else (80, 80, 255)
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            if not face_ok:
                cv2.putText(view, "NO FACE", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
            bx, by, bw, bh = 10, h - 30, 220, 14
            cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (90, 90, 90), 1)
            cv2.rectangle(view, (bx, by), (bx + int(bw * min(ratio / 0.8, 1.0)), by + bh),
                          (120, 255, 140) if mouth_open else (210, 210, 210), -1)
            tx = bx + int(bw * st.open_th / 0.8)
            cv2.line(view, (tx, by - 4), (tx, by + bh + 4), (60, 60, 255), 2)
            if held:
                cv2.putText(view, held.upper(), (w - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow("osu mouth keys (F9 quit)", view)
            cv2.waitKey(1)
    finally:
        release()
        listener.stop()
        cap.release()
        face.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()