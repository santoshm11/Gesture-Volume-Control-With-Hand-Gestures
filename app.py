from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

# ---------------- AUDIO SETUP ----------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

# ---------------- HAND TRACKING ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ---------------- VARIABLES ----------------
current_volume = 0
mute = False
freeze = False

PINCH_DISTANCE = 25
PINCH_TIME = 2
pinch_start = None

volume_smooth = deque(maxlen=5)
volume_graph = deque(maxlen=120)
start_time = time.time()

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global current_volume, mute, freeze, pinch_start

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                lm = hand.landmark
                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

                dist = math.hypot(x2 - x1, y2 - y1)

                # -------- PINCH TO MUTE --------
                if dist < PINCH_DISTANCE:
                    if pinch_start is None:
                        pinch_start = time.time()
                    elif time.time() - pinch_start > PINCH_TIME:
                        mute = not mute
                        pinch_start = None
                else:
                    pinch_start = None

                # -------- VOLUME CONTROL --------
                if not mute and not freeze:
                    raw_vol = np.interp(dist, [20, 180], [vol_min, vol_max])
                    volume_smooth.append(raw_vol)
                    smooth_vol = sum(volume_smooth) / len(volume_smooth)
                    volume.SetMasterVolumeLevel(smooth_vol, None)

                    current_volume = int(
                        np.interp(smooth_vol, [vol_min, vol_max], [0, 100])
                    )

                    volume_graph.append({
                        "time": round(time.time() - start_time, 2),
                        "volume": current_volume
                    })

                # Drawing
                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # -------- UI TEXT --------
        status = "MUTED" if mute else "FROZEN" if freeze else "ACTIVE"
        color = (0, 0, 255) if mute else (255, 255, 0) if freeze else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Volume: {current_volume}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_freeze', methods=['POST'])
def toggle_freeze():
    global freeze
    freeze = not freeze
    return jsonify({"freeze": freeze})

@app.route('/status')
def status():
    return jsonify({
        "volume": 0 if mute else current_volume,
        "mute": mute,
        "freeze": freeze
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
