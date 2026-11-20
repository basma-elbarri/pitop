"""Camera, nearest-person detection, head tracking and follow behaviour.

Detection (best available is used automatically):
  1. YuNet face detector (fast, accurate)  - needs face_detection_yunet_2023mar.onnx next to this file
  2. HOG whole-body person detector        - built into OpenCV, works when the face isn't visible
  3. Haar face detector                    - old fallback if YuNet isn't available

Modes:
  manual - phone controls everything, camera just streams
  look   - head (pan/tilt servos) tracks the nearest person, wheels stay still
  follow - head tracks the nearest person AND the robot drives up to them
"""
import os
import threading
import time

import cv2

# ------------------------------ CONFIG ------------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FLIP_IMAGE = False            # True if the camera is mounted upside down

DETECT_W, DETECT_H = 320, 240 # detection runs on this smaller image for speed
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_SCORE = 0.6             # lower = detects more, but more false alarms
HOG_EVERY = 3                 # run the (slow) body detector every Nth frame

SMOOTHING = 0.5               # 0 = no smoothing, closer to 1 = smoother but laggier
DEADZONE = 0.06               # ignore tiny errors (fraction of half-frame)
PAN_RATE = 90.0               # deg/s head speed at full horizontal error
TILT_RATE = 50.0              # deg/s head speed at full vertical error
BODY_TURN_GAIN = 1 / 35.0     # body turn per degree of head pan
ALIGN_ANGLE = 25.0            # only drive forward when head is within this angle
TARGET_FACE_FRACTION = 0.25   # face width / frame width at stopping distance (~0.6 m)
TARGET_BODY_FRACTION = 0.75   # body height / frame height at stopping distance
MAX_FOLLOW_SPEED = 0.8        # 0..1 of the allowed speed
LOST_HOLD = 1.0               # seconds to wait before searching
SEARCH_SPEED = 30.0           # deg/s head sweep while searching
SEARCH_LIMIT = 60.0
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _find_cascade():
    """Locate haarcascade_frontalface_default.xml (Debian's OpenCV has no cv2.data)."""
    name = "haarcascade_frontalface_default.xml"
    candidates = []
    data = getattr(cv2, "data", None)
    if data is not None:
        candidates.append(data.haarcascades + name)
    candidates += [
        "/usr/share/opencv4/haarcascades/" + name,
        "/usr/share/opencv/haarcascades/" + name,
        "/usr/local/share/opencv4/haarcascades/" + name,
        os.path.join(HERE, name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class Follower:
    MODES = ("manual", "look", "follow")

    def __init__(self, robot):
        self.robot = robot
        self.mode = "manual"
        self.face_seen = False
        self.face_fraction = 0.0     # "closeness": 1.0 = at stopping distance
        self.target_kind = None      # "face" or "body"

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # --- detectors ---
        self.yunet = None
        model = os.path.join(HERE, YUNET_FILE)
        if os.path.exists(model) and hasattr(cv2, "FaceDetectorYN"):
            create = getattr(cv2.FaceDetectorYN, "create", None) or cv2.FaceDetectorYN_create
            self.yunet = create(model, "", (DETECT_W, DETECT_H), YUNET_SCORE, 0.3, 5000)
            print("Face detector: YuNet")

        self.cascade = None
        if self.yunet is None:
            path = _find_cascade()
            if path:
                self.cascade = cv2.CascadeClassifier(path)
                print("Face detector: Haar (weak). Download the YuNet model for better results.")

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._frame_no = 0
        self._last_body = None
        self._ex = 0.0
        self._ey = 0.0
        self._cond = threading.Condition()
        self._jpeg = None
        self._last_seen = 0.0
        self._search_dir = 1
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------ API -----------------------------------
    def set_mode(self, mode):
        if mode not in self.MODES:
            return False
        if mode == "manual":
            self.robot.stop()
        self._last_seen = time.time()
        self.mode = mode
        return True

    def status(self):
        return {
            "mode": self.mode,
            "face": self.face_seen,
            "face_size": round(self.face_fraction, 2),
            "target": self.target_kind,
            "front": self.robot.front,
            "back": self.robot.back,
            "pan": round(self.robot.pan),
            "tilt": round(self.robot.tilt),
        }

    def mjpeg(self):
        while True:
            with self._cond:
                self._cond.wait(timeout=1.0)
                jpeg = self._jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    # ---------------------------- detection --------------------------------
    def _faces(self, frame, w, h):
        """Return the largest face as (x, y, w, h) in full-frame pixels, or None."""
        sx, sy = w / DETECT_W, h / DETECT_H
        small = cv2.resize(frame, (DETECT_W, DETECT_H))
        if self.yunet is not None:
            self.yunet.setInputSize((DETECT_W, DETECT_H))
            _, faces = self.yunet.detect(small)
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda f: f[2] * f[3])
            return int(f[0] * sx), int(f[1] * sy), int(f[2] * sx), int(f[3] * sy)
        if self.cascade is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = self.cascade.detectMultiScale(gray, 1.15, 4, minSize=(20, 20))
            if len(faces) == 0:
                return None
            f = max(faces, key=lambda f: f[2] * f[3])
            return int(f[0] * sx), int(f[1] * sy), int(f[2] * sx), int(f[3] * sy)
        return None

    def _bodies(self, frame, w, h):
        """Return the largest whole-body person as (x, y, w, h), or None."""
        sx, sy = w / DETECT_W, h / DETECT_H
        small = cv2.resize(frame, (DETECT_W, DETECT_H))
        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        best, best_area = None, 0
        for (x, y, bw, bh), wt in zip(rects, weights):
            if float(wt) < 0.4:
                continue
            if bw * bh > best_area:
                best, best_area = (x, y, bw, bh), bw * bh
        if best is None:
            return None
        x, y, bw, bh = best
        return int(x * sx), int(y * sy), int(bw * sx), int(bh * sy)

    def _detect(self, frame):
        """Find the nearest person.  Returns dict(kind, box, cx, cy, closeness) or None."""
        h, w = frame.shape[:2]
        face = self._faces(frame, w, h)
        if face is not None:
            x, y, bw, bh = face
            return dict(kind="face", box=face, cx=x + bw / 2, cy=y + bh / 2,
                        closeness=(bw / w) / TARGET_FACE_FRACTION, w=w, h=h)
        if self._frame_no % HOG_EVERY == 0:
            self._last_body = self._bodies(frame, w, h)
        body = self._last_body
        if body is not None:
            x, y, bw, bh = body
            return dict(kind="body", box=body, cx=x + bw / 2, cy=y + 0.2 * bh,
                        closeness=(bh / h) / TARGET_BODY_FRACTION, w=w, h=h)
        return None

    # ---------------------------- behaviour --------------------------------
    def _act(self, mode, target, dt, now):
        r = self.robot
        if target is not None:
            w, h = target["w"], target["h"]
            ex = (target["cx"] - w / 2) / (w / 2)    # + = person right of center
            ey = (target["cy"] - h / 2) / (h / 2)    # + = person below center
            if not self.face_seen:                   # just (re)acquired: no smoothing lag
                self._ex, self._ey = ex, ey
            self._ex = SMOOTHING * self._ex + (1 - SMOOTHING) * ex
            self._ey = SMOOTHING * self._ey + (1 - SMOOTHING) * ey

            self.face_seen = True
            self.target_kind = target["kind"]
            self.face_fraction = target["closeness"]
            self._last_seen = now

            pan, tilt = r.pan, r.tilt
            if abs(self._ex) > DEADZONE:
                pan += PAN_RATE * self._ex * dt
            if abs(self._ey) > DEADZONE:
                tilt -= TILT_RATE * self._ey * dt
            r.look(pan, tilt)

            if mode == "follow":
                turn = 0.0 if abs(r.pan) < 5 else _clamp(r.pan * BODY_TURN_GAIN, -1, 1)
                fwd = 0.0
                if abs(r.pan) < ALIGN_ANGLE:
                    err = 1.0 - self.face_fraction   # 0 when at the stopping distance
                    fwd = _clamp(err * 1.5, 0, MAX_FOLLOW_SPEED)
                r.set_drive(fwd, turn)
        else:
            self.face_seen = False
            self.target_kind = None
            self.face_fraction = 0.0
            if mode == "follow":
                r.set_drive(0, 0)
            if now - self._last_seen > LOST_HOLD:      # sweep head to search
                new = r.pan + self._search_dir * SEARCH_SPEED * dt
                if abs(new) >= SEARCH_LIMIT:
                    self._search_dir *= -1
                r.look(new, r.tilt * 0.9)

    def _annotate(self, frame, target, mode):
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), (255, 255, 255), 1)
        cv2.line(frame, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), (255, 255, 255), 1)
        if target is not None:
            x, y, bw, bh = target["box"]
            color = (0, 255, 0) if target["kind"] == "face" else (255, 200, 0)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, target["kind"], (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        front = self.robot.front
        label = f"{mode}  front={front if front is not None else '--'}m"
        cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _loop(self):
        last = time.time()
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            if FLIP_IMAGE:
                frame = cv2.flip(frame, -1)
            self._frame_no += 1
            now = time.time()
            dt = min(now - last, 0.2)
            last = now

            mode = self.mode
            target = None
            if mode != "manual":
                try:
                    target = self._detect(frame)
                except Exception as e:   # never let a detector error kill the loop
                    print("detect error:", e)
                self._act(mode, target, dt, now)
            self._annotate(frame, target, mode)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._cond:
                    self._jpeg = buf.tobytes()
                    self._cond.notify_all()

    def close(self):
        self._running = False
        self.cap.release()
