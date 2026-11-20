"""Camera, nearest-face detection, head tracking and follow behaviour.

Modes:
  manual - phone controls everything, camera just streams
  look   - head (pan/tilt servos) tracks the nearest face, wheels stay still
  follow - head tracks the nearest face AND the robot drives up to it
"""
import threading
import time

import cv2

# ------------------------------ CONFIG ------------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FLIP_IMAGE = False            # True if the camera is mounted upside down
DETECT_SCALE = 0.5            # detect on a smaller image for speed

DEADZONE = 0.08               # ignore tiny errors (fraction of half-frame)
PAN_RATE = 70.0               # deg/s head speed at full horizontal error
TILT_RATE = 45.0              # deg/s head speed at full vertical error
BODY_TURN_GAIN = 1 / 40.0     # body turn per degree of head pan
ALIGN_ANGLE = 20.0            # only drive forward when head is within this angle
TARGET_FACE_FRACTION = 0.25   # face width / frame width at stopping distance (~0.6 m)
MAX_FOLLOW_SPEED = 0.8        # 0..1 of the allowed speed
LOST_HOLD = 1.0               # seconds to wait before searching
SEARCH_SPEED = 30.0           # deg/s head sweep while searching
SEARCH_LIMIT = 60.0
# --------------------------------------------------------------------------


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Follower:
    MODES = ("manual", "look", "follow")

    def __init__(self, robot):
        self.robot = robot
        self.mode = "manual"
        self.face_seen = False
        self.face_fraction = 0.0

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

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

    # ---------------------------- vision ----------------------------------
    def _largest_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        small = cv2.equalizeHist(small)
        faces = self.cascade.detectMultiScale(
            small, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # biggest = nearest
        s = 1.0 / DETECT_SCALE
        return int(x * s), int(y * s), int(w * s), int(h * s)

    def _act(self, mode, face, dt, now):
        r = self.robot
        if face is not None:
            x, y, w, h = face
            ex = (x + w / 2 - FRAME_W / 2) / (FRAME_W / 2)   # + = face right of center
            ey = (y + h / 2 - FRAME_H / 2) / (FRAME_H / 2)   # + = face below center
            self.face_seen = True
            self.face_fraction = w / FRAME_W
            self._last_seen = now

            pan, tilt = r.pan, r.tilt
            if abs(ex) > DEADZONE:
                pan += PAN_RATE * ex * dt
            if abs(ey) > DEADZONE:
                tilt -= TILT_RATE * ey * dt
            r.look(pan, tilt)

            if mode == "follow":
                turn = 0.0 if abs(r.pan) < 5 else _clamp(r.pan * BODY_TURN_GAIN, -1, 1)
                fwd = 0.0
                if abs(r.pan) < ALIGN_ANGLE:
                    err = (TARGET_FACE_FRACTION - self.face_fraction) / TARGET_FACE_FRACTION
                    fwd = _clamp(err * 1.5, 0, MAX_FOLLOW_SPEED)
                r.set_drive(fwd, turn)
        else:
            self.face_seen = False
            self.face_fraction = 0.0
            if mode == "follow":
                r.set_drive(0, 0)
            if now - self._last_seen > LOST_HOLD:      # sweep head to search
                new = r.pan + self._search_dir * SEARCH_SPEED * dt
                if abs(new) >= SEARCH_LIMIT:
                    self._search_dir *= -1
                r.look(new, r.tilt * 0.9)

    def _annotate(self, frame, face, mode):
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), (255, 255, 255), 1)
        cv2.line(frame, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), (255, 255, 255), 1)
        if face is not None:
            x, y, fw, fh = face
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
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
            now = time.time()
            dt = min(now - last, 0.2)
            last = now

            mode = self.mode
            face = None
            if mode != "manual":
                face = self._largest_face(frame)
                self._act(mode, face, dt, now)
            self._annotate(frame, face, mode)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._cond:
                    self._jpeg = buf.tobytes()
                    self._cond.notify_all()

    def close(self):
        self._running = False
        self.cap.release()
