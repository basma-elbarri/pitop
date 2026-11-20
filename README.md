# pi-top phone-controlled face-following robot

Drive the robot from your phone, watch the camera, and let it find the nearest
face, look at it, and drive up to it.

## Files
- `robot.py` – wheels (M3 left, M0 right), pan/tilt servos, front/back ultrasonic, safety watchdog
- `follower.py` – camera, nearest-face detection (biggest face = nearest), head tracking, follow mode
- `app.py` – web server + phone UI (joystick, camera view, mode buttons, STOP)

## Setup (on the pi-top)
```
sudo apt install python3-opencv python3-flask      # or: pip3 install flask opencv-python-headless
ROBOT_PIN=4821 python3 app.py
```
Open `http://<pi-top-ip>:5000` on your phone (same Wi-Fi; try `http://pi-top.local:5000`).
Enter the PIN. Find the IP with `hostname -I`.

## Before the first run – check `robot.py` CONFIG
You only told me the motor ports, so these are **guesses** – set them to your wiring:
- `PAN_SERVO_PORT` / `TILT_SERVO_PORT` (default S0 / S1)
- `FRONT_SONAR_PORT` / `BACK_SONAR_PORT` (default D3 / D4)

Then test with the robot **lifted off the ground**, in Manual mode:
1. Push joystick up: both wheels should go forward. If it turns instead, swap `LEFT_MOTOR_PORT`/`RIGHT_MOTOR_PORT`; if it goes backward, the motors' direction needs flipping.
2. Push right: robot should turn right. If left, set `TURN_SIGN = -1`.
3. Pan slider right: camera looks right. If not, `PAN_SIGN = -1`. Same for tilt with `TILT_SIGN`.
4. Wave a hand in front of each sonar; the status line should show the distance.
5. If the camera image is upside down, set `FLIP_IMAGE = True` in `follower.py`.

## Modes
- **Manual** – joystick + camera sliders.
- **Look at me** – camera servos track the nearest face; wheels stay still.
- **Follow me** – head tracks the face, body turns toward where the head points, and the
  robot drives forward until the face fills about `TARGET_FACE_FRACTION` of the frame.
  If it loses the face it stops, waits 1 s, then sweeps the head to search.

## Safety built in
- Wheels stop if the phone stops sending commands for 0.6 s (dropped Wi-Fi, locked phone).
- Front sonar blocks forward motion and back sonar blocks reverse inside 0.30 m.
- If a sonar can't be read, motion in that direction is blocked (`REQUIRE_SONAR`).
- STOP button always returns to Manual with wheels stopped.

## Using it away from home
Don't port-forward this to the internet. Install Tailscale on the pi-top and your phone
(`curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up`) and use the
pi-top's Tailscale IP.

## Tuning
All in `follower.py`: `PAN_RATE`/`TILT_RATE` (head speed), `BODY_TURN_GAIN`,
`TARGET_FACE_FRACTION` (stopping distance), `MAX_FOLLOW_SPEED`.
The face detector is OpenCV's built-in Haar cascade: it needs a mostly front-facing face and
decent light. For a sturdier detector later, swap `_largest_face` for a DNN/MediaPipe model.
