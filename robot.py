"""Hardware layer for the pi-top robot: wheels, pan/tilt servos, ultrasonic
sensors and a safety watchdog.  Edit the CONFIG block to match your wiring."""
import threading
import time

from pitop import DriveController, ServoMotor, UltrasonicSensor

# ------------------------------ CONFIG ------------------------------------
LEFT_MOTOR_PORT = "M3"      # swap with M0 if the robot drives "backwards-turned"
RIGHT_MOTOR_PORT = "M0"
PAN_SERVO_PORT = "S0"       # left/right servo   <-- check your plate
TILT_SERVO_PORT = "S1"      # up/down servo      <-- check your plate
FRONT_SONAR_PORT = "D3"     # front ultrasonic   <-- check your plate
BACK_SONAR_PORT = "D4"      # rear ultrasonic    <-- check your plate

PAN_SIGN = 1        # set to -1 if "look right" moves the camera left
TILT_SIGN = 1       # set to -1 if "look up" moves the camera down
TURN_SIGN = 1       # set to -1 if "turn right" spins the robot left
PAN_CENTER = 0      # servo angle (-90..90) where the camera looks straight
TILT_CENTER = 0
PAN_LIMIT = 80      # logical degrees each side of center
TILT_LIMIT = 50

MAX_SPEED_FRACTION = 0.6    # share of motor top speed you allow
MAX_TURN_FRACTION = 0.6
STOP_DISTANCE = 0.30        # metres: block motion toward an obstacle closer than this
REQUIRE_SONAR = True        # if a sensor can't be read, block that direction
COMMAND_TIMEOUT = 0.6       # seconds without a command -> wheels stop
# --------------------------------------------------------------------------


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Robot:
    def __init__(self):
        self.drive = DriveController(
            left_motor_port=LEFT_MOTOR_PORT, right_motor_port=RIGHT_MOTOR_PORT
        )
        self.pan_servo = ServoMotor(PAN_SERVO_PORT)
        self.tilt_servo = ServoMotor(TILT_SERVO_PORT)
        self.front_sonar = UltrasonicSensor(FRONT_SONAR_PORT, threshold_distance=STOP_DISTANCE)
        self.back_sonar = UltrasonicSensor(BACK_SONAR_PORT, threshold_distance=STOP_DISTANCE)

        self._max_lin = float(getattr(self.drive, "max_motor_speed", 0.25))
        self._max_ang = float(getattr(self.drive, "max_robot_angular_speed", 2.0))

        self._lock = threading.Lock()
        self._forward = 0.0
        self._turn = 0.0
        self._last_cmd = 0.0
        self._was_moving = False

        self.pan = 0.0    # logical: + is right, 0 is center
        self.tilt = 0.0   # logical: + is up, 0 is center
        self.front = None
        self.back = None

        self._running = True
        self.look(0, 0, force=True)
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------------------------- commands --------------------------------
    def set_drive(self, forward, turn):
        """forward and turn are -1..1.  forward>0 = ahead, turn>0 = right."""
        with self._lock:
            self._forward = _clamp(float(forward), -1, 1)
            self._turn = _clamp(float(turn), -1, 1)
            self._last_cmd = time.time()

    def stop(self):
        self.set_drive(0, 0)

    def look(self, pan, tilt, force=False):
        pan = _clamp(float(pan), -PAN_LIMIT, PAN_LIMIT)
        tilt = _clamp(float(tilt), -TILT_LIMIT, TILT_LIMIT)
        if force or abs(pan - self.pan) > 0.3:
            self.pan_servo.target_angle = _clamp(PAN_CENTER + PAN_SIGN * pan, -90, 90)
            self.pan = pan
        if force or abs(tilt - self.tilt) > 0.3:
            self.tilt_servo.target_angle = _clamp(TILT_CENTER + TILT_SIGN * tilt, -90, 90)
            self.tilt = tilt

    # ----------------------------- safety loop ----------------------------
    @staticmethod
    def _read(sensor):
        try:
            return round(float(sensor.distance), 2)
        except Exception:
            return None

    def _blocked(self, distance):
        if distance is None:
            return REQUIRE_SONAR
        return distance < STOP_DISTANCE

    def _loop(self):
        while self._running:
            self.front = self._read(self.front_sonar)
            self.back = self._read(self.back_sonar)
            with self._lock:
                fwd, turn = self._forward, self._turn
                stale = (time.time() - self._last_cmd) > COMMAND_TIMEOUT
            if stale:
                fwd = turn = 0.0
            if fwd > 0 and self._blocked(self.front):
                fwd = 0.0
            if fwd < 0 and self._blocked(self.back):
                fwd = 0.0
            self._apply(fwd, turn)
            time.sleep(0.05)

    def _apply(self, fwd, turn):
        try:
            if fwd == 0 and turn == 0:
                if self._was_moving:
                    self.drive.stop()
                    self._was_moving = False
                return
            self.drive.robot_move(
                fwd * MAX_SPEED_FRACTION * self._max_lin,
                -turn * TURN_SIGN * MAX_TURN_FRACTION * self._max_ang,
            )
            self._was_moving = True
        except Exception as e:  # never let the safety loop die
            print("drive error:", e)

    def close(self):
        self._running = False
        try:
            self.drive.stop()
            self.look(0, 0, force=True)
        except Exception:
            pass
