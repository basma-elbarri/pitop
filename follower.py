                2
            )

            cv2.circle(
                frame,
                (int(target["cx"]), int(target["cy"])),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"PERSON score={target['score']:.2f}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Mode: {mode}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # ------------------------------ MAIN LOOP -------------------------------

    def _loop(self):
        last_time = time.time()

        while self._running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                print("[CAMERA ERROR] Failed to read frame")
                time.sleep(0.1)
                continue

            if FLIP_IMAGE:
                frame = cv2.flip(frame, -1)

            self._frame_no += 1
            now = time.time()
            dt = min(max(now - last_time, 0.001), 0.2)
            last_time = now

            mode = self.mode
            target = None

            if mode != "manual":
                if self._frame_no % HOG_EVERY == 0:
                    try:
                        target = self._detect_person(frame)
                    except Exception as error:
                        print("[DETECTION ERROR]", repr(error))

                self._act(mode, target, dt, now)

                if self._frame_no % STATUS_EVERY == 0:
                    print(
                        "[TRACKING]",
                        "mode=", mode,
                        "target=", "FOUND" if target else "NOT FOUND",
                        "frame=", frame.shape
                    )

            self._annotate(frame, target, mode)

            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if encoded:
                with self._cond:
                    self._jpeg = buffer.tobytes()
                    self._cond.notify_all()

        # Stop motors when the camera loop exits.
        try:
            self.robot.stop()
        except Exception:
            pass

    # ------------------------------- CLEANUP --------------------------------

    def close(self):
        self._running = False

        try:
            self.robot.stop()
        except Exception:
            pass

        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        with self._cond:
            self._cond.notify_all()

        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=2.0)
