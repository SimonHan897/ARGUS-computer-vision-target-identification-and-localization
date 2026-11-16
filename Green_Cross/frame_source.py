"""Camera frame sources for the test harness.

GstUdpSource     the Gazebo gimbal camera's H.264/RTP stream on UDP 5600
                 (ardupilot_gazebo GstCameraPlugin), read with OpenCV + GStreamer
SyntheticSource  renders frames from the vehicle's live pose with synth.py, so the
                 whole harness can be dry-run without Gazebo

Both run a background thread and expose latest() -> (frame, host_time, frame_index).
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

GST_PIPELINE = (
    "udpsrc port={port} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, "
    "encoding-name=(string)H264\" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
)


def opencv_has_gstreamer() -> bool:
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


class _Threaded:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._idx = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _publish(self, frame: np.ndarray, t: float) -> None:
        with self._lock:
            self._frame, self._t = frame, t
            self._idx += 1

    def latest(self):
        with self._lock:
            return self._frame, self._t, self._idx

    def wait_first(self, timeout: float = 15.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.latest()[0] is not None:
                return True
            time.sleep(0.1)
        return False

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class GstUdpSource(_Threaded):
    def __init__(self, port: int = 5600, latency_s: float = 0.10, open_timeout: float = 25.0) -> None:
        super().__init__()
        self.latency_s = latency_s     # stream delay: frame shows the scene ~this long ago
        # Opening a UDP stream blocks until the first video arrives, so open it in a helper
        # thread and give up after open_timeout instead of hanging forever.
        box: dict = {}
        opener = threading.Thread(target=lambda: box.update(
            cap=cv2.VideoCapture(GST_PIPELINE.format(port=port), cv2.CAP_GSTREAMER)), daemon=True)
        opener.start()
        opener.join(open_timeout)
        if opener.is_alive():
            raise RuntimeError(
                f"No video arrived on UDP {port} within {open_timeout:.0f} s. Check: (1) the camera "
                "stream is enabled (python3 sim_targets.py setup), (2) QGroundControl is not "
                "using port 5600 (QGC > Application Settings > Video > Source = disabled), "
                "(3) sudo apt install gstreamer1.0-libav gstreamer1.0-plugins-bad")
        self.cap = box["cap"]
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open the UDP video stream. Check: (1) OpenCV has GStreamer "
                f"(opencv_has_gstreamer() = {opencv_has_gstreamer()}), (2) "
                "sudo apt install gstreamer1.0-libav gstreamer1.0-plugins-good, "
                "(3) streaming was enabled with sim_targets.py setup.")

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                self._publish(frame, time.time() - self.latency_s)
            else:
                time.sleep(0.01)
        self.cap.release()


class SyntheticSource(_Threaded):
    """Render what the downward camera would see, at ~fps, from the vehicle's pose."""

    def __init__(self, vehicle, scene, cam, home, mount: str = "rot0", fps: float = 10.0,
                 seed: int = 1) -> None:
        super().__init__()
        self.vehicle, self.scene, self.cam, self.home = vehicle, scene, cam, home
        self.mount, self.period = mount, 1.0 / fps
        self.rng = np.random.default_rng(seed)
        self.latency_s = 0.0

    def _run(self) -> None:
        from geo import latlon_to_ne
        from synth import render
        while not self._stop.is_set():
            t0 = time.time()
            s = self.vehicle.state()
            if s["lat"] is not None and s["rel_alt"] > 0.3:
                ne = latlon_to_ne(self.home[0], self.home[1], s["lat"], s["lon"])
                img = render(self.scene, self.cam, ne, s["rel_alt"], s["roll"], s["pitch"], s["yaw"],
                             self.mount, rng=self.rng,
                             photometric={"gain": 1.0, "noise": 3, "jpeg": 70, "blur_px": 0.6})
                self._publish(img, t0)
            time.sleep(max(0.0, self.period - (time.time() - t0)))
