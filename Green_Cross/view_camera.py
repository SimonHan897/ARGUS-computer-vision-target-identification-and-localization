#!/usr/bin/env python3
"""Live view of the Gazebo camera stream with the green-X detector overlay.

Use it to check the camera points straight down before a test run.
Keys: q quit, s save a screenshot (to evidence/).
"""

import pathlib
import time

import cv2

from frame_source import GstUdpSource
from green_cross_detector import GreenCrossDetector, draw_debug_overlay


def main() -> None:
    src = GstUdpSource().start()
    if not src.wait_first(15):
        raise SystemExit("No video on UDP 5600. Run: python3 sim_targets.py setup")
    det = GreenCrossDetector()
    out = pathlib.Path("evidence")
    out.mkdir(exist_ok=True)
    last, shots = -1, 0
    while True:
        frame, _, idx = src.latest()
        if idx != last:
            last = idx
            t0 = time.perf_counter()
            res = det.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            view = draw_debug_overlay(frame, res)
            cv2.putText(view, f"{ms:.0f} ms", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow("Gazebo camera + green-X detector (q quit, s save)", view)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            shots += 1
            name = out / f"camera_view_{shots:02d}.png"
            cv2.imwrite(str(name), view)
            print(f"saved {name}")
    src.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
