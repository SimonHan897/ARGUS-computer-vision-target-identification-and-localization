#!/usr/bin/env python3
"""Try the green-X detector on a webcam, an image, or a generated target.

Green-cross version of tools/bullseye_try.py. No ROS needed: it runs the same
GreenCrossDetector the mission code would use.

Examples:
  python3 tools/green_cross_try.py --make-target green_x.png    # target to show on a phone/print
  python3 tools/green_cross_try.py --image photo.jpg            # writes photo_detected.png
  python3 tools/green_cross_try.py --webcam                     # same camera as bullseye_try.py; q quits, s saves
  python3 tools/green_cross_try.py --usb                        # plain USB webcam instead
  python3 tools/green_cross_try.py --webcam --set hsv_s_min=50  # change a setting and compare
  python3 tools/green_cross_try.py --image green_cross_raw_1.png # re-test a saved raw camera frame
  python3 tools/green_cross_try.py --webcam --capture            # hands-free: saves 3 raw frames
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "drone_target_cv"))

from drone_target_cv.green_cross_detector import (  # noqa: E402
    GreenCrossDetector,
    GreenCrossDetectorConfig,
    draw_debug_overlay,
)

# Settings measured on the club PC's laptop camera (ov2740, uncalibrated) looking at the
# C-UASC rules drawing on a phone: pale green (saturation 30-55), thin lines.
WEBCAM_PRESET = dict(use_line_search=True, line_hsv_s_min=30, line_green_dominance_ratio=1.08,
                     line_green_difference_min=15,
                     green_dominance_s_min=70, green_dominance_v_min=45)
# Only the thin-line path gets the looser green test; blobs keep the strict colour test,
# so skin / faces cannot pass as green. Like the bullseye, "greener than red/blue" pixels must
# also be saturated, so the green-tinted white sheet doesn't turn into speckle in dim light.

LIBCAMERA = ("libcamerasrc ! video/x-raw,framerate=30/1 ! videoconvert ! "
             "video/x-raw,format=BGR ! appsink drop=true max-buffers=1")


def build_config(overrides: list[str], preset: dict | None = None) -> GreenCrossDetectorConfig:
    fields = {f.name: f for f in dataclasses.fields(GreenCrossDetectorConfig)}
    values = dict(preset or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        if key not in fields:
            sys.exit(f"Unknown setting '{key}'. Valid: {', '.join(sorted(fields))}")
        default = fields[key].default
        if isinstance(default, bool):
            values[key] = raw.lower() in ("1", "true", "yes")
        elif isinstance(default, tuple):
            values[key] = tuple(type(default[0])(x) for x in raw.split(","))
        else:
            values[key] = type(default)(raw)
        print(f"override {key} = {values[key]}")
    return GreenCrossDetectorConfig(**values)


def make_target(path: str, size: int = 800, arm_frac: float = 0.20) -> None:
    """White square with a bright green corner-to-corner X (arm width = arm_frac of the side).
    Bright, thick green survives a phone screen + uncalibrated webcam much better."""
    frame = np.full((size, size, 3), 255, dtype=np.uint8)
    w = int(size * arm_frac)
    m = w // 2
    cv2.line(frame, (m, m), (size - m, size - m), (0, 190, 0), w, lineType=cv2.LINE_AA)
    cv2.line(frame, (size - m, m), (m, size - m), (0, 190, 0), w, lineType=cv2.LINE_AA)
    cv2.imwrite(path, frame)
    print(f"wrote {path}")


def gray_world(frame: np.ndarray) -> np.ndarray:
    """Simple automatic white balance: scale B, G, R so the scene averages to grey.
    Removes the green/magenta cast of uncalibrated laptop cameras (e.g. ov2740 + libcamera)."""
    f = frame.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


class Tracker:
    """Keep the last detection for a short time when a frame misses (shown as 'held'),
    and report the real per-frame detection rate so the display stays honest."""

    def __init__(self, hold_s: float = 0.4, window: int = 30):
        self.hold_s, self.last, self.last_t, self.hist = hold_s, None, 0.0, []
        self.window = window

    def update(self, result):
        now = time.time()
        self.hist = (self.hist + [result.accepted])[-self.window:]
        if result.accepted:
            self.last, self.last_t = result, now
            return result, False
        if self.last is not None and now - self.last_t <= self.hold_s:
            return self.last, True
        return result, False

    @property
    def rate(self) -> float:
        return sum(self.hist) / len(self.hist) if self.hist else 0.0


def run(detector: GreenCrossDetector, frame: np.ndarray, width: int = 640,
        show_mask: bool = False, wb: bool = False, tracker: "Tracker | None" = None) -> tuple[np.ndarray, str]:
    if wb:
        frame = gray_world(frame)
    # Detect on a small copy (drone-camera size, much faster), draw on the full-size frame
    k = 1.0
    small = frame
    if width and frame.shape[1] > width:
        k = frame.shape[1] / float(width)
        small = cv2.resize(frame, (width, int(round(frame.shape[0] / k))), interpolation=cv2.INTER_AREA)
    start = time.perf_counter()
    result = detector.detect(small)
    ms = (time.perf_counter() - start) * 1000.0
    if show_mask:   # show what the detector counts as "green"
        frame = cv2.resize(cv2.cvtColor(result.green_mask, cv2.COLOR_GRAY2BGR),
                           (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    if k != 1.0:
        result = dataclasses.replace(
            result,
            center=None if result.center is None else (int(result.center[0] * k), int(result.center[1] * k)),
            radius_px=result.radius_px * k,
            contour=None if result.contour is None else (result.contour * k).astype(np.int32))
    held = False
    if tracker is not None:
        result, held = tracker.update(result)
    tag = " (held)" if held else ""
    rate = f" | detected {tracker.rate:.0%} of last {len(tracker.hist)} frames" if tracker else ""
    if result.accepted:
        x, y = result.normalized_center(frame.shape[1], frame.shape[0])
        line = (f"FOUND{tag} x={x:+.2f} y={y:+.2f} conf={result.confidence:.2f} "
                f"rot={result.orientation_deg:.0f} {ms:.0f} ms{rate}")
    else:
        line = f"none ({result.reject_reason}) {ms:.0f} ms{rate}"
    annotated = draw_debug_overlay(frame, result)
    if held:
        cv2.putText(annotated, "HELD (last detection)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 255), 2)
    cv2.putText(annotated, line, (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    return annotated, line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", action="store_true", help="same camera as bullseye_try.py --webcam")
    src.add_argument("--usb", nargs="?", const=0, type=int, metavar="INDEX", help="plain USB webcam")
    src.add_argument("--image")
    src.add_argument("--make-target", metavar="OUT.png")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--capture", action="store_true",
                        help="with --webcam: count down 5 s, save 3 raw frames automatically, then exit")
    parser.add_argument("--wb", action="store_true",
                        help="automatic white balance (try it if colours look tinted)")
    parser.add_argument("--drone-settings", action="store_true",
                        help="use the drone/aerial settings instead of the webcam preset")
    parser.add_argument("--width", type=int, default=640,
                        help="shrink frames to this width before detecting (0 = full size)")
    args = parser.parse_args()

    if args.make_target:
        make_target(args.make_target)
        return
    preset = None if args.drone_settings else WEBCAM_PRESET
    if preset:
        print("using the webcam preset (thin pale-green lines); --drone-settings to turn it off")
    detector = GreenCrossDetector(build_config(args.set, preset))

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read {args.image}")
        annotated, line = run(detector, frame, args.width)
        out = str(pathlib.Path(args.image).with_name(pathlib.Path(args.image).stem + "_detected.png"))
        cv2.imwrite(out, annotated)
        print(line)
        print(f"wrote {out}")
        return

    # --webcam opens the camera exactly like bullseye_try.py (libcamera via GStreamer).
    # --usb uses a plain USB webcam instead.
    if args.usb is not None:
        cap = cv2.VideoCapture(args.usb)
    else:
        cap = cv2.VideoCapture(LIBCAMERA, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit("Could not open the camera. Close any other camera program "
                 "(e.g. bullseye_try.py) and try again.")
    if args.capture:   # hands-free: no key presses needed
        t_end, saved = time.time() + 5.0, 0
        while saved < 3:
            ok, frame = cap.read()
            if not ok:
                break
            left = t_end - time.time()
            view = frame.copy()
            msg = f"hold the green X still... {left:.0f}" if left > 0 else "saving..."
            cv2.putText(view, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imshow("green_cross_try capture", view)
            cv2.waitKey(1)
            if left <= 0:
                saved += 1
                cv2.imwrite(f"green_cross_raw_{saved}.png", frame)
                _, line = run(detector, frame, args.width, False, wb=args.wb)
                print(f"saved green_cross_raw_{saved}.png  ->  {line}")
                time.sleep(0.4)
        cap.release()
        cv2.destroyAllWindows()
        print("done - tell Claude 'saved'")
        return

    shots, show_mask = 0, False
    tracker = Tracker()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, line = run(detector, frame, args.width, show_mask, wb=args.wb, tracker=tracker)
        print(line, end="\r", flush=True)
        cv2.imshow("green_cross_try (q quit, s save, m mask)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("s"):
            shots += 1
            name = f"green_cross_shot_{shots}.png"
            cv2.imwrite(name, annotated)
            cv2.imwrite(f"green_cross_raw_{shots}.png", frame)   # unmarked frame, for tuning
            print(f"\nsaved {name} and green_cross_raw_{shots}.png")
    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
