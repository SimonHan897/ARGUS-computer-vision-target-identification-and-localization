#!/usr/bin/env python3
"""PD7 test run: autonomous search, green-X detection and geolocation in SITL.

One run = take off, fly a lawnmower search over the test area, detect the X in
every camera frame, geolocate each detection, then fly over the estimate,
take extra looks, and land. Everything is logged to runs/<run-id>/.

  python3 run_test.py --run-id R1 --alt 6.5
  python3 run_test.py --run-id R0 --alt 8 --mount rot0          # calibration run
  python3 run_test.py --run-id DRY --source synthetic             # no Gazebo camera

Logged per run (runs/<run-id>/):
  frames.csv      one row per processed frame: pose, detection, pixel, geolocated position
  telemetry.csv   10 Hz vehicle state
  events.log      everything printed to the terminal
  config.json     the exact settings used (altitude, area, mount, detector config)
  truth.json      copied from sim_targets.py (if present) - NOT used during the flight
  annotated.mp4   camera video with the detector overlay and a status bar
  snapshots/      first detections as PNGs
  estimate.json   the in-flight target estimate (lat/lon) and when it was made
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import pathlib
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from frame_source import GstUdpSource, SyntheticSource
from geo import MOUNTS, Camera, latlon_to_ne, ne_to_latlon, pixel_to_ground
from green_cross_detector import GreenCrossDetector, draw_debug_overlay
from vehicle import MissionAbort, Vehicle

COMPETITION_MIN_ALT_M = 6.096   # 20 ft AGL while searching (C-UASC rules)


class Logger:
    def __init__(self, path: pathlib.Path):
        self.f = open(path, "w")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def lawnmower(area_n: tuple[float, float], area_e: tuple[float, float], spacing: float):
    """Waypoints (north, east) sweeping the box in east-west lanes."""
    pts, n, flip = [], area_n[0], False
    while n <= area_n[1] + 1e-6:
        row = [(n, area_e[0]), (n, area_e[1])]
        pts.extend(reversed(row) if flip else row)
        n += spacing
        flip = not flip
    return pts


def robust_estimate(points: np.ndarray, weights: np.ndarray | None = None):
    """Median position after rejecting outliers > 3 MAD (min 1 m) from the median."""
    if len(points) == 0:
        return None
    med = np.median(points, axis=0)
    d = np.hypot(*(points - med).T)
    mad = max(1.0, 1.4826 * float(np.median(d)))
    keep = d <= 3 * mad
    return np.median(points[keep], axis=0), int(keep.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--connect", default="udpin:127.0.0.1:14551")
    ap.add_argument("--alt", type=float, default=8.0, help="search altitude AGL, m (fence max 12.2 m)")
    ap.add_argument("--speed", type=float, default=3.0, help="search speed, m/s")
    ap.add_argument("--area-n", type=float, nargs=2, default=(5.0, 35.0), metavar=("MIN", "MAX"),
                    help="search box north limits from home, m")
    ap.add_argument("--area-e", type=float, nargs=2, default=(-15.0, 15.0), metavar=("MIN", "MAX"),
                    help="search box east limits from home, m")
    ap.add_argument("--overlap", type=float, default=0.4, help="lane overlap fraction of footprint")
    ap.add_argument("--mount", default=None, choices=sorted(MOUNTS),
                    help="camera mount; default = runs/mount.json from calibration, else rot0")
    ap.add_argument("--source", default="gst", choices=["gst", "synthetic"])
    ap.add_argument("--latency", type=float, default=0.10, help="video stream delay, s")
    ap.add_argument("--confirm-looks", type=float, default=6.0,
                    help="seconds to hover over the estimate collecting extra detections")
    ap.add_argument("--truth", default="truth.json", help="copied into the run folder for analysis")
    ap.add_argument("--show", action="store_true", help="show a live camera window with the detector overlay")
    args = ap.parse_args()

    run_dir = pathlib.Path("runs") / args.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists - pick a new --run-id")
    (run_dir / "snapshots").mkdir(parents=True)
    log = Logger(run_dir / "events.log")

    mount = args.mount
    mount_file = pathlib.Path("runs/mount.json")
    if mount is None:
        mount = json.loads(mount_file.read_text())["mount"] if mount_file.exists() else "rot0"
    cam = Camera()
    det = GreenCrossDetector()
    fw, fh = cam.ground_footprint_m(args.alt)
    spacing = max(2.0, (1 - args.overlap) * min(fw, fh))
    if pathlib.Path(args.truth).exists():
        shutil.copy(args.truth, run_dir / "truth.json")

    config = dict(vars(args), mount=mount, camera=dataclasses.asdict(cam), fx=cam.fx,
                  footprint_m=[fw, fh], lane_spacing_m=spacing,
                  detector=dataclasses.asdict(det.config), started=datetime.now().isoformat())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log(f"RUN {args.run_id}: alt {args.alt} m, speed {args.speed} m/s, mount {mount}, "
        f"footprint {fw:.1f} x {fh:.1f} m, lane spacing {spacing:.1f} m")
    if args.alt < COMPETITION_MIN_ALT_M:
        log("WARNING: search altitude below the 20 ft (6.1 m) competition minimum")

    v = Vehicle(args.connect, log)
    frames_f = open(run_dir / "frames.csv", "w", newline="")
    fw_csv = csv.writer(frames_f)
    fw_csv.writerow(["t_s", "phase", "frame_idx", "lat", "lon", "alt_m", "roll_deg", "pitch_deg",
                     "yaw_deg", "accepted", "confidence", "u_px", "v_px", "arms", "regularity",
                     "solidity", "white", "est_n_m", "est_e_m", "est_lat", "est_lon", "det_ms",
                     "reject_reason"])
    tel_f = open(run_dir / "telemetry.csv", "w", newline="")
    tel_csv = csv.writer(tel_f)
    tel_csv.writerow(["t_s", "phase", "mode", "armed", "lat", "lon", "alt_m", "groundspeed_mps",
                      "roll_deg", "pitch_deg", "yaw_deg"])
    video = None
    t0 = time.time()
    state = {"phase": "init", "last_idx": -1, "last_tel": 0.0, "detections": [], "snapshots": 0,
             "frames": 0, "first_det_t": None}
    src = None

    def process_frames() -> None:
        """Called continuously while flying: telemetry log + newest frame through the detector."""
        nonlocal video
        now = time.time()
        if now - state["last_tel"] >= 0.1:
            s = v.state()
            if s["lat"] is not None:
                tel_csv.writerow([round(now - t0, 3), state["phase"], s["mode"], int(s["armed"]),
                                  s["lat"], s["lon"], round(s["rel_alt"], 3),
                                  round(math.hypot(s["vn"], s["ve"]), 2),
                                  round(math.degrees(s["roll"]), 2), round(math.degrees(s["pitch"]), 2),
                                  round(math.degrees(s["yaw"]), 2)])
            state["last_tel"] = now
        if src is None:
            time.sleep(0.01)
            return
        frame, t_frame, idx = src.latest()
        if frame is None or idx == state["last_idx"]:
            time.sleep(0.005)
            return
        state["last_idx"] = idx
        pose = v.pose_at(t_frame)
        if pose is None or v.home is None:
            return
        lat, lon, alt, roll, pitch, yaw = pose
        tic = time.perf_counter()
        res = det.detect(frame)
        ms = (time.perf_counter() - tic) * 1000
        state["frames"] += 1
        est = (None, None, None, None)
        if res.accepted and alt > 1.0:
            ne = latlon_to_ne(*v.home, lat, lon)
            g = pixel_to_ground(res.center[0], res.center[1], cam, ne, alt, roll, pitch, yaw, mount)
            if g is not None:
                glat, glon = ne_to_latlon(*v.home, *g)
                est = (g[0], g[1], glat, glon)
                state["detections"].append((g[0], g[1], res.confidence, now - t0, state["phase"]))
                if state["first_det_t"] is None:
                    state["first_det_t"] = now - t0
                    log(f"FIRST DETECTION at t={now - t0:.1f} s, est N {g[0]:+.2f} E {g[1]:+.2f} m")
        fw_csv.writerow([round(t_frame - t0, 3), state["phase"], idx, lat, lon, round(alt, 3),
                         round(math.degrees(roll), 2), round(math.degrees(pitch), 2),
                         round(math.degrees(yaw), 2), int(res.accepted), round(res.confidence, 3),
                         res.center[0] if res.accepted else "", res.center[1] if res.accepted else "",
                         res.arm_count, round(res.arm_regularity, 3), round(res.solidity, 3),
                         round(res.white_backing, 3),
                         *("" if x is None else round(x, 3) if i < 2 else round(x, 8) for i, x in enumerate(est)),
                         round(ms, 1), res.reject_reason])
        ann = draw_debug_overlay(frame, res)
        bar = (f"{args.run_id} | {state['phase']} | alt {alt:5.2f} m | det {len(state['detections'])}"
               f" | {ms:4.0f} ms")
        cv2.rectangle(ann, (0, ann.shape[0] - 26), (ann.shape[1], ann.shape[0]), (0, 0, 0), -1)
        cv2.putText(ann, bar, (8, ann.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if video is None:
            video = cv2.VideoWriter(str(run_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    10, (ann.shape[1], ann.shape[0]))
        video.write(ann)
        if args.show:
            cv2.imshow(f"LIVE drone camera - {args.run_id}", cv2.resize(ann, None, fx=1.5, fy=1.5))
            cv2.waitKey(1)
        if res.accepted and state["snapshots"] < 8 and len(state["detections"]) % 5 == 1:
            state["snapshots"] += 1
            cv2.imwrite(str(run_dir / "snapshots" / f"det_{state['snapshots']:02d}.png"), ann)

    def fly_to(n: float, e: float, alt: float, tol: float = 1.0, timeout: float = 90) -> None:
        lat, lon = ne_to_latlon(*v.home, n, e)
        v.goto(lat, lon, alt)
        end, last_send = time.time() + timeout, time.time()
        while time.time() < end:
            process_frames()
            s = v.state()
            if s["mode"] != "GUIDED":
                raise MissionAbort(f"left GUIDED (now {s['mode']}) - fence or failsafe?")
            if time.time() - last_send > 2:
                v.goto(lat, lon, alt)
                last_send = time.time()
            if v.distance_to(lat, lon) <= tol and abs(s["rel_alt"] - alt) < 1.0:
                return
        raise MissionAbort(f"timed out flying to N {n:.1f} E {e:.1f}")

    def hover(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            process_frames()

    airborne, rc = False, 1
    try:
        v.connect()
        v.wait_ready()
        if args.source == "gst":
            src = GstUdpSource(latency_s=args.latency).start()
            if not src.wait_first(15):
                raise MissionAbort("no video frames on UDP 5600 - run: python3 sim_targets.py setup")
        else:
            from synth import draw_distractor, draw_green_x, make_scene
            truth = json.loads((run_dir / "truth.json").read_text()) if (run_dir / "truth.json").exists() \
                else {"target": {"north_m": 20.0, "east_m": 5.0, "rot_deg": 20.0}, "decoys": []}
            rng = np.random.default_rng(7)
            scene, _ = make_scene(rng, ground="asphalt", target=False, n_distractors=0, size_m=160)
            t = truth["target"]
            draw_green_x(scene, t["north_m"], t["east_m"], t["rot_deg"])
            for d in truth.get("decoys", []):
                kind = "green_square" if "square" in d["name"] else "bullseye"
                draw_distractor(scene, kind, d["north_m"], d["east_m"], rng)
            src = SyntheticSource(v, scene, cam, v.home, mount).start()
        log(f"camera stream OK ({args.source})")

        state["phase"] = "takeoff"
        v.set_mode("GUIDED")
        v.arm()
        airborne = True
        v.takeoff(args.alt, tick=process_frames)
        v.set_speed(args.speed)

        state["phase"] = "search"
        wps = lawnmower(tuple(args.area_n), tuple(args.area_e), spacing)
        log(f"SEARCH: {len(wps)} waypoints, box N {args.area_n} E {args.area_e}")
        t_search = time.time()
        for i, (n, e) in enumerate(wps, 1):
            fly_to(n, e, args.alt, tol=1.5)
            log(f"  search waypoint {i}/{len(wps)} reached, detections so far {len(state['detections'])}")
        search_s = time.time() - t_search

        if not state["detections"]:
            log("RESULT: target NOT found during search")
            estimate = None
        else:
            pts = np.array([[d[0], d[1]] for d in state["detections"]])
            (en, ee), used = robust_estimate(pts)
            log(f"search estimate N {en:+.2f} E {ee:+.2f} m from {used}/{len(pts)} detections")
            state["phase"] = "confirm"
            fly_to(en, ee, args.alt, tol=1.0)
            hover(args.confirm_looks)
            pts = np.array([[d[0], d[1]] for d in state["detections"]])
            (en, ee), used = robust_estimate(pts)
            lat, lon = ne_to_latlon(*v.home, en, ee)
            estimate = {"north_m": float(en), "east_m": float(ee), "lat": lat, "lon": lon,
                        "detections_used": used, "detections_total": int(len(pts)),
                        "t_first_detection_s": state["first_det_t"], "t_estimate_s": time.time() - t0}
            log(f"RESULT: target estimate {lat:.7f}, {lon:.7f} (N {en:+.2f} E {ee:+.2f} m, "
                f"{used}/{len(pts)} detections)")

        state["phase"] = "return"
        fly_to(0.0, 0.0, args.alt, tol=1.5, timeout=120)
        state["phase"] = "land"
        v.land(tick=process_frames)
        airborne = False
        summary = {"run_id": args.run_id, "estimate": estimate, "search_time_s": search_s,
                   "mission_time_s": time.time() - t0, "frames_processed": state["frames"],
                   "detections": len(state["detections"]), "mount": mount}
        (run_dir / "estimate.json").write_text(json.dumps(summary, indent=2))
        log(f"RUN COMPLETE. Logs in {run_dir}/  -> next: python3 analyze_runs.py")
        rc = 0
    except (MissionAbort, KeyboardInterrupt, RuntimeError) as e:
        log(f"ABORT: {'Ctrl+C' if isinstance(e, KeyboardInterrupt) else e}")
        if airborne or v.state()["armed"]:
            log("Safety action: LAND")
            try:
                state["phase"] = "abort_land"
                v.land(tick=process_frames)
            except Exception as e2:  # noqa: BLE001
                log(f"LAND after abort did not complete: {e2}")
    finally:
        if src is not None:
            src.stop()
        if video is not None:
            video.release()
        frames_f.close()
        tel_f.close()
        v.close()
        if args.show:
            cv2.destroyAllWindows()
    return rc


if __name__ == "__main__":
    sys.exit(main())
