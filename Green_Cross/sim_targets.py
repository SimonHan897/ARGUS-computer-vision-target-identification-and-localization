#!/usr/bin/env python3
"""Gazebo helpers for the PD5/PD7 test: targets, camera aim, video stream.

Every command runs inside the simulator container (docker exec ardupilot-gz-sim),
so nothing extra has to be installed on the host for this part.

  python3 sim_targets.py setup                    # point camera down + start video stream
  python3 sim_targets.py spawn --north 18 --east -6 --rot 20 [--decoys]
  python3 sim_targets.py clear                    # remove targets + decoys
  python3 sim_targets.py topics                   # list camera/gimbal topics (debug)

Gazebo's world frame is ENU (x = East, y = North) with the origin at the drone's
start point = ArduPilot home, so a target at (north, east) metres from home is
spawned at x = east, y = north. The true lat/lon is written to truth.json.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys

from geo import ne_to_latlon

CONTAINER = "ardupilot-gz-sim"
WORLD = "iris_runway"
HOME = (-35.3632621, 149.1652374)     # iris_runway spherical coordinates = SITL home

# Target size: the C-UASC rules give none for the green-X Package Recovery target.
# Assumed 3 m white sheet with an inset green X (arm 3.6 m long, 0.45 m wide).
SHEET_M, X_LEN_M, X_ARM_M = 3.0, 3.6, 0.45


def gz(cmd: str, check: bool = True, timeout: float = 30) -> str:
    full = ["docker", "exec", CONTAINER, "bash", "-lc", cmd]
    print(f"  [gazebo] {cmd[:110]}", flush=True)
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            sys.exit(f"Gazebo command took more than {timeout:.0f} s and was stopped:\n  {cmd}")
        return ""
    if check and r.returncode != 0:
        sys.exit(f"Command failed inside {CONTAINER}:\n  {cmd}\n{r.stderr.strip()}\n"
                 "Is the simulator running (make run in ~/Drone_Sim_Docker)?")
    return r.stdout


# ------------------------------------------------------------------ models
def _box(name, sx, sy, sz, x, y, z, yaw, rgb):
    r, g, b = rgb
    return (f"<link name='{name}'><pose>{x} {y} {z} 0 0 {yaw}</pose>"
            f"<visual name='v'><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"
            f"<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>"
            f"<specular>0.05 0.05 0.05 1</specular></material></visual></link>")


def _cyl(name, radius, z, rgb):
    r, g, b = rgb
    return (f"<link name='{name}'><pose>0 0 {z} 0 0 0</pose>"
            f"<visual name='v'><geometry><cylinder><radius>{radius}</radius><length>0.01</length>"
            f"</cylinder></geometry><material><ambient>{r} {g} {b} 1</ambient>"
            f"<diffuse>{r} {g} {b} 1</diffuse></material></visual></link>")


def green_x_sdf(name: str) -> str:
    white, green = (0.95, 0.95, 0.95), (0.12, 0.58, 0.10)
    links = _box("sheet", SHEET_M, SHEET_M, 0.01, 0, 0, 0.005, 0, white)
    links += _box("arm1", X_LEN_M, X_ARM_M, 0.01, 0, 0, 0.012, math.pi / 4, green)
    links += _box("arm2", X_LEN_M, X_ARM_M, 0.01, 0, 0, 0.013, -math.pi / 4, green)
    return f"<sdf version='1.9'><model name='{name}'><static>true</static>{links}</model></sdf>"


def green_square_sdf(name: str) -> str:  # decoy: green but not an X
    return (f"<sdf version='1.9'><model name='{name}'><static>true</static>"
            + _box("sq", 2.0, 2.0, 0.01, 0, 0, 0.005, 0, (0.12, 0.58, 0.10))
            + "</model></sdf>")


def bullseye_sdf(name: str) -> str:  # decoy: the 5 m red bullseye from the drop missions
    red, white = (0.85, 0.1, 0.1), (0.95, 0.95, 0.95)
    rings = "".join(_cyl(f"r{i}", 2.5 * f, 0.005 + 0.002 * i, red if i % 2 == 0 else white)
                    for i, f in enumerate([1.0, 0.8, 0.6, 0.4, 0.2]))
    return f"<sdf version='1.9'><model name='{name}'><static>true</static>{rings}</model></sdf>"


def spawn(name: str, sdf: str, north: float, east: float, yaw_deg: float) -> None:
    yaw = math.radians(yaw_deg)
    req = (f'sdf: "{sdf}" name: "{name}" allow_renaming: false '
           f"pose: {{position: {{x: {east}, y: {north}, z: 0.0}}, "
           f"orientation: {{w: {math.cos(yaw / 2)}, z: {math.sin(yaw / 2)}}}}}")
    out = gz(f"gz service -s /world/{WORLD}/create --reqtype gz.msgs.EntityFactory "
             f"--reptype gz.msgs.Boolean --timeout 3000 --req {shlex.quote(req)}")
    print(f"spawned {name:12s} at N {north:+.1f} m, E {east:+.1f} m, rot {yaw_deg:.0f} deg  "
          f"-> {out.strip() or '(no reply)'}")


def remove(name: str) -> None:
    req = shlex.quote('name: "%s" type: MODEL' % name)
    gz(f"gz service -s /world/{WORLD}/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean "
       f"--timeout 2000 --req {req}", check=False)


# ---------------------------------------------------------------- camera
def find_topic(fragment: str) -> str | None:
    for line in gz("gz topic -l").splitlines():
        if fragment in line:
            return line.strip()
    return None


def publish(topic: str, msg_type: str, data: str, repeat: int = 3) -> None:
    """gz topic -p sends one message right after connecting, which Gazebo can miss,
    so send it a few times, half a second apart."""
    gz(f"for i in $(seq {repeat}); do gz topic -t {topic} -m {msg_type} -p 'data: {data}'; "
       f"sleep 0.5; done")


def setup(pitch: float) -> None:
    for axis, val in (("roll", 0.0), ("yaw", 0.0), ("pitch", pitch)):
        publish(f"/gimbal/cmd_{axis}", "gz.msgs.Double", str(val))
    print(f"gimbal: roll 0, yaw 0, pitch {pitch} rad")
    topic = find_topic("enable_streaming")
    if not topic:
        sys.exit("No enable_streaming topic found. Run 'python3 sim_targets.py topics' and check "
                 "the camera topic list.")
    publish(topic, "gz.msgs.Boolean", "1", repeat=4)
    print(f"video stream enabled: {topic}\n-> H.264 RTP on udp://127.0.0.1:5600")


def autoaim(out_dir: str = "evidence") -> float:
    """Try both gimbal pitches, keep the one whose image looks like ground, save both views."""
    import pathlib
    import time

    import cv2
    import numpy as np

    from frame_source import GstUdpSource

    pathlib.Path(out_dir).mkdir(exist_ok=True)
    setup(1.5708)
    print("  opening the video stream on UDP 5600 (up to 25 s) ...", flush=True)
    try:
        src = GstUdpSource(latency_s=0.0).start()
    except RuntimeError as e:
        sys.exit(str(e))
    if not src.wait_first(20):
        src.stop()
        sys.exit("No video on UDP 5600 after enabling the stream (see README troubleshooting).")
    scores = {}
    for pitch in (1.5708, -1.5708):
        for axis, val in (("roll", 0.0), ("yaw", 0.0), ("pitch", pitch)):
            publish(f"/gimbal/cmd_{axis}", "gz.msgs.Double", str(val), repeat=2)
        time.sleep(3.0)
        frame = src.latest()[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b, g, r = [c.astype(int) for c in cv2.split(frame)]
        sky = float(np.mean((b > r + 8) & (hsv[..., 2] > 120)))
        scores[pitch] = -sky + 1e-6 * texture      # least sky wins; texture only breaks ties
        name = f"{out_dir}/camera_pitch_{'minus' if pitch < 0 else 'plus'}_1.57.png"
        cv2.imwrite(name, frame)
        print(f"pitch {pitch:+.4f}: texture {texture:8.1f}  sky {sky:5.1%}  -> saved {name}")
    src.stop()
    best = max(scores, key=scores.get)
    for axis, val in (("roll", 0.0), ("yaw", 0.0), ("pitch", best)):
        publish(f"/gimbal/cmd_{axis}", "gz.msgs.Double", str(val), repeat=2)
    print(f"camera pointed down with pitch {best:+.4f} (the view with the least sky)")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--pitch", type=float, default=1.5708,
                   help="gimbal pitch (rad) that points the camera straight down; "
                        "check with view_camera.py and try +1.5708 if it looks up/forward")
    p = sub.add_parser("spawn")
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--rot", type=float, default=0.0)
    p.add_argument("--decoys", action="store_true", help="also spawn a green square and a bullseye")
    p.add_argument("--truth", default="truth.json")
    sub.add_parser("clear")
    sub.add_parser("topics")
    sub.add_parser("autoaim", help="find the gimbal pitch that looks straight down")
    a = ap.parse_args()

    if a.cmd == "setup":
        setup(a.pitch)
    elif a.cmd == "autoaim":
        autoaim()
    elif a.cmd == "topics":
        print("\n".join(l for l in gz("gz topic -l").splitlines()
                        if "camera" in l or "gimbal" in l or "stream" in l))
    elif a.cmd == "clear":
        for n in ("green_x", "decoy_square", "decoy_bullseye"):
            remove(n)
        print("cleared targets")
    elif a.cmd == "spawn":
        for n in ("green_x", "decoy_square", "decoy_bullseye"):
            remove(n)
        spawn("green_x", green_x_sdf("green_x"), a.north, a.east, a.rot)
        truth = {"target": {"north_m": a.north, "east_m": a.east, "rot_deg": a.rot},
                 "home": {"lat": HOME[0], "lon": HOME[1]}, "decoys": []}
        lat, lon = ne_to_latlon(*HOME, a.north, a.east)
        truth["target"].update(lat=lat, lon=lon)
        if a.decoys:
            # decoys 8-10 m away from the X, on the opposite side from home
            for name, sdf, dn, de in (("decoy_square", green_square_sdf, 8.0, 6.0),
                                      ("decoy_bullseye", bullseye_sdf, 2.0, -10.0)):
                spawn(name, sdf(name), a.north + dn, a.east + de, 15.0)
                truth["decoys"].append({"name": name, "north_m": a.north + dn, "east_m": a.east + de})
        with open(a.truth, "w") as f:
            json.dump(truth, f, indent=2)
        print(f"true target lat/lon {lat:.7f}, {lon:.7f}  -> written to {a.truth}")


if __name__ == "__main__":
    main()
