# ARGUS – Green X Target Detection & Localization (PD 5 + PD 7)

**Simon Han · Purdue NSDS VIP – ARGUS drone team (C-UASC) · Autonomy & Perception**

Computer vision that finds the competition's **Package Recovery target** (a white sheet with a green X)
from a drone camera and works out its **GPS position**, then a **formal flight-test procedure** that proves
it in ArduPilot SITL + Gazebo with logs, analysis and pass/fail criteria.

- **PD 5 – Computer vision:** OpenCV green-X detector (colour + shape), geolocation from the camera ray and the drone's attitude, a labeled-dataset evaluation (v1 vs v2), and a live webcam test tool.
- **PD 7 – Flight test:** written test procedure, 3 autonomous SITL flights at 6.5 / 8 / 10 m with decoy targets, automatic log analysis against pass/fail criteria.

No ROS: Python 3, OpenCV, pymavlink, Gazebo `gz` CLI.

## Results (real logs in `results/`)

| Run | Altitude | Final position error | Detection rate | False positives | Result |
|---|---|---|---|---|---|
| R1 | 6.5 m | 0.05 m | 95.2 % | 0 / 96 | **PASS** |
| R2 | 8 m | 0.04 m | 91.4 % | 0 / 100 | **PASS** |
| R3 | 10 m | 0.04 m | 88.7 % | 0 / 103 | **PASS** |

Requirement: within 3 m (competition rule: 9 m), at ≥ 20 ft AGL, within 10 min, ≥ 80 % detection, ≤ 2 % false positives, ≥ 5 Hz.
Every run passed every criterion. Decoys (a red bullseye and a green square) were in view and never accepted.

> Honest note: these are simulator numbers. The final estimate averages ~100 detections; single
> detections were off by a median of 0.3–0.6 m. Real GPS error, wind and a real camera will make the
> error larger. The next steps are a real-camera bench test and a field test (see `Green_Cross/TEST_PROCEDURE.md`).

## How it works

1. **Colour:** green pixels (HSV + "green clearly stronger than red and blue", with a saturation floor so a green-tinted white sheet or skin doesn't count).
2. **Shape:** each green blob is checked for an X – exactly 4 arms about 90° apart (ring sampling), X-like solidity, a filled centre, and white sheet between the arms. A thin-line path (two long straight green lines crossing near their middles) handles thin, pale X's on a phone screen.
3. **Geolocation:** pixel → camera ray (pinhole model) → rotated by the camera mount and the drone's roll/pitch/yaw from MAVLink → intersected with flat ground → latitude/longitude. Many detections are averaged.
4. **Mission:** take off, lawnmower search, confirm pass over the estimate, land.

## Problems found and fixed during testing

- Simulator had no video encoder → `fix_sim_camera.sh` installs the GStreamer plugins in the Docker image.
- Camera pointed **up** at the drone's belly (green propellers caused a false detection) → camera setup now picks the view with the least sky (`results/evidence/camera_pitch_*.png`).
- Webcam: thin, pale X not detected → thin-line detection path; face falsely detected → strict colour for blobs + line-must-be-green checks; X lost in dim light + motion blur → saturation floor like the team's bullseye detector (dim + blur: 40 % → 100 % in a side-by-side test).

## Repository layout

| Path | What |
|---|---|
| `src/drone_target_cv/drone_target_cv/green_cross_detector.py` | The detector (PD 5) |
| `src/drone_target_cv/test/test_green_cross_detector.py` | Unit tests (11 pass) |
| `tools/green_cross_try.py` | Live webcam / image test tool (`--webcam`, `m` mask view, `s` save) |
| `Green_Cross/` | Flight test code: `run_test.py`, `geo.py`, `vehicle.py`, `sim_targets.py`, `analyze_runs.py`, `eval_dataset.py`, `run_all.sh`, `demo.sh` |
| `Green_Cross/TEST_PROCEDURE.md` | **PD 7 formal test procedure** |
| `results/runs/analysis/` | `RESULTS.md`, maps, timelines, error chart |
| `results/runs/R0`–`R3/`, `DEMO_1115_190214/` | Raw logs: telemetry, per-frame detections, events, camera video, snapshots |
| `results/pd5/` | Labeled-dataset metrics, v1 vs v2 chart, example montage |
| `results/evidence/` | Camera-angle fix screenshots |

Demo videos (webcam detection and a full recorded flight) are in my e-portfolio.

## Run it

```bash
# webcam test (show a green X on your phone)
python3 tools/green_cross_try.py --webcam
# full automatic test: dataset eval, sim, calibration, R1-R3, analysis
bash Green_Cross/run_all.sh
# one repeatable demo flight
bash Green_Cross/demo.sh
```
Simulator: ArduPilot Copter 4.6.3 SITL + Gazebo Harmonic in Docker (`iris_with_gimbal`, `iris_runway`).
