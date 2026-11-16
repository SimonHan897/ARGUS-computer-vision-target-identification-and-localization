# Test Procedure TP-ARGUS-CV-001: Autonomous Green-X Target Identification and Localization

| | |
|---|---|
| **Document** | TP-ARGUS-CV-001, Rev A |
| **Program** | NSDS VIP – ARGUS (C-UASC 2026), Autonomy & Perception |
| **Author** | Simon Han |
| **PD items** | #5 Computer-vision target identification and localization · #7 Flight-test procedure, log analysis and technical documentation |
| **Test type** | Software-in-the-loop (SITL) flight test: ArduPilot Copter 4.6.3 + Gazebo Harmonic |
| **Status** | Ready for execution |

---

## 1. Purpose

This procedure verifies that the ARGUS autonomy stack can **find the green-X target, identify it, and report its position (latitude/longitude)** during an autonomous search flight. It also checks that the result meets the C-UASC accuracy, altitude and time limits.

The procedure is written so that anyone can repeat it, and so that the same steps carry over from simulation to a real flight.

## 2. Scope

**Capability under test:** autonomous search (lawnmower pattern), per-frame green-X detection, and pixel → ground → lat/lon geolocation, followed by a robust final estimate.

**In scope:**
- Detection of the green X on a white sheet.
- Rejection of decoys: a green square and the red bullseye.
- Localization accuracy.
- Altitude and time compliance.
- Processing rate.

**Out of scope:**
- Ground-control-point (numbered) targets.
- Real-world lighting and wind.
- Hardware camera calibration.

These are covered in the sim-to-real step (section 13).

## 3. References

| Ref | Document | Used for |
|---|---|---|
| R1 | C-UASC 2026 Competition Rules, rev 2025-11-10 – *Target Identification and Localization* | Localize within **9 m**; **10 min** to identify, classify and localize |
| R2 | C-UASC 2026 Rules – altitude limits | Do not descend below **20 ft (6.1 m) AGL** while searching |
| R3 | C-UASC 2026 Rules – *Package Recovery* | Target is a **large white target with a green X** (no dimensions given) |
| R4 | Individual Semester Development Plan, items 5 and 7 | Completion criteria |
| R5 | `README.md` in this repository | Software description and setup |

## 4. Test article and configuration

Record the exact versions in the report (section 12).

| Item | Configuration |
|---|---|
| Detector | `green_cross_detector.py` **v2** in `ARGUS_2027/src/drone_target_cv/drone_target_cv/` |
| Geolocation | `geo.py`: pinhole camera model plus roll/pitch/yaw, intersected with flat ground |
| Mission / logging | `run_test.py` (pymavlink, built from PD 3) |
| Analysis | `analyze_runs.py` |
| Autopilot | ArduPilot Copter 4.6.3 SITL in `Drone_Sim_Docker` |
| Simulator | Gazebo Harmonic, world `iris_runway`, model `iris_with_gimbal` |
| Camera | gimbal_small_3d: 640 × 480 px, horizontal FOV 2.0 rad, 10 Hz, H.264 over UDP 5600. Gimbal held fixed to the body, pointing down (not stabilized). |
| Target (assumed) | White sheet 3.0 × 3.0 m with an inset green X: arms 3.6 m long and 0.45 m wide. The rules give no size; update this row when the real size is known. |
| Decoys | Green square 2 × 2 m; red/white bullseye 5 m diameter |
| Geofence | `FENCE_ALT_MAX` = 12.2 m, so all test altitudes are ≤ 10 m |

## 5. Requirements and pass/fail criteria

Each run is scored by `analyze_runs.py` using these definitions. Ground truth is used **only** in the analysis, never in flight.

| ID | Requirement | Source | Measure | Pass if |
|---|---|---|---|---|
| **TC1** | Localize the target accurately | R1 (9 m), with a 3× team margin | Horizontal distance from the final estimate to the true target | **≤ 3.0 m** (also report ≤ 9 m) |
| **TC2** | Detect the target reliably when it is in view | Team | Correct detections ÷ frames with the whole target in view, in the search and confirm phases. "Correct" means geolocated within 1.5 m of truth. | **≥ 80 %** |
| **TC3** | Do not report decoys or background as the target | R1 (classification) | Accepted detections geolocated more than 3 m from truth ÷ all accepted detections | **≤ 2 %** |
| **TC4** | Respect the minimum search altitude | R2 | Lowest altitude AGL logged during the search phase | **≥ 6.1 m (20 ft)** |
| **TC5** | Finish within the competition time | R1 | Time from takeoff to final estimate | **≤ 600 s** |
| **TC6** | Process camera frames fast enough | Team | Frames processed ÷ flight time | **≥ 5 Hz** |

A run **passes overall** when TC1–TC6 all pass.

## 6. Test environment and equipment

- Ubuntu 22.04/24.04 workstation (club-drive PC) with Docker, the `Drone_Sim_Docker` image, and QGroundControl 5.0.8.
- This repository at `~/ARGUS_2027/Green_Cross` with the virtual environment from `setup_env.sh`.
- A screen recorder (GNOME, using **Print Screen → video**) for evidence.

## 7. Safety and abort criteria

These are simulated, but the rules are written as they would be for a real flight.

| Condition | Automatic action |
|---|---|
| Any step times out | `run_test.py` commands **LAND** |
| Vehicle leaves GUIDED mode (fence breach or failsafe) | **LAND** |
| Operator presses Ctrl+C | **LAND** |
| Video stream lost before takeoff | Run is not started |
| Altitude below 6.1 m during search | Recorded; TC4 fails (no automatic action) |

The operator also watches QGroundControl throughout. If the vehicle behaves unexpectedly, press **Ctrl+C**.

## 8. Test matrix

| Run | Purpose | Alt (m) | Target N, E (m from home) | Target rot (°) | Decoys |
|---|---|---|---|---|---|
| **R0** | Boresight calibration (choose the camera mount) | 8.0 | 18, −6 | 20 | no |
| **R1** | Minimum legal search altitude | 6.5 | 22, +5 | 0 | yes |
| **R2** | Nominal | 8.0 | 28, −8 | 35 | yes |
| **R3** | Highest altitude allowed by the fence | 10.0 | 15, +9 | 60 | yes |

The search box for every run is N 5–35 m, E −15 to +15 m from home, flown at 3 m/s.

R0 is only used to select the camera mount. It is reported, but it does not count toward pass/fail.

## 9. Procedure

### 9.1 One-time setup
- [ ] 1. `cd ~/ARGUS_2027/Green_Cross && bash setup_env.sh`. It must print `GStreamer support: YES`.
- [ ] 2. `source .venv/bin/activate && python3 -m pytest -q tests ../src/drone_target_cv/test/test_green_cross_detector.py`. All tests must pass. Screenshot the output for the evidence package (E1).

### 9.2 Start the simulator (before every test session)
- [ ] 3. Terminal 1: `cd ~/Drone_Sim_Docker && make run`. Wait for the Iris in Gazebo and **Ready to Fly** in QGC.
- [ ] 4. Terminal 2: `cd ~/ARGUS_2027/Green_Cross && source .venv/bin/activate`
- [ ] 5. `python3 sim_targets.py setup`. This sets the gimbal angle and starts the video stream.
- [ ] 6. `python3 view_camera.py`. With the drone on the ground, the view must show the **ground under the drone** (runway or grass), not the sky or horizon.
  - If it doesn't, run `python3 sim_targets.py setup --pitch -1.5708` and check again.
  - Record the pitch value that works.
  - Press `s` to save a screenshot (E2), then `q` to quit.

### 9.3 Each run (repeat for R0, R1, R2, R3 using the matrix in section 8)
- [ ] 7. Spawn the target, for example for R1:
  `python3 sim_targets.py spawn --north 22 --east 5 --rot 0 --decoys`
  - Confirm the X (and the decoys) appear in Gazebo.
  - `truth.json` is written; don't open it until the analysis.
- [ ] 8. Start the screen recording (E3).
- [ ] 9. Start the run, for example: `python3 run_test.py --run-id R1 --alt 6.5`
  - For R0 only, add `--mount rot0`.
- [ ] 10. Watch the run. The terminal should show:
  - takeoff → `SEARCH` → search waypoints → `FIRST DETECTION` → `RESULT: target estimate …` → return → `Landed and DISARMED` → `RUN COMPLETE`.
  - Record any `ABORT` and its reason.
- [ ] 11. Stop the recording. Check that `runs/<id>/` contains `frames.csv`, `telemetry.csv`, `annotated.mp4` and `estimate.json`.
- [ ] 12. **After R0 only:** `python3 analyze_runs.py --calibrate R0`. This writes `runs/mount.json`, which later runs use automatically. Screenshot the output (E4).
- [ ] 13. Wait for **Ready to Fly** before the next run. If the vehicle doesn't re-arm, restart the sim (step 3, then step 5).

### 9.4 Analysis
- [ ] 14. `python3 analyze_runs.py R1 R2 R3`. Screenshot the PASS/FAIL table in the terminal (E5).
- [ ] 15. Collect `runs/analysis/RESULTS.md`, `summary.csv`, `*_map.png`, `*_timeline.png` and `all_runs_error.png` (E6).

## 10. Data recorded

| File | Content | Rate |
|---|---|---|
| `runs/<id>/frames.csv` | One row per processed frame: pose, detection result, pixel, scores, geolocated N/E and lat/lon, processing time | ~10 Hz |
| `runs/<id>/telemetry.csv` | Mode, armed state, lat/lon, altitude, ground speed, attitude | 10 Hz |
| `runs/<id>/events.log` | Everything printed during the run | – |
| `runs/<id>/config.json` | All settings, including the detector configuration and camera mount | – |
| `runs/<id>/annotated.mp4`, `snapshots/` | Camera video with the detector overlay | 10 Hz |
| `runs/<id>/estimate.json` | In-flight estimate and timing | – |
| `runs/<id>/truth.json` | True target position (analysis only) | – |

## 11. Analysis method

1. **Geolocation is recomputed** from the logged pixel plus pose for every accepted detection, using the calibrated mount. The analysis therefore doesn't depend on the flight code's own answer.
2. **Final estimate** = median of all detections after rejecting outliers more than 3 MAD (at least 1 m) from the median. This is the same rule used in flight.
3. **"Target in view"** means all four sheet corners project inside the image at the frame's logged pose. That is the denominator for TC2.
4. TC1–TC6 are computed as in section 5. A run passes overall when all six pass.
5. **Error sources to discuss in the report:**
   - Video latency (about 0.1 s) at 3 m/s ≈ 0.3 m.
   - Attitude error.
   - Pixel quantization: about 0.04 m/px at 6.5 m and 0.06 m/px at 10 m.
   - Flat-ground assumption.

## 12. Report

The completed report is `TEST_REPORT.md`. It was written using `RESULTS.md` and the figures. Include:
- versions (git commit of ARGUS_2027, ArduPilot version);
- any deviations from this procedure;
- the pass/fail table;
- analysis of failures;
- recommended next steps.

## 13. Progression to real flight (sim-to-real)

After all SITL runs pass:
1. **Bench test:** run `tools/green_cross_try.py` with the ARGUS camera aimed at a printed green X under real lighting. Tune the HSV settings if needed.
2. **Camera calibration:** use the team's `camera_calibration.md`. Replace the FOV model with the calibrated `fx`/`fy` values.
3. **Field test** at low altitude over the real target, with a surveyed target position:
   - Add a real-camera frame source with the same `latest()` interface as `GstUdpSource`.
   - Then run steps 9–15 unchanged.
