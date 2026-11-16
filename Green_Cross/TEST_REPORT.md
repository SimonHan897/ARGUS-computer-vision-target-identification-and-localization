# Test Report: Autonomous Green-X Target Identification and Localization

| | |
|---|---|
| **Program** | NSDS VIP – ARGUS (C-UASC), Autonomy & Perception |
| **Author** | Simon Han |
| **Test case** | CUA-TC-003 |
| **Requirement** | CUA-PER-002 |
| **Procedure** | TP-ARGUS-CV-001 (`TEST_PROCEDURE.md`) |
| **Test date** | 11/15/2026 (simulation) |
| **Status** | Simulation complete; field flight not yet run |

## 1. Objective

Verify that the drone can autonomously search for the Package Recovery target (a large white target with a green X), ignore look-alike targets, and report its GPS position accurately enough to score the mission. The C-UASC rules require the position within 9 m and within 10 minutes, without flying below 20 ft AGL.

## 2. System under test

- **Detector:** `green_cross_detector.py` v2. It checks colour (green), then shape (four arms about 90° apart, white sheet between the arms), so it rejects grass, the red bullseye and green squares.
- **Localization:** `geo.py`. The target's center pixel is projected to the ground using the camera model, the drone's attitude and altitude, then converted to latitude/longitude. Many detections are combined into one estimate.
- **Mission:** `run_test.py`. Autonomous lawnmower search at 3 m/s, a confirmation pass over the estimate, then landing (pymavlink).
- **Simulator:** ArduPilot Copter 4.6.3 SITL with Gazebo Harmonic (`iris_with_gimbal`, downward camera).

## 3. Pass/fail criteria (each run)

| ID | Criterion | Pass if |
|---|---|---|
| TC1 | Reported position error | ≤ 3 m (rule: 9 m) |
| TC2 | Detection rate when the target is fully in view | ≥ 80 % |
| TC3 | False detections (decoys or background) | ≤ 2 % |
| TC4 | Lowest altitude while searching | ≥ 20 ft (6.1 m) AGL |
| TC5 | Time from takeoff to reported position | ≤ 10 min |

## 4. Test matrix

| Run | Purpose | Altitude | Target |
|---|---|---|---|
| R0 | Camera mount calibration (not scored) | 8 m | Fixed position |
| R1 | Lowest legal altitude | 6.5 m (21 ft) | Moved and rotated, decoys in view |
| R2 | Nominal | 8 m (26 ft) | Moved and rotated, decoys in view |
| R3 | Highest altitude in the sim | 10 m (33 ft) | Moved and rotated, decoys in view |

A red bullseye and a green square were placed in every scored run as decoys.

## 5. Results

| Run | Position error | Detection rate | False detections | Lowest altitude | Time to position | Result |
|---|---|---|---|---|---|---|
| R1 | 0.05 m | 95.2 % | 0 / 96 | 6.31 m | 69 s | **PASS** |
| R2 | 0.04 m | 91.4 % | 0 / 100 | 7.72 m | 62 s | **PASS** |
| R3 | 0.04 m | 88.7 % | 0 / 103 | 9.58 m | 62 s | **PASS** |

**All 3 runs passed all criteria.** The decoys were in the camera view during the runs and were never reported as the target.

Full results, maps and timelines: [`results/runs/analysis/`](../results/runs/analysis/).

## 6. Analysis

- **Accuracy:** the final position error (0.04–0.05 m) is far inside the 3 m requirement and the 9 m rule. Single detections were off by a median of 0.3–0.6 m; combining about 100 detections into one estimate removed most of that error.
- **Detection rate:** it fell slightly as altitude increased (95 % at 6.5 m, 89 % at 10 m), because the X covers fewer pixels when the drone is higher.
- **Calibration:** run R0 compared 8 possible camera mountings. The correct one gave a median error of 0.52 m, against 1.4–4.6 m for the others.
- **Problems found and fixed during testing:**
  - The simulator sent no video because a video plugin was missing from the Docker image; `fix_sim_camera.sh` installs it.
  - The camera pointed up and saw the drone's green propellers, causing false detections; the camera setup now checks that it is looking at the ground.
  - The test froze with no error message; timeouts and clear error output were added.

## 7. Limitations

- These are simulation results. Real GPS error, wind, vibration and a real camera will make the error larger.
- The simulator's geofence capped altitude at about 12 m (40 ft), so 60 ft was not tested.
- The rules don't give the target size. The test assumed a 3 m white sheet with a green X.

## 8. Conclusion and next steps

The capability **passed in simulation** (3 of 3 runs), so it is ready to move to field testing:

1. Calibrate the real ARGUS camera and replace the simulated camera model.
2. Bench-check the detector with a printed green X under real lighting (`tools/green_cross_try.py`).
3. Run the field part of CUA-TC-003 (steps 2–8): searches at 25, 40 and 60 ft over a surveyed green X with decoys.
4. Record the field results in CUA-TC-003 and in the RVTM Verification Results for CUA-PER-002.
