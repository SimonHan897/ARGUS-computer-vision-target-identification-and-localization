# PD 5 + PD 7: Green-X target detection, localization and SITL flight test

NSDS VIP – ARGUS (C-UASC 2026), Simon Han

- **PD 5:** build and improve a vision pipeline that detects the competition's green-X target and outputs its location, then measure it on a labeled dataset.
- **PD 7:** write a formal test procedure for that capability, run it at least 3 times in SITL, analyze the logs, and report pass/fail.

The two items are combined: the PD 7 test runs are the "controlled setup" that gives PD 5 its quantitative metric.

> The C-UASC rules describe the Package Recovery target as *"a large white target with a green X"* but give no dimensions. This code assumes a 3 m white sheet with a 0.45 m-wide X (`sim_targets.py`, `synth.py`). Update those constants when you know the real size.

---

## How the team's bullseye detector works (`ARGUS_2027/.../target_detector.py`)

`RedTargetDetector.detect(image)` runs four stages on every frame:

1. **Even out the lighting:** CLAHE on the L channel of LAB colour space, so shadows and glare change colour less.
2. **Red mask:** a pixel is "red" if its hue is in either red band (hue wraps around at 0/180) **or** it is red-dominant (R ≥ 1.35 × G and B, with enough saturation and brightness). Both tests are OR-ed together.
3. **Clean and group:** a morphological open/close removes speckle. A large closing kernel (31 px) then merges the separate red rings of a bullseye into one cluster. Each cluster becomes a candidate.
4. **Score each cluster:**
   - *circularity* = 4πA / P²;
   - *solid score*: how much of the enclosing circle is red (solid red circle);
   - *bullseye score*: cast 32 rays from the centre and count red↔white transitions; a bullseye has 3 or more;
   - *centre consistency*: the red centroid is near the circle centre.

   These are combined into one confidence. The best candidate above the thresholds wins, and the result reports the centre pixel plus the reason when nothing is accepted.

## The green-X detector (`green_cross_detector.py`, same API)

It follows the same structure as the bullseye detector, with the shape checks changed for an X:

| Stage | Bullseye | Green X |
|---|---|---|
| Colour mask | red hue OR red-dominant | **green** hue (35–90) OR green-dominant (G ≥ 1.25 × R and B) |
| Candidates | merged red clusters | each green blob, **plus "sheet first"**: find white regions, re-join them across the arms, and take only the green inside. This keeps the X separate when its tips touch surrounding grass. |
| Shape test | circularity + ring transitions | **arm count**: sample circles around the centre and count green runs (an X has 4) · **arm regularity**: the arms are about 90° apart · **solidity** (area ÷ convex hull) in the X range · **white between the arms** · **centre filled** |
| Extra | – | **small-target refinement:** tiny specks next to white are re-analyzed on a 4× upscaled crop |
| Output | centre, confidence, reason | centre, confidence, **arm orientation**, reason; `normalized_center()` uses the team's −1…+1 convention |

**v1 → v2** (driven by a failure analysis on the labeled set):
- arm-regularity gate (rejects grass clumps);
- white sheet required;
- solidity limit raised from 0.85 to 0.92 (blur thickens the arms);
- white threshold set relative to the scene's brightness (dim light);
- small-target refinement.

v1 is kept in `archive/` so you can measure the difference yourself.

**Geolocation** (`geo.py`) turns a detection into a position:
- pixel → camera ray, using the pinhole model;
- rotate the ray by the camera mount and the drone's roll/pitch/yaw (from MAVLink `ATTITUDE`);
- intersect the ray with the ground at the drone's altitude → north/east offset → lat/lon.

The camera mount isn't assumed. Run **R0** tries all 8 axis-aligned mounts and keeps the one that best matches ground truth (boresight calibration).

## Files

| File | What it is |
|---|---|
| `green_cross_detector.py` | Small shim that loads the team detector (v2) from `../src/drone_target_cv/drone_target_cv/`, so there's only one copy |
| `archive/green_cross_detector_v1.py` | First version, for the before/after comparison |
| `geo.py` | Camera model, pixel ↔ ground ↔ lat/lon, the 8 camera mounts |
| `synth.py` | Labeled synthetic aerial images (ground textures, target, decoys, blur/noise/lighting) |
| `eval_dataset.py` | **PD 5 metric:** precision / recall / F1 / localization error on a labeled set |
| `compare_versions.py` | v1-vs-v2 chart |
| `sim_targets.py` | Gazebo: spawn the X + decoys at a known position, aim the camera down, start the video stream |
| `view_camera.py` | Live Gazebo camera view with the detector overlay |
| `vehicle.py` | pymavlink vehicle (PD 3 code, with threaded telemetry + pose history) |
| `frame_source.py` | Gazebo UDP video reader (plus a synthetic source for dry runs) |
| `run_test.py` | **PD 7 test run:** search → detect → geolocate → confirm → land, with full logs |
| `analyze_runs.py` | **PD 7 analysis:** recompute everything from the logs, score TC1–TC6, make the figures and `RESULTS.md` |
| `TEST_PROCEDURE.md` | **PD 7 formal test procedure** (requirements, test matrix, steps, pass/fail) |
| `TEST_REPORT.md` | Completed PD 7 test report (SITL runs R1–R3) |
| `tests/` | Geometry unit tests (the detector tests are in `../src/drone_target_cv/test/`) |
| `setup_env.sh` | One-time host setup |

Other files added to ARGUS_2027:
- `src/drone_target_cv/drone_target_cv/green_cross_detector.py`
- `src/drone_target_cv/test/test_green_cross_detector.py`
- `tools/green_cross_try.py` (the green version of `bullseye_try.py`)

## Run order

```bash
cd ~/ARGUS_2027/Green_Cross
bash setup_env.sh                     # once; must say "GStreamer support: YES"
source .venv/bin/activate
python3 -m pytest -q tests ../src/drone_target_cv/test/test_green_cross_detector.py         # all tests pass
```

**PD 5: labeled-dataset metric (no simulator needed, about 2–3 min each)**
```bash
python3 eval_dataset.py --detector v1 --out eval_v1
python3 eval_dataset.py --detector v2 --out eval_v2      # same seed = same 600 images
python3 compare_versions.py eval_v1 eval_v2 pd5_v1_vs_v2.png
```
Outputs: `eval_v*/metrics.json`, `pd5_metrics.png`, `pd5_examples_montage.png`, `results.csv`.

**PD 7: SITL flight tests.** Follow `TEST_PROCEDURE.md` section 9 step by step. In short:
```bash
# terminal 1
cd ~/Drone_Sim_Docker && make run
# terminal 2
python3 sim_targets.py setup && python3 view_camera.py          # camera must look DOWN
python3 sim_targets.py spawn --north 18 --east -6 --rot 20      # R0
python3 run_test.py --run-id R0 --alt 8 --mount rot0
python3 analyze_runs.py --calibrate R0
python3 sim_targets.py spawn --north 22 --east 5 --rot 0 --decoys && python3 run_test.py --run-id R1 --alt 6.5
python3 sim_targets.py spawn --north 28 --east -8 --rot 35 --decoys && python3 run_test.py --run-id R2 --alt 8
python3 sim_targets.py spawn --north 15 --east 9 --rot 60 --decoys && python3 run_test.py --run-id R3 --alt 10
python3 analyze_runs.py R1 R2 R3
```

**Altitude limits:**
- **≤ 10 m:** the sim's geofence caps altitude at 12.2 m.
- **≥ 6.1 m:** the competition's 20 ft search minimum.

## E-portfolio evidence checklist

| # | Evidence | How |
|---|---|---|
| E1 | Unit tests passing | Screenshot of `pytest -q tests` |
| E2 | Camera looking down with the X detected | `view_camera.py`, press `s` |
| E3 | Screen recording of one full run (R2 is a good choice) | Print Screen → video: Gazebo + QGC + terminal |
| E4 | Calibration output | Screenshot of `analyze_runs.py --calibrate R0` |
| E5 | PASS/FAIL table | Screenshot of the `analyze_runs.py R1 R2 R3` terminal output |
| E6 | Analysis figures | `runs/analysis/*_map.png`, `all_runs_error.png`, `*_timeline.png` |
| E7 | Detector in action | `runs/R*/snapshots/*.png`, `runs/R*/annotated.mp4` |
| E8 | PD 5 metric and v1 → v2 improvement | `pd5_v1_vs_v2.png`, `eval_v2/pd5_metrics.png`, `eval_v2/pd5_examples_montage.png` |
| E9 | Documents | `TEST_PROCEDURE.md`, the completed `TEST_REPORT.md` |

## Troubleshooting

- **`No video on UDP 5600`:**
  - Run `python3 sim_targets.py setup` again.
  - Check `bash setup_env.sh` printed GStreamer **YES**.
  - Close QGC's video widget if it's also listening on port 5600.
- **The camera looks forward or up:** run `python3 sim_targets.py setup --pitch -1.5708`.
- **`left GUIDED`:** the fence triggered. Keep `--alt` at 10 or below and the search box near home.
- **`docker exec` fails:** the simulator isn't running (`make run`), or your user isn't in the `docker` group.
- **The X doesn't appear in Gazebo:** run `python3 sim_targets.py topics` and check the world name is `iris_runway`.
