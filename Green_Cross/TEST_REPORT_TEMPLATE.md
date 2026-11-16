# Test Report TR-ARGUS-CV-001: Green-X Identification and Localization (SITL)

*Fill in every `[ ]` from your own runs (`runs/analysis/RESULTS.md`, `summary.csv`, figures). Delete these italic notes when you're done.*

| | |
|---|---|
| **Procedure** | TP-ARGUS-CV-001 Rev A |
| **Tester** | Simon Han |
| **Date(s)** | [ ] |
| **ARGUS_2027 commit** | [ `git -C ~/ARGUS_2027 rev-parse --short HEAD` ] |
| **This repo commit** | [ ] |
| **ArduPilot / Gazebo** | Copter 4.6.3 SITL / Gazebo Harmonic (`Drone_Sim_Docker`) |
| **Camera mount (from R0)** | [ `runs/mount.json` ] |
| **Gimbal pitch used** | [ −1.5708 or +1.5708 ] |

## 1. Summary
[ Two or three sentences: did the capability meet the requirements? Which runs passed or failed, and the headline numbers (final error, detection rate). ]

## 2. Results

| Run | Alt (m) | Final error (m) | TC1 ≤ 3 m | TC2 det. rate ≥ 80 % | TC3 FP ≤ 2 % | TC4 ≥ 6.1 m | TC5 ≤ 600 s | TC6 ≥ 5 Hz | Overall |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 6.5 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R2 | 8.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| R3 | 10.0 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

*Figures to insert:* `all_runs_error.png`, one `R*_map.png`, one `R*_timeline.png`, one detection snapshot from `runs/R*/snapshots/`.

## 3. Calibration run (R0)
[ The `--calibrate R0` output: the median error for each mount, which mount was chosen, and why. ]

## 4. Deviations from the procedure
[ Anything done differently: re-runs, the sim restarted, a different gimbal pitch, aborts and their reasons. Write "None" if there were none. ]

## 5. Analysis
- **Localization error sources:** [ video latency, attitude, pixel size at each altitude, flat-ground assumption. Which one dominated, based on how the error changes with altitude? ]
- **Detection:** [ frames missed, and why (reject reasons in `frames.csv`). Were any decoys accepted? ]
- **Failures:** [ for every FAIL: the root cause and what would fix it ]

## 6. PD 5 detector metric (labeled dataset)
*From `eval_dataset.py` and `compare_versions.py` on your machine.*

| Version | Images | Precision | Recall | F1 | Median loc. error (m) |
|---|---|---|---|---|---|
| v1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| v2 | [ ] | [ ] | [ ] | [ ] | [ ] |

[ What changed from v1 to v2 and the effect on the numbers. Where recall drops (altitude bins), and the recommended search altitude for this camera. ]

## 7. Conclusions and recommended next steps
- [ Requirement status ]
- [ e.g., bench test with the real ARGUS camera and a printed target; camera calibration; field test (TP section 13) ]
- [ Confirm the real target dimensions with the competition organizers and update `SHEET_M` / `X_*` ]
