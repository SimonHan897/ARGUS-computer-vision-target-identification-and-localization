#!/usr/bin/env python3
"""PD7 log analysis: score every test run against the pass/fail criteria.

  python3 analyze_runs.py --calibrate R0     # boresight calibration: pick the camera mount
  python3 analyze_runs.py R1 R2 R3           # analyse runs (default: every run in runs/)

Everything is recomputed from the logged pixels + poses (frames.csv), not taken
from the in-flight result, so the analysis is independent of the flight code.
Ground truth (truth.json) is only used here, never during the flight.

Outputs (runs/analysis/):
  summary.csv            one row per run with every metric and PASS/FAIL
  RESULTS.md             tables to paste into the test report
  <run>_map.png          flight track, detections, truth, estimate, 3 m / 9 m circles
  <run>_timeline.png     altitude and detections over time
  all_runs_error.png     final localization error per run vs the limits
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np

from geo import MOUNTS, Camera, ground_to_pixel, latlon_to_ne, pixel_to_ground

# ------------------------------------------------------------ pass/fail limits
RULE_LOCALIZATION_M = 9.0      # C-UASC: localize within 9 m
REQ_LOCALIZATION_M = 3.0       # team requirement: 3x margin on the rule
REQ_DETECTION_RATE = 0.80      # of frames where the whole target is in view
REQ_FALSE_POS_RATE = 0.02      # accepted detections > FP_DIST_M from truth
FP_DIST_M = 3.0
CORRECT_DIST_M = 1.5           # a detection this close to truth counts as correct
MIN_ALT_M = 6.096              # 20 ft AGL while searching
MAX_TIME_S = 600.0             # 10 min to identify + localize
MIN_RATE_HZ = 5.0              # frames processed per second
SHEET_HALF_M = 1.5

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def load_run(run: pathlib.Path):
    rows = list(csv.DictReader(open(run / "frames.csv")))
    tel = list(csv.DictReader(open(run / "telemetry.csv")))
    cfg = json.loads((run / "config.json").read_text())
    truth = json.loads((run / "truth.json").read_text()) if (run / "truth.json").exists() else None
    est = json.loads((run / "estimate.json").read_text()) if (run / "estimate.json").exists() else None
    return rows, tel, cfg, truth, est


def geolocate(rows, home, cam, mount):
    out = []
    for r in rows:
        if r["accepted"] != "1" or not r["u_px"]:
            out.append(None)
            continue
        ne = latlon_to_ne(*home, float(r["lat"]), float(r["lon"]))
        out.append(pixel_to_ground(float(r["u_px"]), float(r["v_px"]), cam, ne, float(r["alt_m"]),
                                   math.radians(float(r["roll_deg"])), math.radians(float(r["pitch_deg"])),
                                   math.radians(float(r["yaw_deg"])), mount))
    return out


def target_in_view(r, home, cam, mount, tn, te, rot_deg) -> bool:
    a = math.radians(rot_deg)
    corners = np.array([[x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)]
                        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]) * SHEET_HALF_M + [tn, te]
    ne = latlon_to_ne(*home, float(r["lat"]), float(r["lon"]))
    u, v, ok = ground_to_pixel(corners[:, 0], corners[:, 1], cam, ne, float(r["alt_m"]),
                               math.radians(float(r["roll_deg"])), math.radians(float(r["pitch_deg"])),
                               math.radians(float(r["yaw_deg"])), mount)
    return bool(np.all(ok & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)))


def robust_estimate(points):
    if len(points) == 0:
        return None, 0
    pts = np.asarray(points)
    med = np.median(pts, axis=0)
    d = np.hypot(*(pts - med).T)
    keep = d <= 3 * max(1.0, 1.4826 * float(np.median(d)))
    return np.median(pts[keep], axis=0), int(keep.sum())


def calibrate(run: pathlib.Path) -> None:
    rows, _, cfg, truth, _ = load_run(run)
    if truth is None:
        raise SystemExit(f"{run}/truth.json missing - calibration needs the known target position")
    home = (truth["home"]["lat"], truth["home"]["lon"])
    tn, te = truth["target"]["north_m"], truth["target"]["east_m"]
    cam = Camera(**cfg["camera"])
    scores = {}
    for m in MOUNTS:
        g = [x for x in geolocate(rows, home, cam, m) if x is not None]
        scores[m] = float(np.median([math.hypot(n - tn, e - te) for n, e in g])) if g else float("inf")
    best = min(scores, key=scores.get)
    print("median detection error by camera mount (m):")
    for m, s in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {m:14s} {s:8.2f}{'   <- best' if m == best else ''}")
    pathlib.Path("runs/mount.json").write_text(json.dumps(
        {"mount": best, "from_run": run.name, "median_error_m_by_mount": scores}, indent=2))
    print(f"wrote runs/mount.json (mount = {best}); use it for every later run")


def analyse(run: pathlib.Path, out: pathlib.Path) -> dict:
    rows, tel, cfg, truth, est_file = load_run(run)
    cam = Camera(**cfg["camera"])
    mount = cfg["mount"]
    if truth is None:
        raise SystemExit(f"{run}/truth.json missing")
    home = (truth["home"]["lat"], truth["home"]["lon"])
    tn, te, rot = truth["target"]["north_m"], truth["target"]["east_m"], truth["target"]["rot_deg"]
    geo = geolocate(rows, home, cam, mount)
    flight = [i for i, r in enumerate(rows) if r["phase"] in ("search", "confirm")]

    in_view = [i for i in flight if target_in_view(rows[i], home, cam, mount, tn, te, rot)]
    correct = [i for i in in_view if geo[i] is not None
               and math.hypot(geo[i][0] - tn, geo[i][1] - te) <= CORRECT_DIST_M]
    accepted = [i for i in flight if geo[i] is not None]
    false_pos = [i for i in accepted if math.hypot(geo[i][0] - tn, geo[i][1] - te) > FP_DIST_M]
    det_err = [math.hypot(geo[i][0] - tn, geo[i][1] - te) for i in accepted
               if math.hypot(geo[i][0] - tn, geo[i][1] - te) <= FP_DIST_M]
    est, used = robust_estimate([geo[i] for i in accepted])
    final_err = math.hypot(est[0] - tn, est[1] - te) if est is not None else float("inf")

    tel_alt = [(float(t["t_s"]), float(t["alt_m"])) for t in tel if t["phase"] == "search"]
    min_alt = min(a for _, a in tel_alt) if tel_alt else float("nan")
    t_takeoff = next((float(t["t_s"]) for t in tel if t["phase"] == "takeoff"), 0.0)
    t_est = est_file["estimate"]["t_estimate_s"] if est_file and est_file.get("estimate") else float("inf")
    time_to_est = t_est - t_takeoff
    flight_t = [float(rows[i]["t_s"]) for i in flight]
    rate = len(flight) / max(1e-6, flight_t[-1] - flight_t[0]) if len(flight) > 1 else 0.0
    det_rate = len(correct) / len(in_view) if in_view else float("nan")
    fp_rate = len(false_pos) / len(accepted) if accepted else 0.0
    det_ms = [float(rows[i]["det_ms"]) for i in flight if rows[i]["det_ms"]]

    checks = {
        "TC1_localization": (final_err <= REQ_LOCALIZATION_M, f"{final_err:.2f} m <= {REQ_LOCALIZATION_M} m"),
        "TC1b_rule_9m": (final_err <= RULE_LOCALIZATION_M, f"{final_err:.2f} m <= {RULE_LOCALIZATION_M} m"),
        "TC2_detection_rate": (det_rate >= REQ_DETECTION_RATE,
                               f"{det_rate:.1%} of {len(in_view)} in-view frames >= {REQ_DETECTION_RATE:.0%}"),
        "TC3_false_positives": (fp_rate <= REQ_FALSE_POS_RATE,
                                f"{len(false_pos)}/{len(accepted)} = {fp_rate:.1%} <= {REQ_FALSE_POS_RATE:.0%}"),
        "TC4_min_altitude": (min_alt >= MIN_ALT_M, f"{min_alt:.2f} m >= {MIN_ALT_M:.2f} m (20 ft)"),
        "TC5_time": (time_to_est <= MAX_TIME_S, f"{time_to_est:.0f} s <= {MAX_TIME_S:.0f} s"),
        "TC6_processing_rate": (rate >= MIN_RATE_HZ, f"{rate:.1f} Hz >= {MIN_RATE_HZ} Hz"),
    }
    m = {"run": run.name, "alt_m": cfg["alt"], "mount": mount, "target_n": tn, "target_e": te,
         "target_rot": rot, "frames": len(flight), "frames_target_in_view": len(in_view),
         "correct_detections": len(correct), "accepted_detections": len(accepted),
         "false_positives": len(false_pos), "detection_rate": round(det_rate, 4),
         "false_positive_rate": round(fp_rate, 4),
         "detection_err_median_m": round(float(np.median(det_err)), 3) if det_err else None,
         "detection_err_p95_m": round(float(np.percentile(det_err, 95)), 3) if det_err else None,
         "estimate_n": round(float(est[0]), 3) if est is not None else None,
         "estimate_e": round(float(est[1]), 3) if est is not None else None,
         "final_error_m": round(final_err, 3), "detections_used": used,
         "min_search_alt_m": round(min_alt, 3), "time_to_estimate_s": round(time_to_est, 1),
         "processing_rate_hz": round(rate, 2),
         "det_ms_median": round(float(np.median(det_ms)), 1) if det_ms else None}
    for k, (ok, why) in checks.items():
        m[k] = "PASS" if ok else "FAIL"
        m[k + "_detail"] = why
    m["overall"] = "PASS" if all(ok for k, (ok, _) in checks.items() if k != "TC1b_rule_9m") else "FAIL"
    plot_run(run, out, rows, tel, geo, flight, home, truth, est, final_err)
    return m


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_run(run, out, rows, tel, geo, flight, home, truth, est, final_err):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    tn, te = truth["target"]["north_m"], truth["target"]["east_m"]
    track = np.array([latlon_to_ne(*home, float(t["lat"]), float(t["lon"])) for t in tel])
    fig, ax = plt.subplots(figsize=(7.5, 7), facecolor=BG)
    _style(ax)
    ax.plot(track[:, 1], track[:, 0], color=INK2, lw=1, label="Flight track")
    pts = np.array([g for i, g in enumerate(geo) if g is not None and i in set(flight)])
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=10, color=S1, alpha=0.6, label="Geolocated detections")
    for r_m, ls, lab in ((REQ_LOCALIZATION_M, "-", "3 m requirement"), (RULE_LOCALIZATION_M, "--", "9 m rule")):
        ax.add_patch(plt.Circle((te, tn), r_m, fill=False, ec=INK2, ls=ls, lw=1))
        ax.text(te + r_m * 0.72, tn + r_m * 0.72, lab, fontsize=8, color=INK2)
    ax.plot(te, tn, marker="*", ms=16, color=S3, mec=INK, mew=0.6, ls="", label="True target")
    if est is not None:
        ax.plot(est[1], est[0], marker="X", ms=11, color=S2, mec=INK, mew=0.6, ls="",
                label=f"Estimate ({final_err:.2f} m error)")
    for d in truth.get("decoys", []):
        ax.plot(d["east_m"], d["north_m"], marker="s", ms=9, mfc="none", mec=S4, ls="")
        ax.text(d["east_m"] + 0.8, d["north_m"], d["name"].replace("decoy_", "decoy: "), fontsize=8, color=INK2)
    ax.plot(0, 0, marker="^", ms=9, color=INK, ls="")
    ax.text(0.8, -0.5, "home", fontsize=8, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("East of home (m)")
    ax.set_ylabel("North of home (m)")
    ax.set_title(f"{run.name}: search track and target localization", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / f"{run.name}_map.png", dpi=150, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=BG)
    _style(ax)
    t = [float(x["t_s"]) for x in tel]
    ax.plot(t, [float(x["alt_m"]) for x in tel], color=S1, lw=2)
    ax.axhline(MIN_ALT_M, color=INK2, ls="--", lw=0.8)
    ax.text(t[0], MIN_ALT_M + 0.2, "20 ft minimum", fontsize=8, color=INK2)
    dt = [float(rows[i]["t_s"]) for i in flight if geo[i] is not None]
    ax.scatter(dt, [0.3] * len(dt), marker="|", s=80, color=S2)
    ax.text(t[0], 0.7, "detections", fontsize=8, color=S2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude AGL (m)")
    ax.set_title(f"{run.name}: altitude and detection timeline", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"{run.name}_timeline.png", dpi=150, facecolor=BG)
    plt.close(fig)


def plot_all(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    _style(ax)
    names = [f"{r['run']}\n{r['alt_m']} m" for r in results]
    errs = [r["final_error_m"] for r in results]
    bars = ax.bar(names, errs, color=S1, width=0.55)
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{e:.2f} m", ha="center",
                fontsize=9, color=INK2)
    ax.axhline(REQ_LOCALIZATION_M, color=INK2, lw=1)
    ax.text(len(names) - 0.5, REQ_LOCALIZATION_M + 0.08, "3 m requirement", ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0, max(REQ_LOCALIZATION_M * 1.2, max(errs) * 1.2))
    ax.set_ylabel("Final localization error (m)")
    ax.set_title("Final target localization error per test run", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "all_runs_error.png", dpi=150, facecolor=BG)
    plt.close(fig)


def write_md(results, out):
    tcs = ["TC1_localization", "TC2_detection_rate", "TC3_false_positives", "TC4_min_altitude",
           "TC5_time", "TC6_processing_rate"]
    lines = ["# PD7 test results (generated by analyze_runs.py)", "",
             "| Run | Alt (m) | Final error (m) | Detection rate | False pos. | Min alt (m) | "
             "Time to estimate (s) | Rate (Hz) | Overall |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['alt_m']} | {r['final_error_m']} | {r['detection_rate']:.1%} | "
                     f"{r['false_positives']}/{r['accepted_detections']} | {r['min_search_alt_m']} | "
                     f"{r['time_to_estimate_s']} | {r['processing_rate_hz']} | **{r['overall']}** |")
    lines += ["", "## Pass/fail by criterion", "", "| Run | " + " | ".join(tcs) + " |",
              "|---" * (len(tcs) + 1) + "|"]
    for r in results:
        lines.append(f"| {r['run']} | " + " | ".join(f"{r[k]} ({r[k + '_detail']})" for k in tcs) + " |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run ids (folders in runs/); default all")
    ap.add_argument("--calibrate", metavar="RUN_ID")
    ap.add_argument("--out", default="runs/analysis", help="where to write the charts and RESULTS.md")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(pathlib.Path("runs") / a.calibrate)
        return
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = a.runs or sorted(p.name for p in pathlib.Path("runs").iterdir()
                           if p.is_dir() and (p / "frames.csv").exists())
    results = []
    for rid in ids:
        r = analyse(pathlib.Path("runs") / rid, out)
        results.append(r)
        print(f"\n=== {rid} (alt {r['alt_m']} m, mount {r['mount']}) -> {r['overall']}")
        for k in [k for k in r if k.startswith("TC") and not k.endswith("_detail")]:
            print(f"  {k:22s} {r[k]:4s}  {r[k + '_detail']}")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    plot_all(results, out)
    write_md(results, out)
    print(f"\nwrote {out}/summary.csv, RESULTS.md and figures")


if __name__ == "__main__":
    main()
