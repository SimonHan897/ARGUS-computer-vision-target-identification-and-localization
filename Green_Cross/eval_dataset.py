#!/usr/bin/env python3
"""PD5 quantitative evaluation of the green-X detector on a labeled synthetic set.

Every image is rendered from a known camera pose over a known scene, so each one
carries exact labels: where the X centre is in pixels, whether the whole X is in
view, and which distractors are present.

Scoring (per image)
  positive  = the whole X is in view ("full")
  correct   = detector accepted AND centre within max(6 px, 0.35 x X radius) of truth
  TP = positive & correct     FN = positive & not correct
  FP = accepted & not correct  TN = not positive & not accepted
  "partial" (X cut by the image edge) is reported separately, not in P/R.
For every TP the detected pixel is geolocated with the true pose -> ground error (m).

Usage:
  python3 eval_dataset.py --detector v1 --out eval_v1     # before
  python3 eval_dataset.py --detector v2 --out eval_v2     # after (same seed = same images)
  python3 compare_versions.py eval_v1 eval_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import time

import cv2
import numpy as np

from geo import Camera, ground_to_pixel, pixel_to_ground
from synth import make_scene, random_photometric, render, target_visibility, x_tips


def sample_pose(rng, cam, positive: bool, truth):
    alt = float(rng.uniform(6.1, 30.0))           # C-UASC: >= 20 ft AGL while searching
    roll, pitch = (float(x) for x in np.clip(rng.normal(0, 0.06, 2), -0.18, 0.18))
    yaw = float(rng.uniform(-math.pi, math.pi))
    fw, fh = cam.ground_footprint_m(alt)
    if positive:
        r = 0.45 * min(fw, fh)
        ne = (float(rng.uniform(-r, r)), float(rng.uniform(-r, r)))
    elif truth["distractors"] and rng.random() < 0.7:   # look at a distractor, X out of view
        _, n, e = truth["distractors"][int(rng.integers(len(truth["distractors"])))]
        ne = (n, e)
    else:
        ang = rng.uniform(0, 2 * math.pi)
        d = 0.75 * math.hypot(fw, fh) + rng.uniform(0, 6)
        ne = (d * math.cos(ang), d * math.sin(ang))
    return ne, alt, roll, pitch, yaw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=100)
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--save-examples", type=int, default=6)
    ap.add_argument("--detector", choices=["v1", "v2"], default="v2",
                    help="v1 = archived first version, v2 = current (for the before/after comparison)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cam = Camera()
    if args.detector == "v1":
        from archive.green_cross_detector_v1 import GreenCrossDetector, draw_debug_overlay
    else:
        from green_cross_detector import GreenCrossDetector, draw_debug_overlay
    det = GreenCrossDetector()
    print(f"detector {args.detector}, seed {args.seed}, {args.scenes} scenes x {args.poses} poses")
    rows, examples = [], {"TP": [], "FP": [], "FN": [], "TN": []}

    t_start = time.time()
    for s in range(args.scenes):
        has_target = rng.random() > 0.2               # 20% of scenes: no X at all (clutter only)
        gm, truth = make_scene(rng, target=has_target)
        for p in range(args.poses):
            want_pos = has_target and rng.random() < 0.7
            ne, alt, roll, pitch, yaw = sample_pose(rng, cam, want_pos, truth)
            photo = random_photometric(rng)
            img = render(gm, cam, ne, alt, roll, pitch, yaw, rng=rng, photometric=photo)
            if has_target:
                u, v, vis = target_visibility(cam, ne, alt, roll, pitch, yaw, truth["rot_deg"])
                tips = x_tips(0, 0, truth["rot_deg"])
                tu, tv, _ = ground_to_pixel(tips[:, 0], tips[:, 1], cam, ne, alt, roll, pitch, yaw)
                radius_px = float(np.mean(np.hypot(tu - u, tv - v)))
            else:
                u = v = float("nan")
                vis, radius_px = "none", 0.0

            t0 = time.perf_counter()
            res = det.detect(img)
            ms = (time.perf_counter() - t0) * 1000

            px_err = ground_err = float("nan")
            correct = False
            if res.accepted and vis != "none":
                px_err = math.hypot(res.center[0] - u, res.center[1] - v)
                correct = px_err <= max(6.0, 0.35 * radius_px)
                if correct:
                    g = pixel_to_ground(res.center[0], res.center[1], cam, ne, alt, roll, pitch, yaw)
                    ground_err = math.hypot(*g) if g else float("nan")
            positive = vis == "full"
            if positive:
                outcome = "TP" if correct else "FN"
                if res.accepted and not correct:
                    outcome = "FN+FP"
            elif vis == "partial":
                outcome = "partial_hit" if correct else ("FP" if res.accepted else "partial_miss")
            else:
                outcome = "FP" if res.accepted else "TN"

            lowlight = photo["gain"] < 0.8
            blurry = photo["blur_px"] >= 1.5 or photo["motion_px"] >= 5
            rows.append(dict(scene=s, pose=p, ground=truth["ground"], has_target=has_target,
                             visibility=vis, alt_m=round(alt, 2), roll_deg=round(math.degrees(roll), 1),
                             pitch_deg=round(math.degrees(pitch), 1), target_radius_px=round(radius_px, 1),
                             distractors="|".join(d[0] for d in truth["distractors"]),
                             gain=round(photo["gain"], 2), blurry=blurry, lowlight=lowlight,
                             shadow=photo["shadow"], accepted=res.accepted,
                             confidence=round(res.confidence, 3), outcome=outcome,
                             px_err=round(px_err, 2), ground_err_m=round(ground_err, 3),
                             reject_reason=res.reject_reason, det_ms=round(ms, 2)))
            key = outcome.split("+")[0] if outcome in ("TP", "FN", "FP", "FN+FP", "TN") else None
            if key and len(examples[key]) < args.save_examples:
                ann = draw_debug_overlay(img, res)
                if vis != "none":
                    cv2.circle(ann, (int(u), int(v)), 5, (0, 0, 255), -1)
                cv2.putText(ann, f"{outcome}  alt {alt:.1f} m  {truth['ground']}", (10, cam.height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                name = f"{key}_{len(examples[key]):02d}.png"
                cv2.imwrite(str(out / "examples" / name), ann)
                examples[key].append(name)
        print(f"\rscene {s + 1}/{args.scenes}", end="", flush=True)
    print(f"\nrendered + scored {len(rows)} images in {time.time() - t_start:.0f} s")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = summarize(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["overall"], indent=2))
    plot(rows, metrics, out)
    montage(out, examples)


def _pr(rows):
    tp = sum(r["outcome"] == "TP" for r in rows)
    fn = sum(r["outcome"] in ("FN", "FN+FP") for r in rows)
    fp = sum(r["outcome"] in ("FP", "FN+FP") for r in rows)
    tn = sum(r["outcome"] == "TN" for r in rows)
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * rc / (p + rc) if p == p and rc == rc and p + rc else float("nan")
    return dict(n=len(rows), TP=tp, FN=fn, FP=fp, TN=tn, precision=round(p, 4),
                recall=round(rc, 4), f1=round(f1, 4))


def summarize(rows):
    tp = [r for r in rows if r["outcome"] == "TP"]
    gerr = np.array([r["ground_err_m"] for r in tp])
    perr = np.array([r["px_err"] for r in tp])
    part = [r for r in rows if r["visibility"] == "partial"]
    overall = _pr(rows)
    overall.update(
        center_px_err_median=round(float(np.median(perr)), 2),
        center_px_err_p95=round(float(np.percentile(perr, 95)), 2),
        ground_err_m_median=round(float(np.median(gerr)), 3),
        ground_err_m_p95=round(float(np.percentile(gerr, 95)), 3),
        ground_err_m_max=round(float(gerr.max()), 3),
        partial_images=len(part),
        partial_recall=round(sum(r["outcome"] == "partial_hit" for r in part) / max(1, len(part)), 3),
        det_ms_median=round(float(np.median([r["det_ms"] for r in rows])), 1),
        det_ms_p95=round(float(np.percentile([r["det_ms"] for r in rows], 95)), 1),
    )
    by = {}
    bins = [(6.1, 10), (10, 15), (15, 20), (20, 30.01)]
    by["altitude_m"] = {f"{a:g}-{b:.0f}": _pr([r for r in rows if a <= r["alt_m"] < b]) for a, b in bins}
    by["ground"] = {g: _pr([r for r in rows if r["ground"] == g]) for g in sorted({r["ground"] for r in rows})}
    by["condition"] = {
        "clean": _pr([r for r in rows if not (r["blurry"] or r["lowlight"] or r["shadow"])]),
        "blur/motion": _pr([r for r in rows if r["blurry"]]),
        "low light": _pr([r for r in rows if r["lowlight"]]),
        "shadow edge": _pr([r for r in rows if r["shadow"]]),
    }
    fp_src = {}
    for r in rows:
        if r["outcome"] in ("FP", "FN+FP"):
            k = r["distractors"] or "background only"
            fp_src[k] = fp_src.get(k, 0) + 1
    fn_reasons = {}
    for r in rows:
        if r["outcome"] in ("FN", "FN+FP"):
            fn_reasons[r["reject_reason"]] = fn_reasons.get(r["reject_reason"], 0) + 1
    return {"overall": overall, "by": by, "false_positive_scenes": fp_src, "miss_reasons": fn_reasons}


def plot(rows, metrics, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, bg, s1, s2 = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 11, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": grid})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=bg)
    for a in (a1, a2):
        a.set_facecolor(bg); a.grid(axis="y", color=grid); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    alt = metrics["by"]["altitude_m"]
    labels = list(alt.keys())
    x = np.arange(len(labels))
    rec = [alt[k]["recall"] * 100 for k in labels]
    pre = [alt[k]["precision"] * 100 for k in labels]
    b1 = a1.bar(x - 0.19, rec, 0.36, color=s1, label="Recall")
    b2 = a1.bar(x + 0.19, pre, 0.36, color=s2, label="Precision")
    for bars in (b1, b2):
        for b in bars:
            a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", fontsize=9, color=ink2)
    a1.set_xticks(x, [f"{k} m" for k in labels]); a1.set_ylim(0, 110); a1.set_ylabel("%")
    a1.set_title("Recall and precision by altitude", loc="left", color=ink)
    a1.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    g = np.sort([r["ground_err_m"] for r in rows if r["outcome"] == "TP"])
    a2.plot(g, np.arange(1, len(g) + 1) / len(g) * 100, color=s1, lw=2)
    med = metrics["overall"]["ground_err_m_median"]; p95 = metrics["overall"]["ground_err_m_p95"]
    for val, lab in [(med, "median"), (p95, "95th pct")]:
        a2.axvline(val, color=ink2, lw=0.8, ls=(0, (3, 3)))
        a2.text(val, 8 if lab == "median" else 20, f" {lab} {val:.2f} m", color=ink2, fontsize=9)
    a2.set_xlabel("Ground localization error of detected centre (m)"); a2.set_ylabel("% of detections")
    a2.set_title("Localization error (true pose, detector only)", loc="left", color=ink)
    o = metrics["overall"]
    fig.suptitle(f"Green-X detector on {o['n']} labeled images  |  precision {o['precision']:.1%}  "
                 f"recall {o['recall']:.1%}  F1 {o['f1']:.3f}", x=0.01, ha="left",
                 fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "pd5_metrics.png", dpi=150, facecolor=bg)


def montage(out, examples):
    tiles = []
    for key in ("TP", "TN", "FP", "FN"):
        for name in examples[key][:3]:
            tiles.append(cv2.resize(cv2.imread(str(out / "examples" / name)), (320, 240)))
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.full((240, 320, 3), 250, np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(out / "pd5_examples_montage.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
