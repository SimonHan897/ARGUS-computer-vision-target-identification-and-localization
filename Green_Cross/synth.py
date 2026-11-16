"""Synthetic aerial imagery with exact ground truth.

Builds a top-down ground map (asphalt / grass / dirt, runway paint, the green-X
target and distractors) and renders what a downward camera at a given pose sees,
using the same camera model as geo.py. Used for:
  * the labeled PD5 evaluation set (eval_dataset.py)
  * a "no simulator" dry run of the PD7 test harness (run_test.py --source synthetic)

Target assumption (C-UASC rules give no size for the Package Recovery target):
  white square sheet SHEET_M on a side, green X corner-to-corner, arm width ARM_M.
Change these to the real dimensions when known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from geo import Camera, MOUNTS, ground_to_pixel, rot_ned_body

SHEET_M = 3.0
ARM_M = 0.45
WHITE = (235, 235, 232)
GREEN = (45, 150, 35)          # BGR
RED = (35, 35, 205)


@dataclass
class GroundMap:
    size_m: float = 90.0
    res_m: float = 0.04
    image: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = int(self.size_m / self.res_m)
        if self.image is None:
            self.image = np.zeros((n, n, 3), np.uint8)

    @property
    def half(self) -> float:
        return self.size_m / 2.0

    def to_px(self, north: float, east: float) -> tuple[float, float]:
        """(col, row) in the map image; map is centred on (0,0), north up."""
        return (east + self.half) / self.res_m, (self.half - north) / self.res_m

    def poly(self, pts_ne: np.ndarray, color) -> None:
        px = np.array([self.to_px(n, e) for n, e in pts_ne], np.float32)
        cv2.fillPoly(self.image, [np.round(px * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)

    def circle(self, n: float, e: float, r_m: float, color) -> None:
        c = self.to_px(n, e)
        cv2.circle(self.image, (int(round(c[0] * 16)), int(round(c[1] * 16))),
                   int(round(r_m / self.res_m * 16)), color, -1, cv2.LINE_AA, shift=4)


def _rot(pts: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return pts @ r.T


def texture(rng: np.random.Generator, shape: tuple[int, int], kind: str) -> np.ndarray:
    h, w = shape
    def noise(sigma):
        g = rng.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), sigma)
        g /= (g.std() + 1e-6)
        return cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    if kind == "asphalt":
        base = np.array([88, 88, 86], np.float32)
        img = base + 10 * noise(8)[..., None] + 6 * fine[..., None]
    elif kind == "grass":
        base = np.array([45, 105, 70], np.float32) + rng.normal(0, 8, 3)
        n1 = noise(5)[..., None]
        img = base + np.array([10, 30, 14]) * n1 + 8 * fine[..., None]
    else:  # dirt
        base = np.array([95, 125, 150], np.float32)
        img = base + 14 * noise(10)[..., None] + 6 * fine[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_green_x(gm: GroundMap, n: float, e: float, rot_deg: float,
                 sheet_m: float = SHEET_M, arm_m: float = ARM_M) -> None:
    s = sheet_m / 2
    sheet = np.array([[-s, -s], [s, -s], [s, s], [-s, s]])
    gm.poly(_rot(sheet, rot_deg) + [n, e], WHITE)
    L, w = s * math.sqrt(2), arm_m / 2
    bar = np.array([[-L, -w], [L, -w], [L, w], [-L, w]])
    # draw bars clipped to the sheet: render in a temp mask
    tmp = GroundMap(gm.size_m, gm.res_m, np.zeros_like(gm.image[..., 0:1]).repeat(3, 2))
    for a in (45, 135):
        tmp.poly(_rot(_rot(bar, a), rot_deg) + [n, e], (255, 255, 255))
    sheet_mask = GroundMap(gm.size_m, gm.res_m, np.zeros_like(gm.image))
    sheet_mask.poly(_rot(sheet, rot_deg) + [n, e], (255, 255, 255))
    sel = (tmp.image[..., 0] > 127) & (sheet_mask.image[..., 0] > 127)
    gm.image[sel] = GREEN


def x_tips(n: float, e: float, rot_deg: float, sheet_m: float = SHEET_M) -> np.ndarray:
    s = sheet_m / 2
    return _rot(np.array([[-s, -s], [s, -s], [s, s], [-s, s]]), rot_deg) + [n, e]


DISTRACTORS = ("green_square", "green_tarp", "bullseye", "gcp", "grass_patch", "plus_white")


def draw_distractor(gm: GroundMap, kind: str, n: float, e: float, rng: np.random.Generator) -> None:
    rot = rng.uniform(0, 90)
    if kind == "green_square":                               # solid green, X-sized
        s = rng.uniform(1.0, 2.0)
        gm.poly(_rot(np.array([[-s, -s], [s, -s], [s, s], [-s, s]]), rot) + [n, e], GREEN)
    elif kind == "green_tarp":                               # long green rectangle
        a, b = rng.uniform(1.5, 3.0), rng.uniform(0.4, 0.9)
        gm.poly(_rot(np.array([[-a, -b], [a, -b], [a, b], [-a, b]]), rot) + [n, e], (40, 140, 60))
    elif kind == "bullseye":                                 # 5 m red bullseye (other mission)
        for frac, col in [(1.0, RED), (0.8, WHITE), (0.6, RED), (0.4, WHITE), (0.2, RED)]:
            gm.circle(n, e, 2.5 * frac, col)
    elif kind == "gcp":                                      # 1.2 m black/white GCP
        s = 0.6
        gm.poly(_rot(np.array([[-s, -s], [s, -s], [s, s], [-s, s]]), rot) + [n, e], WHITE)
        gm.poly(_rot(np.array([[-s, -s], [0, -s], [0, 0], [-s, 0]]), rot) + [n, e], (25, 25, 25))
        gm.poly(_rot(np.array([[0, 0], [s, 0], [s, s], [0, s]]), rot) + [n, e], (25, 25, 25))
    elif kind == "grass_patch":                              # irregular bright-green patch
        pts = np.array([[math.cos(t) * r, math.sin(t) * r] for t, r in
                        zip(np.linspace(0, 2 * math.pi, 14, endpoint=False), rng.uniform(1, 3, 14))])
        gm.poly(pts + [n, e], (50, 160, 60))
    elif kind == "plus_white":                               # white plus sign (no green)
        L, w = 1.2, 0.25
        for a in (0, 90):
            gm.poly(_rot(_rot(np.array([[-L, -w], [L, -w], [L, w], [-L, w]]), a), rot) + [n, e], WHITE)


def make_scene(rng: np.random.Generator, ground: str | None = None, target: bool = True,
               n_distractors: int | None = None, size_m: float = 150.0, res_m: float = 0.05):
    """Return (GroundMap, truth dict). Target at (0,0) with random rotation."""
    gm = GroundMap(size_m, res_m)
    ground = ground or rng.choice(["asphalt", "grass", "dirt", "mixed"], p=[.3, .35, .15, .2])
    h, w = gm.image.shape[:2]
    if ground == "mixed":
        a, g = texture(rng, (h, w), "asphalt"), texture(rng, (h, w), "grass")
        split = np.zeros((h, w), np.float32)
        split[:, : w // 2 + int(rng.integers(-w // 6, w // 6))] = 1
        split = cv2.GaussianBlur(split, (0, 0), 3)[..., None]
        gm.image[:] = (a * split + g * (1 - split)).astype(np.uint8)
    else:
        gm.image[:] = texture(rng, (h, w), ground)
    if ground in ("asphalt", "mixed") and rng.random() < 0.6:  # runway paint stripes
        e0 = rng.uniform(-20, 20)
        for k in range(-6, 7):
            gm.poly(np.array([[k * 6 - 1.5, e0 - .15], [k * 6 + 1.5, e0 - .15],
                              [k * 6 + 1.5, e0 + .15], [k * 6 - 1.5, e0 + .15]]), (225, 225, 225))
    truth = {"ground": str(ground), "target": target, "rot_deg": 0.0, "distractors": []}
    if target:
        rot = float(rng.uniform(0, 90))
        draw_green_x(gm, 0.0, 0.0, rot)
        truth["rot_deg"] = rot
    k = int(rng.integers(0, 4)) if n_distractors is None else n_distractors
    for _ in range(k):
        kind = str(rng.choice(DISTRACTORS))
        for _try in range(20):
            n, e = rng.uniform(-14, 14, 2)
            if math.hypot(n, e) > 5.5:
                break
        draw_distractor(gm, kind, n, e, rng)
        truth["distractors"].append((kind, float(n), float(e)))
    return gm, truth


def render(gm: GroundMap, cam: Camera, drone_ne: tuple[float, float], alt: float,
           roll: float, pitch: float, yaw: float, mount: str = "rot0",
           rng: np.random.Generator | None = None, photometric: dict | None = None) -> np.ndarray:
    """Render the camera image at the given pose (vectorised ray casting + remap)."""
    uu, vv = np.meshgrid(np.arange(cam.width, dtype=np.float32) + 0.5,
                         np.arange(cam.height, dtype=np.float32) + 0.5)
    d = np.stack([(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu)], -1)
    r = (rot_ned_body(roll, pitch, yaw) @ MOUNTS[mount]).astype(np.float32)
    dn = d @ r.T
    hit = dn[..., 2] > 1e-4
    t = np.where(hit, alt / np.maximum(dn[..., 2], 1e-4), 0)
    north = drone_ne[0] + t * dn[..., 0]
    east = drone_ne[1] + t * dn[..., 1]
    map_x = ((east + gm.half) / gm.res_m).astype(np.float32)
    map_y = ((gm.half - north) / gm.res_m).astype(np.float32)
    # constant border = plain ground colour, so nothing is mirrored past the map edge
    border = tuple(int(c) for c in np.median(gm.image[::50, ::50].reshape(-1, 3), axis=0))
    img = cv2.remap(gm.image, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    img[~hit] = (200, 170, 140)
    if photometric:
        img = apply_photometric(img, photometric, rng or np.random.default_rng())
    return img


def random_photometric(rng: np.random.Generator) -> dict:
    return {"gain": float(rng.uniform(0.6, 1.3)),
            "tint": rng.normal(0, 0.06, 3).tolist(),
            "blur_px": float(rng.choice([0, 0, 0.8, 1.5, 2.5])),
            "motion_px": int(rng.choice([0, 0, 0, 5, 9])),
            "noise": float(rng.uniform(1, 7)),
            "jpeg": int(rng.integers(35, 90)),
            "shadow": bool(rng.random() < 0.25)}


def apply_photometric(img: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32) * p.get("gain", 1.0) * (1 + np.array(p.get("tint", [0, 0, 0])))
    if p.get("shadow"):
        h, w = img.shape[:2]
        x0 = rng.uniform(0, w)
        m = np.ones((h, w), np.float32)
        m[:, int(x0):] = 0.55
        f *= cv2.GaussianBlur(m, (0, 0), 15)[..., None]
    if p.get("blur_px", 0) > 0:
        f = cv2.GaussianBlur(f, (0, 0), p["blur_px"])
    if p.get("motion_px", 0) > 1:
        k = np.zeros((p["motion_px"], p["motion_px"]), np.float32)
        k[p["motion_px"] // 2, :] = 1.0 / p["motion_px"]
        f = cv2.filter2D(f, -1, k)
    f += rng.normal(0, p.get("noise", 0), f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    q = p.get("jpeg", 0)
    if q:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def target_visibility(cam: Camera, drone_ne, alt, roll, pitch, yaw, rot_deg,
                      mount: str = "rot0", n: float = 0.0, e: float = 0.0):
    """Return (u, v, visibility) where visibility is 'full', 'partial' or 'none'."""
    pts = np.vstack([[n, e], x_tips(n, e, rot_deg)])
    u, v, ok = ground_to_pixel(pts[:, 0], pts[:, 1], cam, drone_ne, alt, roll, pitch, yaw, mount)
    inside = ok & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    if not inside[0]:
        vis = "none"
    elif inside.all():
        vis = "full"
    else:
        vis = "partial"
    return float(u[0]), float(v[0]), vis
