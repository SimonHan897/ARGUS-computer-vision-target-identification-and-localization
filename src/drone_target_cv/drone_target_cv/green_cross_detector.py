"""Pure OpenCV detector for the C-UASC green-X target (white sheet with a green X).

Same shape as ``target_detector.RedTargetDetector`` (config dataclass ->
``detect(image)`` -> result with ``center`` / ``confidence`` / ``reject_reason``
and a ``draw_debug_overlay`` helper), so mission code can swap detectors.

Pipeline
--------
1. Light normalisation      CLAHE on the L channel (same as the bullseye detector).
2. Green mask               HSV hue band  OR  "green-dominant" pixels (G >> R, B).
3. Mask clean-up            open (remove speckle) + close (bridge thin arms).
4. Candidates, two ways:
     a) each green blob on its own
     b) "sheet first": find white (bright, low-saturation) regions, fill their
        convex hull, and take only the green inside. This keeps the X separate
        from surrounding grass when the arm tips touch the sheet edge.
5. Score each blob as an X:
     arm_count     sample a ring around the blob centre and count green runs -> an X has 4
     arm_regularity the 4 arms should be ~90 deg apart
     solidity      area / convex-hull area -> an X sits around 0.45-0.75
                   (a solid green square is ~1.0, grass is ragged)
     white_backing a band just outside the X should be bright, low-saturation (the white sheet)
     center_fill   the crossing point of an X is green
6. v2 small-target refinement: specks too small to survive step 3 that touch
   something white are re-analysed on a 4x upscaled crop.
7. Thin-line path: two long green line segments (Hough) that cross near both of
   their middles at 45-135 deg with white between the arms (the C-UASC rules
   drawing uses thin lines, which the blob tests break up).
8. Pick the best blob above the thresholds; report centre pixel, confidence,
   arm orientation and the reason when nothing is accepted.

v1 -> v2 changes (from the PD5 failure analysis): arm-regularity gate, white
sheet required, solidity limit 0.85 -> 0.92, brightness-relative white
threshold, small-target refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class GreenCrossDetectorConfig:
    """Tunable settings. Defaults are tuned on the synthetic set + Gazebo."""

    # size gates
    min_target_area_px: float = 40.0
    max_target_area_frac: float = 0.6        # reject blobs covering most of the frame
    # decision thresholds
    min_detection_confidence: float = 0.55
    min_arm_score: float = 0.5               # 1.0 = exactly 4 arms, 0.5 = 3 or 5 (partial / occluded)
    min_arm_regularity: float = 0.6          # v2: arms must be roughly 90 deg apart (grass is not)
    min_solidity: float = 0.30
    max_solidity: float = 0.92               # v2: was 0.85; blur thickens arms
    min_white_backing: float = 0.30          # v2: was 0 (off); the target IS a white sheet
    # pre-processing
    use_light_normalization: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size_px: int = 8
    hsv_blur_kernel_px: int = 3
    morph_open_px: int = 3
    morph_close_px: int = 5
    # green colour gates (OpenCV hue is 0-180)
    hsv_h_min: int = 35
    hsv_h_max: int = 90
    hsv_s_min: int = 70
    hsv_s_max: int = 255
    hsv_v_min: int = 40
    hsv_v_max: int = 255
    green_dominance_ratio: float = 1.25
    green_difference_min: int = 25
    green_min_channel: int = 60
    # like the bullseye's red_dominance_s_min / v_min: a "greener than red and blue" pixel must
    # also be saturated, so a white sheet or skin under a green colour cast does not count.
    # 0 = off (drone default, unchanged); the webcam preset turns it on.
    green_dominance_s_min: int = 0
    green_dominance_v_min: int = 0
    # ring sampling for arm counting
    ring_radius_fracs: tuple[float, ...] = (0.55, 0.7, 0.85)
    ring_samples: int = 180
    expected_arms: int = 4
    # white-backing: the wedges BETWEEN the arms (annulus, multiples of enclosing radius)
    backing_inner_frac: float = 0.30
    backing_outer_frac: float = 0.85
    backing_v_min: int = 150                 # upper bound; v2 lowers it in dim images
    backing_v_rel: float = 0.80              # v2: white >= this x the 99th-percentile brightness
    backing_s_max: int = 70
    # two-stage path: find the white sheet first, then the green inside it.
    # Needed when the X touches surrounding grass (blobs would otherwise merge).
    use_sheet_search: bool = True
    sheet_min_area_px: float = 150.0
    # closing kernels used to re-join the white sheet across the X arms
    # (a corner-to-corner X splits the sheet into 4 triangles); several sizes
    # so it works from far (thin arms) to close (thick arms)
    sheet_close_px: tuple[int, ...] = (7, 15, 31)
    # v2 small-target refinement: blobs smaller than this are re-analysed on an
    # upscaled crop (thin arms vanish in the full-resolution clean-up)
    refine_radius_px: float = 16.0
    refine_scale: int = 4
    refine_max_seeds: int = 8
    # thin-line X path: two long green line segments that cross (the C-UASC rules
    # drawing uses thin lines, which blob-shape tests break up)
    use_line_search: bool = False            # off for aerial (runway paint can look like lines); webcam preset turns it on
    line_min_len_frac: float = 0.04          # min segment length, fraction of image width
    line_max_gap_px: int = 10
    line_min_angle_deg: float = 45.0         # angle between the two lines: 45..135 deg
    line_min_white: float = 0.45             # white between the arms near the crossing
    # the thin-line path may use a looser green test (pale lines on a phone/print), but
    # only the line path: blobs keep the strict colour test above
    line_hsv_s_min: int = 70
    line_green_dominance_ratio: float = 1.25
    line_green_difference_min: int = 25
    line_min_green_along: float = 0.7        # each line must be green along >= 70% of its length
    line_min_green_contrast: float = 12.0    # line must be greener than the white beside it
                                             # (G - max(R, B), in 0-255 units); beats colour casts


@dataclass(frozen=True)
class GreenCrossCandidate:
    center: tuple[int, int]
    radius_px: float
    area_px: float
    confidence: float
    arm_count: int
    arm_score: float
    arm_regularity: float
    solidity: float
    solidity_score: float
    white_backing: float
    center_fill: float
    orientation_deg: float
    contour: np.ndarray
    source: str = "blob"     # "blob" (filled X) or "lines" (thin-line X found as two crossing lines)


@dataclass(frozen=True)
class GreenCrossResult:
    center: tuple[int, int] | None
    confidence: float
    radius_px: float
    area_px: float
    arm_count: int
    arm_score: float
    arm_regularity: float
    solidity: float
    white_backing: float
    center_fill: float
    orientation_deg: float
    blob_count: int
    reject_reason: str
    green_mask: np.ndarray
    contour: np.ndarray | None

    @property
    def accepted(self) -> bool:
        return self.center is not None

    def normalized_center(self, width: int, height: int) -> tuple[float, float] | None:
        """Centre in the team's convention: x,y in [-1, 1], (0,0) = image centre."""
        if self.center is None:
            return None
        return ((self.center[0] - width / 2.0) / (width / 2.0),
                (self.center[1] - height / 2.0) / (height / 2.0))


class GreenCrossDetector:
    """Detect a green X (on a white sheet) in a BGR image."""

    def __init__(self, config: GreenCrossDetectorConfig | None = None) -> None:
        self.config = config or GreenCrossDetectorConfig()

    # ------------------------------------------------------------------ API
    def detect(self, image: np.ndarray) -> GreenCrossResult:
        bgr = _ensure_bgr(image)
        cands, green, blob_count, seeds = self._candidates(bgr)
        if self.config.refine_scale > 1:
            cands.extend(self._refine_small(bgr, seeds))

        best = best_rejected = None
        for cand in cands:
            if best_rejected is None or cand.confidence > best_rejected.confidence:
                best_rejected = cand
            if self._passes(cand) and (best is None or cand.confidence > best.confidence):
                best = cand
        if best is not None:
            return self._result(best, blob_count, "accepted", green, accepted=True)
        if best_rejected is not None:
            return self._result(best_rejected, blob_count, self._reject_reason(best_rejected),
                                green, accepted=False)
        if cv2.countNonZero(green) == 0:
            reason = "no green pixels after thresholding"
        elif blob_count == 0:
            reason = "green pixels removed by mask cleanup"
        else:
            reason = "green blobs too small or too large"
        return GreenCrossResult(None, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                blob_count, reason, green, None)

    def _candidates(self, bgr: np.ndarray):
        """Score every candidate blob. Also return 'seeds' for small-target refinement."""
        c = self.config
        normalized = self._normalize_light(bgr)
        raw_green = self._green_mask(normalized, bgr)
        green = self._clean_mask(raw_green)
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

        h, w = green.shape
        max_area = c.max_target_area_frac * h * w
        n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
        blob_count = max(0, n - 1)

        blobs: list[np.ndarray] = []
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if c.min_target_area_px <= area <= max_area:
                blobs.append(np.where(labels == i, 255, 0).astype(np.uint8))
        if c.use_sheet_search:
            blobs.extend(self._green_inside_sheets(hsv, green))

        cands = []
        for blob in blobs:
            area = float(cv2.countNonZero(blob))
            if area < c.min_target_area_px or area > max_area:
                continue
            cand = self._score(blob, area, hsv)
            if cand is not None:
                cands.append(cand)
        if c.use_line_search:
            line_cfg = replace(c, hsv_s_min=c.line_hsv_s_min,
                               green_dominance_ratio=c.line_green_dominance_ratio,
                               green_difference_min=c.line_green_difference_min,
                               green_dominance_s_min=0, green_dominance_v_min=0)
            line_green = GreenCrossDetector(line_cfg)._green_mask(normalized, bgr)
            cands.extend(self._line_candidates(line_green, hsv, bgr))

        # seeds: small raw-green specks that sit on/next to something white
        seeds: list[tuple[float, float, float]] = []
        n2, _, st2, cen2 = cv2.connectedComponentsWithStats(raw_green, 8)
        white = self._white_mask(hsv)
        for i in np.argsort(-st2[1:, cv2.CC_STAT_AREA])[: 4 * c.refine_max_seeds] + 1 if n2 > 1 else []:
            area = float(st2[i, cv2.CC_STAT_AREA])
            r = 0.5 * math.hypot(st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT])
            if area < 4 or r > c.refine_radius_px:
                continue
            x0, y0 = st2[i, cv2.CC_STAT_LEFT], st2[i, cv2.CC_STAT_TOP]
            pad = 3
            win = white[max(0, y0 - pad):y0 + st2[i, cv2.CC_STAT_HEIGHT] + pad,
                        max(0, x0 - pad):x0 + st2[i, cv2.CC_STAT_WIDTH] + pad]
            if win.size and np.count_nonzero(win) >= 3:
                seeds.append((float(cen2[i][0]), float(cen2[i][1]), r))
            if len(seeds) >= c.refine_max_seeds:
                break
        return cands, green, blob_count, seeds

    def _refine_small(self, bgr: np.ndarray, seeds) -> list[GreenCrossCandidate]:
        """Re-run detection on an upscaled crop around each small seed."""
        c = self.config
        k = int(c.refine_scale)
        sub = GreenCrossDetector(replace(
            c, refine_scale=1, morph_open_px=1, morph_close_px=3 * k // 2,
            min_target_area_px=c.min_target_area_px, hsv_blur_kernel_px=k + 1,
            sheet_close_px=tuple(int(x * k / 2) for x in c.sheet_close_px)))
        h, w = bgr.shape[:2]
        out = []
        for sx, sy, r in seeds:
            half = int(max(12, 2.5 * r))
            x0, y0 = max(0, int(sx) - half), max(0, int(sy) - half)
            x1, y1 = min(w, int(sx) + half + 1), min(h, int(sy) + half + 1)
            crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            cands, _, _, _ = sub._candidates(crop)
            for cd in cands:
                cx, cy = x0 + (cd.center[0] + 0.5) / k - 0.5, y0 + (cd.center[1] + 0.5) / k - 0.5
                contour = (cd.contour.astype(np.float32) / k + [x0, y0]).astype(np.int32)
                out.append(replace(cd, center=(int(round(cx)), int(round(cy))),
                                   radius_px=cd.radius_px / k, area_px=cd.area_px / (k * k),
                                   contour=contour))
        return out

    # ------------------------------------------------------- mask building
    def _normalize_light(self, bgr: np.ndarray) -> np.ndarray:
        c = self.config
        if not c.use_light_normalization:
            return bgr
        tiles = max(2, int(c.clahe_tile_grid_size_px))
        clahe = cv2.createCLAHE(clipLimit=max(0.1, float(c.clahe_clip_limit)),
                                tileGridSize=(tiles, tiles))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    def _green_mask(self, normalized: np.ndarray, raw_bgr: np.ndarray) -> np.ndarray:
        c = self.config
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        k = _odd(c.hsv_blur_kernel_px)
        if k > 1:
            hsv = cv2.GaussianBlur(hsv, (k, k), 0)
        hsv_mask = cv2.inRange(hsv, np.array([c.hsv_h_min, c.hsv_s_min, c.hsv_v_min], np.uint8),
                               np.array([c.hsv_h_max, c.hsv_s_max, c.hsv_v_max], np.uint8))
        b, g, r = [ch.astype(np.float32) for ch in cv2.split(raw_bgr)]
        ratio = max(1.0, float(c.green_dominance_ratio))
        dom = ((g >= c.green_min_channel) & (g >= r * ratio) & (g >= b * ratio)
               & ((g - np.maximum(r, b)) >= c.green_difference_min))
        if c.green_dominance_s_min > 0 or c.green_dominance_v_min > 0:
            dom &= (hsv[..., 1] >= c.green_dominance_s_min) & (hsv[..., 2] >= c.green_dominance_v_min)
        dom_mask = np.where(dom, 255, 0).astype(np.uint8)
        return cv2.bitwise_or(hsv_mask, dom_mask)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        c = self.config
        if _odd(c.morph_open_px) > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(c.morph_open_px),) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if _odd(c.morph_close_px) > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(c.morph_close_px),) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _white_v_min(self, hsv: np.ndarray) -> int:
        """v2: in dim images the sheet is grey, so scale the threshold to the scene."""
        c = self.config
        p99 = float(np.percentile(hsv[..., 2][::4, ::4], 99))
        return int(max(70, min(c.backing_v_min, c.backing_v_rel * p99)))

    def _white_mask(self, hsv: np.ndarray) -> np.ndarray:
        c = self.config
        return cv2.inRange(hsv, np.array([0, 0, self._white_v_min(hsv)], np.uint8),
                           np.array([180, c.backing_s_max, 255], np.uint8))

    def _green_inside_sheets(self, hsv: np.ndarray, green: np.ndarray) -> list[np.ndarray]:
        """Stage-2 candidates: green pixels inside the convex hull of each white region."""
        c = self.config
        white = self._white_mask(hsv)
        out: list[np.ndarray] = []
        seen: set[tuple[int, int, int]] = set()
        for k in c.sheet_close_px:
            k = _odd(k)
            # closing (dilate then erode) bridges the arm gaps but keeps the outer edge
            joined = cv2.morphologyEx(
                white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < c.sheet_min_area_px:
                    continue
                hull = np.zeros_like(green)
                cv2.fillConvexPoly(hull, cv2.convexHull(cnt), 255)
                inside = cv2.bitwise_and(green, hull)
                if cv2.countNonZero(inside) < c.min_target_area_px:
                    continue
                n, lab, st, cen = cv2.connectedComponentsWithStats(inside, 8)
                if n <= 1:
                    continue
                j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                key = (int(cen[j][0]) // 4, int(cen[j][1]) // 4, int(st[j, cv2.CC_STAT_AREA]) // 16)
                if key in seen:
                    continue
                seen.add(key)
                out.append(np.where(lab == j, 255, 0).astype(np.uint8))
        return out

    def _line_candidates(self, raw_green: np.ndarray, hsv: np.ndarray,
                         bgr: np.ndarray) -> list[GreenCrossCandidate]:
        """Thin-line X: pairs of long green segments that cross near both their middles."""
        c = self.config
        h, w = raw_green.shape
        min_len = max(12, int(c.line_min_len_frac * w))
        white = self._white_mask(hsv)
        # only green that touches white (the target is drawn on a white sheet); this
        # removes the colour noise of dim / uncalibrated cameras before line finding
        near_white = cv2.dilate(white, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        green_on_white = cv2.bitwise_and(raw_green, near_white)
        lines = cv2.HoughLinesP(green_on_white, 1, np.pi / 180, threshold=max(10, min_len // 2),
                                minLineLength=min_len, maxLineGap=c.line_max_gap_px)
        if lines is None:
            return []
        segs = [tuple(map(float, l)) for l in np.asarray(lines).reshape(-1, 4)]
        segs.sort(key=lambda q: -math.hypot(q[2] - q[0], q[3] - q[1]))   # longest first
        segs = segs[:50]

        def green_along(q) -> float:
            n = max(2, int(math.hypot(q[2] - q[0], q[3] - q[1])))
            xs = np.clip(np.round(np.linspace(q[0], q[2], n)).astype(int), 0, w - 1)
            ys = np.clip(np.round(np.linspace(q[1], q[3], n)).astype(int), 0, h - 1)
            # allow 1 px misalignment of the Hough line
            g = cv2.dilate(raw_green, np.ones((3, 3), np.uint8))
            return float(np.count_nonzero(g[ys, xs])) / n

        segs = [q for q in segs if green_along(q) >= c.line_min_green_along]
        b_, g_, r_ = [ch.astype(np.int16) for ch in cv2.split(bgr)]
        greenness = g_ - np.maximum(r_, b_)                      # how much greener than red/blue

        def line_greenness(q) -> float:
            n = max(2, int(math.hypot(q[2] - q[0], q[3] - q[1])))
            xs = np.clip(np.round(np.linspace(q[0], q[2], n)).astype(int), 0, w - 1)
            ys = np.clip(np.round(np.linspace(q[1], q[3], n)).astype(int), 0, h - 1)
            return float(np.median(greenness[ys, xs]))
        out: list[GreenCrossCandidate] = []
        for i in range(len(segs)):
            x1, y1, x2, y2 = segs[i]
            d1 = np.array([x2 - x1, y2 - y1])
            l1 = float(np.hypot(*d1))
            for j in range(i + 1, len(segs)):
                x3, y3, x4, y4 = segs[j]
                d2 = np.array([x4 - x3, y4 - y3])
                l2 = float(np.hypot(*d2))
                cross = d1[0] * d2[1] - d1[1] * d2[0]
                sin_a = abs(cross) / (l1 * l2 + 1e-9)
                angle = math.degrees(math.asin(min(1.0, sin_a)))
                if angle < c.line_min_angle_deg:
                    continue
                # intersection parameters along each segment (0..1 = inside)
                t = ((x3 - x1) * d2[1] - (y3 - y1) * d2[0]) / cross
                u = ((x3 - x1) * d1[1] - (y3 - y1) * d1[0]) / cross
                # an X: the crossing is well inside BOTH segments (not a T or an L)
                if not (0.25 <= t <= 0.75 and 0.25 <= u <= 0.75):
                    continue
                cx, cy = x1 + t * d1[0], y1 + t * d1[1]
                balance = min(l1, l2) / max(l1, l2)
                if balance < 0.4:
                    continue
                radius = 0.5 * max(l1, l2)
                # white between the arms: annulus around the crossing, minus the lines.
                # Work on a small patch around the crossing (much faster than full-frame images).
                rr = max(3, int(0.45 * radius))
                x0, y0 = max(0, int(cx) - rr - 2), max(0, int(cy) - rr - 2)
                x1p, y1p = min(w, int(cx) + rr + 3), min(h, int(cy) + rr + 3)
                if x1p - x0 < 3 or y1p - y0 < 3:
                    continue
                pc = (int(cx) - x0, int(cy) - y0)
                ring = np.zeros((y1p - y0, x1p - x0), np.uint8)
                cv2.circle(ring, pc, rr, 255, -1)
                cv2.circle(ring, pc, max(1, int(0.15 * radius)), 0, -1)
                cv2.line(ring, (int(x1) - x0, int(y1) - y0), (int(x2) - x0, int(y2) - y0), 0, 5)
                cv2.line(ring, (int(x3) - x0, int(y3) - y0), (int(x4) - x0, int(y4) - y0), 0, 5)
                sel = ring > 0
                n = int(np.count_nonzero(sel))
                white_frac = float(np.count_nonzero(white[y0:y1p, x0:x1p][sel])) / n if n else 0.0
                if white_frac < c.line_min_white:
                    continue
                # colour contrast: both lines clearly greener than the white between the arms
                wsel = white[y0:y1p, x0:x1p][sel] > 0
                if not wsel.any():
                    continue
                bg_green = float(np.median(greenness[y0:y1p, x0:x1p][sel][wsel]))
                contrast = min(line_greenness(segs[i]), line_greenness(segs[j])) - bg_green
                if contrast < c.line_min_green_contrast:
                    continue
                centrality = 1.0 - (abs(t - 0.5) + abs(u - 0.5))      # 1 = crossing at both middles
                regularity = 1.0 - abs(90.0 - angle) / 45.0
                confidence = float(np.clip(0.35 * centrality + 0.25 * balance
                                           + 0.25 * white_frac + 0.15 * max(0.0, regularity), 0, 1))
                ang1 = math.degrees(math.atan2(d1[1], d1[0])) % 90.0
                contour = np.array([[[int(x1), int(y1)]], [[int(cx), int(cy)]], [[int(x2), int(y2)]],
                                    [[int(cx), int(cy)]], [[int(x3), int(y3)]], [[int(cx), int(cy)]],
                                    [[int(x4), int(y4)]], [[int(cx), int(cy)]]], np.int32)
                out.append(GreenCrossCandidate(
                    (int(round(cx)), int(round(cy))), radius, l1 + l2, confidence, 4, 1.0,
                    max(0.0, regularity), 0.0, 0.0, white_frac, 1.0, ang1, contour, "lines"))
        out.sort(key=lambda k: -(k.confidence + 0.002 * k.area_px))
        return out[:5]

    # ------------------------------------------------------------- scoring
    def _score(self, blob: np.ndarray, area: float, hsv: np.ndarray) -> GreenCrossCandidate | None:
        c = self.config
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        m = cv2.moments(blob, binaryImage=True)
        if m["m00"] <= 0:
            return None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        (_, _), radius = cv2.minEnclosingCircle(contour)
        radius = max(2.0, float(radius))

        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
        solidity = float(cv2.contourArea(contour)) / hull_area
        # peak score inside the X band, falling to 0 at the gate edges
        solidity_score = _tent(solidity, c.min_solidity, 0.58, c.max_solidity)

        arm_counts, arm_angles = [], []
        for frac in c.ring_radius_fracs:
            count, angles = _ring_runs(blob, (cx, cy), radius * frac, c.ring_samples)
            arm_counts.append(count)
            if count == c.expected_arms:
                arm_angles = angles
        arm_count = int(np.median(arm_counts))
        if arm_count == c.expected_arms:
            arm_score = 1.0
        elif abs(arm_count - c.expected_arms) == 1:
            arm_score = 0.5
        else:
            arm_score = 0.0

        arm_regularity, orientation = 0.0, 0.0
        if len(arm_angles) == c.expected_arms:
            a = np.sort(np.array(arm_angles))
            gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
            ideal = 360.0 / c.expected_arms
            arm_regularity = float(max(0.0, 1.0 - np.mean(np.abs(gaps - ideal)) / (ideal / 2)))
            # circular mean of angles folded to [0, 90)
            folded = np.radians((a % ideal) * (360.0 / ideal))
            orientation = float(np.degrees(math.atan2(np.sin(folded).mean(),
                                                      np.cos(folded).mean())) * ideal / 360.0) % ideal

        white_backing = _white_between_arms(hsv, blob, (cx, cy), radius * c.backing_inner_frac,
                                            radius * c.backing_outer_frac, self._white_v_min(hsv),
                                            c.backing_s_max)
        icx, icy = int(round(cx)), int(round(cy))
        r_in = max(1, int(radius * 0.08))
        patch = blob[max(0, icy - r_in):icy + r_in + 1, max(0, icx - r_in):icx + r_in + 1]
        center_fill = float(np.count_nonzero(patch)) / max(1, patch.size)

        confidence = (0.40 * arm_score * (0.5 + 0.5 * arm_regularity)
                      + 0.20 * solidity_score
                      + 0.25 * white_backing
                      + 0.15 * center_fill)
        return GreenCrossCandidate((icx, icy), radius, area, float(np.clip(confidence, 0, 1)),
                                   arm_count, arm_score, arm_regularity, solidity, solidity_score,
                                   white_backing, center_fill, orientation, contour)

    def _passes(self, cand: GreenCrossCandidate) -> bool:
        c = self.config
        solid_ok = cand.source == "lines" or c.min_solidity <= cand.solidity <= c.max_solidity
        return (cand.confidence >= c.min_detection_confidence
                and cand.arm_score >= c.min_arm_score
                and cand.arm_regularity >= c.min_arm_regularity
                and solid_ok
                and cand.white_backing >= c.min_white_backing)

    def _reject_reason(self, cand: GreenCrossCandidate) -> str:
        c = self.config
        if cand.arm_score < c.min_arm_score:
            return f"best blob has {cand.arm_count} arms (need {c.expected_arms})"
        if cand.arm_regularity < c.min_arm_regularity:
            return f"best blob arms unevenly spaced (regularity {cand.arm_regularity:.2f})"
        if not (c.min_solidity <= cand.solidity <= c.max_solidity):
            return f"best blob solidity {cand.solidity:.2f} outside X range"
        if cand.white_backing < c.min_white_backing:
            return "best blob not on a white background"
        if cand.confidence < c.min_detection_confidence:
            return "best blob below confidence threshold"
        return "best blob below detection thresholds"

    @staticmethod
    def _result(cand: GreenCrossCandidate, blob_count: int, reason: str, mask: np.ndarray,
                *, accepted: bool) -> GreenCrossResult:
        return GreenCrossResult(cand.center if accepted else None, cand.confidence, cand.radius_px,
                                cand.area_px, cand.arm_count, cand.arm_score, cand.arm_regularity,
                                cand.solidity, cand.white_backing, cand.center_fill,
                                cand.orientation_deg, blob_count, reason, mask, cand.contour)


# ---------------------------------------------------------------- overlay
def draw_debug_overlay(image: np.ndarray, result: GreenCrossResult) -> np.ndarray:
    out = _ensure_bgr(image).copy()
    if result.contour is not None:
        cv2.drawContours(out, [result.contour], -1,
                         (0, 255, 255) if result.accepted else (0, 165, 255), 2)
    if result.accepted and result.center is not None:
        cv2.drawMarker(out, result.center, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.circle(out, result.center, int(result.radius_px), (255, 0, 255), 1)
    status = "GREEN X" if result.accepted else f"NO TARGET: {result.reject_reason}"
    lines = [status,
             f"conf={result.confidence:.2f} arms={result.arm_count} reg={result.arm_regularity:.2f}",
             f"solid={result.solidity:.2f} white={result.white_backing:.2f} "
             f"fill={result.center_fill:.2f} rot={result.orientation_deg:.0f}"]
    for i, line in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- helpers
def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _odd(v: int) -> int:
    v = max(1, int(v))
    return v if v % 2 else v + 1


def _tent(x: float, lo: float, peak: float, hi: float) -> float:
    if x <= lo or x >= hi:
        return 0.0
    return (x - lo) / (peak - lo) if x <= peak else (hi - x) / (hi - peak)


def _ring_runs(mask: np.ndarray, center: tuple[float, float], radius: float,
               samples: int) -> tuple[int, list[float]]:
    """Count green runs on a circle; return (count, centre angle of each run in degrees)."""
    h, w = mask.shape
    # at least one sample per pixel of circumference, so thin arms (1-2 px) are never skipped
    samples = max(int(samples), int(math.ceil(2 * math.pi * radius)))
    ang = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = np.round(center[0] + radius * np.cos(ang)).astype(int)
    ys = np.round(center[1] + radius * np.sin(ang)).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vals = np.zeros(samples, dtype=bool)
    vals[inside] = mask[ys[inside], xs[inside]] > 0
    if vals.all() or not vals.any():
        return (0 if not vals.any() else 1), []
    # rotate so the sequence starts on an "off" sample, then find runs
    start = int(np.argmin(vals))
    v = np.roll(vals, -start)
    runs, angles, i = 0, [], 0
    min_len = 2 if samples >= 360 else 1  # ignore single-sample noise
    while i < samples:
        if v[i]:
            j = i
            while j < samples and v[j]:
                j += 1
            if j - i >= min_len:
                runs += 1
                mid = (start + (i + j - 1) / 2.0) % samples
                angles.append(float(np.degrees(ang[int(mid)])))
            i = j
        else:
            i += 1
    return runs, angles


def _white_between_arms(hsv: np.ndarray, blob: np.ndarray, center: tuple[float, float],
                        r_in: float, r_out: float, v_min: int, s_max: int) -> float:
    """Fraction of the non-green pixels between the arms that look like a white sheet."""
    h, w = hsv.shape[:2]
    ring = np.zeros((h, w), np.uint8)
    c = (int(round(center[0])), int(round(center[1])))
    cv2.circle(ring, c, max(1, int(round(r_out))), 255, -1)
    cv2.circle(ring, c, max(0, int(round(r_in))), 0, -1)
    grown = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    sel = (ring > 0) & (grown == 0)
    n = int(np.count_nonzero(sel))
    if n == 0:
        return 0.0
    s, v = hsv[..., 1][sel], hsv[..., 2][sel]
    return float(np.count_nonzero((v >= v_min) & (s <= s_max))) / n
