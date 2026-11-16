"""Synthetic image tests for the green-X detector (run: python3 -m pytest test/)."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from drone_target_cv.green_cross_detector import (  # noqa: E402
    GreenCrossDetector,
    GreenCrossDetectorConfig,
)

ASPHALT = (82, 82, 82)
WHITE = (240, 240, 240)
GREEN = (40, 150, 30)
RED = (28, 30, 210)


def _frame(color=ASPHALT) -> np.ndarray:
    return np.full((480, 640, 3), color, dtype=np.uint8)


def _grass(seed: int = 0) -> np.ndarray:
    g = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, (480, 640)), (0, 0), 6)
    return np.dstack([40 + 20 * g, 110 + 40 * g, 50 + 20 * g]).clip(0, 255).astype(np.uint8)


def _green_x(frame, center, half, rot=0.0, arm_frac=0.2):
    """White square sheet with a corner-to-corner green X, rotated by rot degrees."""
    m = cv2.getRotationMatrix2D(center, rot, 1.0)
    sq = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], np.float32) + center
    sq = cv2.transform(sq[None], m)[0].astype(np.int32)
    cv2.fillConvexPoly(frame, sq, WHITE)
    sheet = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(sheet, sq, 255)
    arms = np.zeros_like(sheet)
    w = max(2, int(2 * half * arm_frac))
    cv2.line(arms, tuple(sq[0]), tuple(sq[2]), 255, w)
    cv2.line(arms, tuple(sq[1]), tuple(sq[3]), 255, w)
    frame[(arms > 0) & (sheet > 0)] = GREEN


def test_detects_green_x_on_asphalt():
    f = _frame()
    _green_x(f, (320, 240), 60)
    r = GreenCrossDetector().detect(f)
    assert r.accepted
    assert abs(r.center[0] - 320) <= 3 and abs(r.center[1] - 240) <= 3
    assert r.arm_count == 4


def test_detects_rotated_x_off_centre():
    f = _frame()
    _green_x(f, (180, 330), 45, rot=25)
    r = GreenCrossDetector().detect(f)
    assert r.accepted
    assert abs(r.center[0] - 180) <= 3 and abs(r.center[1] - 330) <= 3


def test_detects_small_far_away_x():
    f = _frame()
    _green_x(f, (400, 200), 14)
    assert GreenCrossDetector().detect(f).accepted


def test_detects_x_on_grass_when_arms_touch_grass():
    f = _grass()
    _green_x(f, (400, 200), 50, rot=10)
    r = GreenCrossDetector().detect(f)
    assert r.accepted
    assert abs(r.center[0] - 400) <= 4 and abs(r.center[1] - 200) <= 4


def test_rejects_solid_green_square():
    f = _frame()
    cv2.rectangle(f, (250, 180), (370, 300), GREEN, -1)
    assert not GreenCrossDetector().detect(f).accepted


def test_rejects_plain_grass():
    assert not GreenCrossDetector().detect(_grass(3)).accepted


def test_rejects_red_bullseye():
    f = _frame()
    for radius, color in [(100, RED), (80, WHITE), (60, RED), (40, WHITE), (20, RED)]:
        cv2.circle(f, (320, 240), radius, color, -1)
    assert not GreenCrossDetector().detect(f).accepted


def test_rejects_white_plus_without_green():
    f = _frame()
    cv2.rectangle(f, (300, 180), (340, 300), WHITE, -1)
    cv2.rectangle(f, (260, 220), (380, 260), WHITE, -1)
    assert not GreenCrossDetector().detect(f).accepted


def test_normalized_center_matches_team_convention():
    f = _frame()
    _green_x(f, (480, 120), 40)
    r = GreenCrossDetector().detect(f)
    x, y = r.normalized_center(640, 480)
    assert abs(x - 0.5) < 0.02 and abs(y + 0.5) < 0.02


def test_thin_line_x_like_rules_drawing_with_line_search():
    """C-UASC rules drawing: thin pale-green X inside a white rectangle."""
    f = _frame((40, 45, 42))
    cv2.rectangle(f, (220, 140), (420, 340), WHITE, -1)
    cv2.line(f, (225, 145), (415, 335), (170, 215, 160), 3)
    cv2.line(f, (415, 145), (225, 335), (170, 215, 160), 3)
    cfg = GreenCrossDetectorConfig(use_line_search=True, line_hsv_s_min=30,
                                   line_green_dominance_ratio=1.08, line_green_difference_min=15)
    r = GreenCrossDetector(cfg).detect(f)
    assert r.accepted
    assert abs(r.center[0] - 320) <= 4 and abs(r.center[1] - 240) <= 4


def test_webcam_settings_reject_skin_coloured_blob():
    """A face-like skin blob under a green colour cast must not pass as the X."""
    f = _frame((60, 70, 62))
    cv2.ellipse(f, (320, 240), (90, 120), 0, 0, 360, (120, 150, 165), -1)   # skin, BGR
    cv2.ellipse(f, (290, 210), (12, 6), 0, 0, 360, (40, 40, 40), -1)
    cv2.ellipse(f, (350, 210), (12, 6), 0, 0, 360, (40, 40, 40), -1)
    cast = np.clip(f.astype(np.float32) * [0.85, 1.15, 0.85], 0, 255).astype(np.uint8)
    cfg = GreenCrossDetectorConfig(use_line_search=True, line_hsv_s_min=30,
                                   line_green_dominance_ratio=1.08, line_green_difference_min=15)
    assert not GreenCrossDetector(cfg).detect(cast).accepted
