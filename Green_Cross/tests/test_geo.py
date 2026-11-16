"""Geometry tests: pixel <-> ground round trip and sign conventions (python3 -m pytest tests/)."""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from geo import MOUNTS, Camera, ground_to_pixel, latlon_to_ne, ne_to_latlon, pixel_to_ground  # noqa: E402


def test_round_trip_all_mounts():
    cam, rng = Camera(), np.random.default_rng(1)
    for m in MOUNTS:
        for _ in range(100):
            n, e = rng.uniform(-10, 10, 2)
            alt = rng.uniform(5, 30)
            r, p = rng.uniform(-0.2, 0.2, 2)
            y = rng.uniform(-3, 3)
            tn, te = n + rng.uniform(-4, 4), e + rng.uniform(-4, 4)
            u, v, ok = ground_to_pixel(np.array([tn]), np.array([te]), cam, (n, e), alt, r, p, y, m)
            if ok[0]:
                g = pixel_to_ground(u[0], v[0], cam, (n, e), alt, r, p, y, m)
                assert math.hypot(g[0] - tn, g[1] - te) < 1e-6


def test_straight_down_centre_pixel_is_below_drone():
    g = pixel_to_ground(320, 240, Camera(), (3.0, -2.0), 10.0, 0, 0, 0.7)
    assert abs(g[0] - 3.0) < 1e-9 and abs(g[1] + 2.0) < 1e-9


def test_north_is_up_in_image_with_nose_north():
    u, v, _ = ground_to_pixel(np.array([5.0]), np.array([0.0]), Camera(), (0, 0), 10, 0, 0, 0)
    assert v[0] < 240 and abs(u[0] - 320) < 1e-6


def test_latlon_round_trip():
    lat, lon = ne_to_latlon(-35.3632621, 149.1652374, 20.0, 15.0)
    n, e = latlon_to_ne(-35.3632621, 149.1652374, lat, lon)
    assert abs(n - 20) < 1e-6 and abs(e - 15) < 1e-6
