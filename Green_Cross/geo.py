"""Camera geometry: pixel <-> ground point <-> lat/lon for a downward camera.

Frames
------
  NED     local level frame at home: x = North, y = East, z = Down (metres)
  body    FRD: x = Forward (nose), y = Right, z = Down
  camera  OpenCV optical frame: x = image right, y = image down, z = out of the lens

A pixel (u, v) becomes a ray  d_cam = [(u-cx)/fx, (v-cy)/fy, 1].
It is rotated into NED with   d_ned = R_ned_body(roll, pitch, yaw) @ R_body_cam @ d_cam
and intersected with the flat ground plane (z = 0) below the drone at height h:
    t = h / d_ned.z ,   ground = drone_NE + t * d_ned.xy

The mount (R_body_cam) is not assumed: ``MOUNTS`` lists the 8 axis-aligned ways a
nadir camera can sit (image rotated 0/90/180/270 deg, optionally mirrored). The
calibration run picks the one that best explains the data (boresight calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 480
    hfov_rad: float = 2.0          # ardupilot_gazebo gimbal_small_3d camera

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(self.hfov_rad / 2.0)

    @property
    def fy(self) -> float:
        return self.fx              # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def ground_footprint_m(self, alt_m: float) -> tuple[float, float]:
        """Width, height of the ground area seen straight down from alt_m."""
        return (2 * alt_m * (self.cx / self.fx), 2 * alt_m * (self.cy / self.fy))


# ------------------------------------------------------------------ rotations
def rot_ned_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body(FRD) -> NED rotation from ZYX Euler angles (radians), as MAVLink ATTITUDE."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _mount(k90: int, mirror: bool) -> np.ndarray:
    """Camera->body rotation for a nadir camera.

    k90=0, mirror=False: image top points to the nose, image right to the right wing.
    k90 rotates the image about the lens axis in 90 deg steps; mirror flips left/right.
    """
    base = np.array([[0.0, -1.0, 0.0],    # cam x (right) -> body +y, cam y (down) -> body -x
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])
    a = k90 * math.pi / 2
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    m = rz @ base
    if mirror:
        m = m @ np.diag([-1.0, 1.0, 1.0])
    return m


MOUNTS: dict[str, np.ndarray] = {
    f"rot{k * 90}{'_mirror' if mir else ''}": _mount(k, mir)
    for mir in (False, True) for k in range(4)
}
DEFAULT_MOUNT = "rot0"


# --------------------------------------------------------------- projection
def pixel_to_ground(u: float, v: float, cam: Camera, drone_ne: tuple[float, float],
                    alt_agl: float, roll: float, pitch: float, yaw: float,
                    mount: str = DEFAULT_MOUNT) -> tuple[float, float] | None:
    """Ground (north, east) metres hit by pixel (u, v); None if the ray misses the ground."""
    d_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    d = rot_ned_body(roll, pitch, yaw) @ MOUNTS[mount] @ d_cam
    if d[2] <= 1e-6 or alt_agl <= 0:
        return None
    t = alt_agl / d[2]
    return drone_ne[0] + t * d[0], drone_ne[1] + t * d[1]


def ground_to_pixel(north: np.ndarray, east: np.ndarray, cam: Camera,
                    drone_ne: tuple[float, float], alt_agl: float, roll: float, pitch: float,
                    yaw: float, mount: str = DEFAULT_MOUNT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ground points (arrays) into the image. Returns u, v, in_front mask."""
    r_cam_ned = (rot_ned_body(roll, pitch, yaw) @ MOUNTS[mount]).T
    p = np.stack([np.asarray(north) - drone_ne[0], np.asarray(east) - drone_ne[1],
                  np.full(np.shape(north), alt_agl)], axis=0).reshape(3, -1)
    c = r_cam_ned @ p
    z = c[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.fx * c[0] / z + cam.cx
        v = cam.fy * c[1] / z + cam.cy
    shape = np.shape(north)
    return u.reshape(shape), v.reshape(shape), (z > 1e-6).reshape(shape)


# ------------------------------------------------------------------ lat / lon
def ne_to_latlon(home_lat: float, home_lon: float, north: float, east: float) -> tuple[float, float]:
    lat = home_lat + math.degrees(north / EARTH_RADIUS_M)
    lon = home_lon + math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(home_lat))))
    return lat, lon


def latlon_to_ne(home_lat: float, home_lon: float, lat: float, lon: float) -> tuple[float, float]:
    north = math.radians(lat - home_lat) * EARTH_RADIUS_M
    east = math.radians(lon - home_lon) * EARTH_RADIUS_M * math.cos(math.radians(home_lat))
    return north, east
