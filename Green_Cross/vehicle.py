"""pymavlink vehicle wrapper for vision missions (grown out of PD3).

Differences from the PD3 version:
  * telemetry is read in a background thread, so the vision loop never blocks it
  * keeps a short pose history (position + attitude) so each camera frame can be
    paired with the pose at the moment it was captured (pose_at)
  * non-blocking goto (send target, then poll reached()) so frames keep being
    processed while the drone flies
"""

from __future__ import annotations

from collections import deque
import bisect
import math
import threading
import time

from pymavlink import mavutil

PREARM_BIT = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK
POSITION_ONLY_MASK = 0b0000_1101_1111_1000  # use x,y,z only (ignore vel, acc, yaw, yaw rate)
MODES = {name: num for num, name in mavutil.mode_mapping_acm.items()}


class MissionAbort(Exception):
    pass


class Vehicle:
    def __init__(self, connect: str, log=print):
        self.log = log
        log(f"Connecting to {connect} ...")
        self.m = mavutil.mavlink_connection(connect, source_system=250)
        self.lock = threading.Lock()
        self.s = {"mode": "?", "armed": False, "lat": None, "lon": None, "rel_alt": 0.0,
                  "vn": 0.0, "ve": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                  "fix": 0, "sats": 0, "prearm_ok": False, "t_pos": 0.0, "t_att": 0.0}
        self.hist: deque = deque(maxlen=600)   # (t, lat, lon, alt, roll, pitch, yaw)
        self.acks: dict[int, int] = {}
        self.statustext: deque = deque(maxlen=50)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.home: tuple[float, float] | None = None

    # ------------------------------------------------------------ connection
    def connect(self, timeout: float = 30) -> None:
        end = time.time() + timeout
        while time.time() < end:
            hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if hb is None or hb.type == mavutil.mavlink.MAV_TYPE_GCS:
                continue
            if hb.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                continue
            self.m.target_system, self.m.target_component = hb.get_srcSystem(), hb.get_srcComponent()
            self.log(f"Heartbeat from system {self.m.target_system} component {self.m.target_component}")
            break
        else:
            raise MissionAbort("No heartbeat. Is the simulator (make run) running?")
        for msg_id, hz in {mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 20,
                           mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 30,
                           mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS: 2,
                           mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: 2}.items():
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                         mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                         msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self.m.close()          # release UDP 14551 for the next program
        except Exception:  # noqa: BLE001
            pass

    def _reader(self) -> None:
        last_hb = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now - last_hb > 1.0:
                self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                          mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                last_hb = now
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            while msg is not None:
                self._handle(msg, time.time())
                msg = self.m.recv_match(blocking=False)

    def _handle(self, msg, t: float) -> None:
        typ = msg.get_type()
        s = self.s
        with self.lock:
            if typ == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system \
                    and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                s["mode"] = mavutil.mode_string_v10(msg)
                s["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            elif typ == "GLOBAL_POSITION_INT":
                s["lat"], s["lon"] = msg.lat / 1e7, msg.lon / 1e7
                s["rel_alt"] = msg.relative_alt / 1000.0
                s["vn"], s["ve"] = msg.vx / 100.0, msg.vy / 100.0
                s["t_pos"] = t
                self._push(t)
            elif typ == "ATTITUDE":
                s["roll"], s["pitch"], s["yaw"] = msg.roll, msg.pitch, msg.yaw
                s["t_att"] = t
                self._push(t)
            elif typ == "GPS_RAW_INT":
                s["fix"], s["sats"] = msg.fix_type, msg.satellites_visible
            elif typ == "SYS_STATUS":
                s["prearm_ok"] = bool(msg.onboard_control_sensors_health & PREARM_BIT)
            elif typ == "COMMAND_ACK":
                self.acks[msg.command] = msg.result
            elif typ == "STATUSTEXT":
                self.statustext.append(msg.text)
                self.log(f"  AUTOPILOT: {msg.text}")

    def _push(self, t: float) -> None:
        s = self.s
        if s["lat"] is not None:
            self.hist.append((t, s["lat"], s["lon"], s["rel_alt"], s["roll"], s["pitch"], s["yaw"]))

    # ---------------------------------------------------------------- state
    def state(self) -> dict:
        with self.lock:
            return dict(self.s)

    def pose_at(self, t: float) -> tuple | None:
        """Linearly interpolated (lat, lon, alt, roll, pitch, yaw) at host time t."""
        with self.lock:
            h = list(self.hist)
        if not h:
            return None
        times = [x[0] for x in h]
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return h[0][1:]
        if i >= len(h):
            return h[-1][1:]
        a, b = h[i - 1], h[i]
        f = (t - a[0]) / max(1e-6, b[0] - a[0])
        out = [a[k] + f * (b[k] - a[k]) for k in range(1, 6)]
        dyaw = (b[6] - a[6] + math.pi) % (2 * math.pi) - math.pi
        out.append(a[6] + f * dyaw)
        return tuple(out)

    def wait_until(self, cond, timeout: float, what: str, tick=None) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if cond(self.state()):
                return
            if tick:
                tick()
            else:
                time.sleep(0.05)
        raise MissionAbort(f"Timed out after {timeout:.0f}s waiting for: {what}")

    # -------------------------------------------------------------- commands
    def command(self, cmd: int, *params: float, timeout: float = 5) -> bool:
        p = list(params) + [0.0] * (7 - len(params))
        self.acks.pop(cmd, None)
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component, cmd, 0, *p)
        end = time.time() + timeout
        while time.time() < end:
            if cmd in self.acks:
                ok = self.acks[cmd] == mavutil.mavlink.MAV_RESULT_ACCEPTED
                name = mavutil.mavlink.enums["MAV_CMD"][cmd].name
                res = mavutil.mavlink.enums["MAV_RESULT"][self.acks[cmd]].name
                self.log(f"  COMMAND_ACK {name}: {res}")
                return ok
            time.sleep(0.02)
        self.log("  (no COMMAND_ACK)")
        return False

    def set_mode(self, mode: str, timeout: float = 10) -> None:
        self.log(f"Setting mode {mode}")
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                     mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, MODES[mode])
        self.wait_until(lambda s: s["mode"] == mode, timeout, f"mode {mode}")

    def wait_ready(self, timeout: float = 90) -> None:
        self.log("Waiting for GPS 3D fix and pre-arm checks ...")
        self.wait_until(lambda s: s["fix"] >= 3 and s["lat"] is not None and s["prearm_ok"],
                        timeout, "GPS fix + pre-arm OK")
        s = self.state()
        self.home = (s["lat"], s["lon"])
        self.log(f"Ready. Home = {s['lat']:.7f}, {s['lon']:.7f}")

    def arm(self, timeout: float = 30) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1):
                self.wait_until(lambda s: s["armed"], 5, "armed")
                self.log("Motors ARMED")
                return
            time.sleep(2)
        raise MissionAbort("Vehicle refused to arm")

    def takeoff(self, alt: float, timeout: float = 60, tick=None) -> None:
        self.log(f"TAKEOFF to {alt} m")
        if not self.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt):
            raise MissionAbort("Takeoff rejected")
        self.wait_until(lambda s: s["rel_alt"] >= alt * 0.95, timeout, f"altitude {alt} m", tick)

    def set_speed(self, mps: float) -> None:
        self.command(mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 1, mps, -1)

    def goto(self, lat: float, lon: float, alt: float) -> None:
        self.m.mav.set_position_target_global_int_send(
            0, self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, POSITION_ONLY_MASK,
            int(lat * 1e7), int(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0)

    def distance_to(self, lat: float, lon: float) -> float:
        s = self.state()
        if s["lat"] is None:
            return float("inf")
        dn = math.radians(lat - s["lat"]) * 6378137.0
        de = math.radians(lon - s["lon"]) * 6378137.0 * math.cos(math.radians(s["lat"]))
        return math.hypot(dn, de)

    def land(self, timeout: float = 120, tick=None) -> None:
        self.set_mode("LAND")
        self.wait_until(lambda s: not s["armed"], timeout, "touchdown + disarm", tick)
        self.log("Landed and DISARMED")
