"""Uses the team copy of the detector in ARGUS_2027/src/drone_target_cv (single source)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "drone_target_cv"))

from drone_target_cv.green_cross_detector import (  # noqa: E402,F401
    GreenCrossCandidate,
    GreenCrossDetector,
    GreenCrossDetectorConfig,
    GreenCrossResult,
    draw_debug_overlay,
)

__all__ = ["GreenCrossCandidate", "GreenCrossDetector", "GreenCrossDetectorConfig",
           "GreenCrossResult", "draw_debug_overlay"]
