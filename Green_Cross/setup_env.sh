#!/usr/bin/env bash
# One-time setup for PD5/PD7 on the Ubuntu host (the simulator itself runs in Docker).
#  - system OpenCV (has GStreamer, needed to read the Gazebo camera stream on UDP 5600)
#  - GStreamer H.264 decoder plugins
#  - a .venv that can see the system OpenCV, plus pymavlink
set -euo pipefail
cd "$(dirname "$0")"

sudo apt install -y python3-venv python3-opencv python3-numpy python3-matplotlib python3-yaml \
     gstreamer1.0-libav gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-tools

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv --system-site-packages .venv
fi
.venv/bin/pip install -q pymavlink pytest

.venv/bin/python - <<'PY'
import cv2
from frame_source import opencv_has_gstreamer
print("OpenCV", cv2.__version__, "| GStreamer support:", "YES" if opencv_has_gstreamer() else "NO")
if not opencv_has_gstreamer():
    raise SystemExit("This OpenCV has no GStreamer. If 'pip install opencv-python' was used, "
                     "run 'pip uninstall opencv-python' so the system python3-opencv is used.")
import pymavlink; print("pymavlink OK")
PY
echo "Setup done. Activate with:  source .venv/bin/activate"
