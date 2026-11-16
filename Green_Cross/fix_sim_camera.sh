#!/usr/bin/env bash
# One-time fix: the simulator image is missing the GStreamer packages the Gazebo
# drone camera needs to stream video (Gazebo log: "GstCameraPlugin: failed to create
# GStreamer elements"). This installs them INSIDE the running simulator container and
# saves the result into your local image, so every later 'make run' keeps the fix.
# It does not change any files in ~/Drone_Sim_Docker or ~/ARGUS_2027.
set -euo pipefail
C=ardupilot-gz-sim
IMG=ardupilot-gz:4.6.3

if ! docker ps --format '{{.Names}}' | grep -q "^$C\$"; then
    echo "The simulator is not running. Start it first:  cd ~/Drone_Sim_Docker && make run"
    exit 1
fi
echo ">>> Installing the missing video packages inside the simulator (1-2 min)..."
docker exec -u root "$C" bash -c "apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y -qq --no-install-recommends gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly"
echo ">>> Checking the encoder is now available..."
docker exec "$C" bash -c "gst-inspect-1.0 x264enc >/dev/null && gst-inspect-1.0 rtph264pay >/dev/null && gst-inspect-1.0 udpsink >/dev/null" \
    && echo "    x264enc, rtph264pay, udpsink: OK"
echo ">>> Saving this into the local simulator image ($IMG)..."
docker commit "$C" "$IMG" >/dev/null
echo
echo "DONE. Now restart the simulator so Gazebo loads the new packages:"
echo "  1) press Ctrl+C in the SIMULATOR window and wait for it to stop"
echo "  2) run:  bash ~/ARGUS_2027/Green_Cross/run_all.sh"
