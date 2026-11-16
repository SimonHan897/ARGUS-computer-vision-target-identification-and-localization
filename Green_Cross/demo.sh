#!/usr/bin/env bash
# Repeatable drone flight demo (for screen recordings). Your official R1-R3 results are never touched.
#
#   bash ~/ARGUS_2027/Green_Cross/demo.sh                    # 8 m, target N 28 E -8, with decoys
#   bash ~/ARGUS_2027/Green_Cross/demo.sh 10 15 9 60          # altitude, north, east, rotation
#
# Each flight gets its own folder: runs/DEMO_<date>_<time>/ (video, logs, map, results).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
ALT=${1:-8}; NORTH=${2:-28}; EAST=${3:--8}; ROT=${4:-35}
ID="DEMO_$(date +%m%d_%H%M%S)"
step() { echo; echo "=================================================================="; echo ">>> $*"; echo "=================================================================="; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! Send Claude a screenshot of this terminal."; exit 1; }

source .venv/bin/activate || fail "no Python environment - run: bash setup_env.sh"

step "1. Simulator"
if docker ps --format '{{.Names}}' | grep -q '^ardupilot-gz-sim$'; then
    echo "simulator running"
else
    echo "starting the simulator in a new window (leave it open)..."
    gnome-terminal --title="SIMULATOR - leave open" -- bash -c "cd ~/Drone_Sim_Docker && make run; exec bash" \
        || fail "start it yourself: cd ~/Drone_Sim_Docker && make run   (then run this again)"
fi
echo "waiting until the drone is on the ground and ready to fly (up to 5 min)..."
python3 - <<'PY' || fail "drone never became ready (is it still flying/landing? wait and run again)"
import time
from vehicle import Vehicle
for _ in range(30):
    v = Vehicle("udpin:127.0.0.1:14551", print)
    try:
        v.connect(timeout=10); v.wait_ready(timeout=240); v.close(); raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print("  not ready yet:", e); v.close(); time.sleep(5)
raise SystemExit(1)
PY

step "2. Camera straight down + target at N $NORTH E $EAST (rot $ROT) with decoys"
python3 sim_targets.py autoaim || fail "camera setup"
python3 sim_targets.py spawn --north "$NORTH" --east "$EAST" --rot "$ROT" --decoys || fail "placing the target"

step "3. Ready to fly $ID at $ALT m"
echo "Arrange your windows (Gazebo: right-click the drone > Follow)."
echo "Start your screen recording now (PrtSc > video icon > Screen > record)."
read -rp "Press Enter to take off... "
python3 run_test.py --run-id "$ID" --alt "$ALT" --show || fail "flight (see runs/$ID/events.log)"

step "4. Results for $ID"
python3 analyze_runs.py "$ID" --out "runs/$ID/analysis" || fail "analysis"
echo
echo "Done. You can stop the screen recording."
echo "Everything for this flight:  ~/ARGUS_2027/Green_Cross/runs/$ID/"
echo "   annotated.mp4 (drone camera video), analysis/${ID}_map.png, analysis/RESULTS.md, events.log"
echo "Run again any time:  bash ~/ARGUS_2027/Green_Cross/demo.sh"
