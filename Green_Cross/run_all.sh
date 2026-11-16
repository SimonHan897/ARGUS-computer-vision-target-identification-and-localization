#!/usr/bin/env bash
# Runs the whole PD5 + PD7 sequence. Usage:  bash ~/ARGUS_2027/Green_Cross/run_all.sh
# Everything printed is also saved to run_all.log
cd "$(dirname "$0")"
exec > >(tee -a run_all.log) 2>&1
export PYTHONUNBUFFERED=1   # show Python output immediately (not in big chunks)
step() { echo; echo "=================================================================="; echo ">>> $*"; echo "=================================================================="; }
fail() { echo; echo "!!! STOPPED: $*"; echo "!!! Send Claude the file run_all.log (or a screenshot of this terminal)."; exit 1; }

step "0. Python environment"
[[ -x .venv/bin/python ]] || bash setup_env.sh || fail "setup_env.sh failed"
source .venv/bin/activate
python3 -m pytest -q tests ../src/drone_target_cv/test/test_green_cross_detector.py || fail "unit tests failed"

step "1. PD5 labeled-dataset metric"
if [[ -f eval_v1/metrics.json && -f eval_v2/metrics.json ]]; then
    echo "already done (eval_v1/, eval_v2/ exist) - skipping"
else
    python3 eval_dataset.py --detector v1 --out eval_v1 || fail "eval v1"
    python3 eval_dataset.py --detector v2 --out eval_v2 || fail "eval v2"
fi
python3 compare_versions.py eval_v1 eval_v2 pd5_v1_vs_v2.png || fail "compare_versions"

step "2. Simulator"
if docker ps --format '{{.Names}}' | grep -q '^ardupilot-gz-sim$'; then
    echo "simulator already running"
else
    echo "starting the simulator in a new terminal window (leave it open)..."
    gnome-terminal --title="SIMULATOR - leave open" -- bash -c "cd ~/Drone_Sim_Docker && make run; exec bash" \
        || fail "could not open a terminal for the simulator - run 'cd ~/Drone_Sim_Docker && make run' yourself, then run this script again"
fi
echo "waiting for the drone to be ready to fly (up to 5 min)..."
python3 - <<'PY' || fail "simulator never became ready"
import time
from vehicle import Vehicle
for attempt in range(30):
    v = Vehicle("udpin:127.0.0.1:14551", print)
    try:
        v.connect(timeout=10)
        v.wait_ready(timeout=240)
        v.close()
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print("  not ready yet:", e)
        v.close()
        time.sleep(5)
raise SystemExit(1)
PY

step "3. Camera: point it straight down and start the video stream"
python3 sim_targets.py autoaim || fail "camera setup"

run() {  # run <id> <alt> <north> <east> <rot> [--decoys] [extra run_test args]
    local id=$1 alt=$2 n=$3 e=$4 rot=$5; shift 5
    local decoys="" ; [[ "$1" == "--decoys" ]] && { decoys="--decoys"; shift; }
    step "Run $id: altitude $alt m, target N $n E $e rot $rot ${decoys}"
    if [[ -f runs/$id/estimate.json ]]; then echo "already done - skipping"; return 0; fi
    rm -rf "runs/$id"
    python3 sim_targets.py spawn --north "$n" --east "$e" --rot "$rot" $decoys || fail "spawn for $id"
    python3 run_test.py --run-id "$id" --alt "$alt" --show "$@" || fail "run $id (see runs/$id/events.log)"
    echo "--- $id result ---"; grep -E "RESULT|FIRST DETECTION" "runs/$id/events.log" || true
    sleep 8
}

run R0 8   18 -6 20 --mount rot0
step "Calibration from R0"
python3 analyze_runs.py --calibrate R0 || fail "calibration"
run R1 6.5 22  5  0 --decoys
run R2 8   28 -8 35 --decoys
run R3 10  15  9 60 --decoys

step "4. Analysis"
python3 analyze_runs.py R1 R2 R3 || fail "analysis"
echo
echo "ALL DONE. Results: runs/analysis/RESULTS.md and figures in runs/analysis/"
echo "PD5 charts: pd5_v1_vs_v2.png, eval_v2/pd5_metrics.png"
