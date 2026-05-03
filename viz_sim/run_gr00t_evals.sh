#!/usr/bin/env bash
# GR00T-N1.7-DROID evaluation sweep.
#
# 5 robosuite tasks (Panda, fixed-base) + 10 robocasa atomic tasks (PandaOmron,
# mobile manipulator with base/torso pinned to 0). EPISODES per task default 10.
#
# Boots a single GR00T server with ATTN=1 (so the attention payload is
# attached to every inference response and recorded in the per-episode npz),
# runs every (robot, task) cell, then kills the server.
#
# Run from the repo root:
#     bash viz_sim/run_gr00t_evals.sh
#
# Override knobs:
#     EPISODES=10        episodes per task
#     GPU=0              CUDA_VISIBLE_DEVICES for the server
#     PORT=5555          ZMQ port
#     SERVER_LOG=...     where the server log goes
#     SKIP_ROBOSUITE=1   only do robocasa tasks
#     SKIP_ROBOCASA=1    only do robosuite tasks
set -uo pipefail

REPO=/home/edward/projects/openpi_vis
LOG_DIR="${REPO}/results/_logs/gr00t"
mkdir -p "${LOG_DIR}"

EPISODES="${EPISODES:-10}"
GPU="${GPU:-0}"
PORT="${PORT:-5555}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/server.log}"
SKIP_ROBOSUITE="${SKIP_ROBOSUITE:-0}"
SKIP_ROBOCASA="${SKIP_ROBOCASA:-0}"

# task → prompt
declare -A ROBOSUITE_PROMPT=(
    [Lift]="pick up the cube"
    [Stack]="stack the red cube on top of the green cube"
    [Door]="open the door"
    [PickPlaceCan]="pick up the can and place it in the bin"
    [NutAssemblySquare]="put the square nut onto the peg"
)
ROBOSUITE_TASKS=(Lift Stack Door PickPlaceCan NutAssemblySquare)

declare -A ROBOCASA_PROMPT=(
    [OpenDoor]="open the cabinet door"
    [CloseDoor]="close the cabinet door"
    [OpenMicrowave]="open the microwave"
    [CloseMicrowave]="close the microwave"
    [OpenDrawer]="open the drawer"
    [CloseDrawer]="close the drawer"
    [TurnOnMicrowave]="turn on the microwave"
    [TurnOnSinkFaucet]="turn on the sink faucet"
    [TurnOffSinkFaucet]="turn off the sink faucet"
    [TurnOnToaster]="turn on the toaster"
)
ROBOCASA_TASKS=(OpenDoor CloseDoor OpenMicrowave CloseMicrowave OpenDrawer
                CloseDrawer TurnOnMicrowave TurnOnSinkFaucet TurnOffSinkFaucet
                TurnOnToaster)

cleanup() {
    echo "[orch] cleaning up server..."
    pkill -f 'serve_gr00t_attn' 2>/dev/null || true
    sleep 2
    pkill -9 -f 'serve_gr00t_attn' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Boot the GR00T server with attention capture ─────────────────────────────
echo "[orch] === boot GR00T server (ATTN=1, GPU=${GPU}, PORT=${PORT}) ==="
pkill -f 'serve_gr00t_attn' 2>/dev/null || true
sleep 2
(
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${GPU}" ATTN=1 PORT="${PORT}" \
        bash viz_sim/run_gr00t_server.sh \
        > "${SERVER_LOG}" 2>&1
) &
SERVER_PID=$!

echo "[orch] server PID=${SERVER_PID}; waiting up to 5 min for readiness..."
for i in $(seq 1 60); do
    if grep -q "Server is ready and listening" "${SERVER_LOG}" 2>/dev/null \
       || grep -q "Server ready" "${SERVER_LOG}" 2>/dev/null; then
        echo "[orch] server up after ${i} × 5s polls"
        break
    fi
    if grep -qE "^Traceback|FileNotFoundError|RuntimeError|CUDA out of memory" \
       "${SERVER_LOG}" 2>/dev/null; then
        echo "[orch] server failed to start, see ${SERVER_LOG}"
        tail -40 "${SERVER_LOG}"
        exit 1
    fi
    sleep 5
done
sleep 3
echo "[orch] server warm; starting eval sweep"

# ── Robosuite sweep (Panda) ─────────────────────────────────────────────────
if [ "${SKIP_ROBOSUITE}" != "1" ]; then
    echo "[orch] === robosuite (Panda, ${EPISODES} eps × 5 tasks) ==="
    for task in "${ROBOSUITE_TASKS[@]}"; do
        prompt="${ROBOSUITE_PROMPT[${task}]}"
        log="${LOG_DIR}/eval_robosuite_${task}.log"
        echo "[orch] gr00t_droid / Panda / ${task} / '${prompt}'"
        conda run -n robocasa_sim --no-capture-output \
            python "${REPO}/viz_sim/eval_runner.py" \
            --model gr00t_droid --robot Panda \
            --task "${task}" --episodes "${EPISODES}" \
            --port "${PORT}" --prompt "${prompt}" \
            > "${log}" 2>&1 || echo "[orch] ${task} eval errored, see ${log}"
    done
fi

# ── Robocasa sweep (PandaOmron) ─────────────────────────────────────────────
if [ "${SKIP_ROBOCASA}" != "1" ]; then
    echo "[orch] === robocasa (PandaOmron, ${EPISODES} eps × 10 tasks) ==="
    for task in "${ROBOCASA_TASKS[@]}"; do
        prompt="${ROBOCASA_PROMPT[${task}]}"
        log="${LOG_DIR}/eval_robocasa_${task}.log"
        echo "[orch] gr00t_droid / PandaOmron / ${task} / '${prompt}'"
        conda run -n robocasa_sim --no-capture-output \
            python "${REPO}/viz_sim/eval_runner.py" \
            --model gr00t_droid --robot PandaOmron \
            --task "${task}" --episodes "${EPISODES}" \
            --port "${PORT}" --prompt "${prompt}" \
            > "${log}" 2>&1 || echo "[orch] ${task} eval errored, see ${log}"
    done
fi

echo "[orch] all sweeps done. Per-episode npz under /mnt/sda/edward/projects/robocasa_365/gr00t_droid/"
echo "[orch] Logs under ${LOG_DIR}/"
