#!/bin/bash
# Sweep eval over (model × task) on real robocasa kitchen envs (no perturbation).
# Brings the policy server up/down per model so each ckpt's serve-side transforms
# (RobocasaInputs/Outputs vs DroidInputs vs LiberoInputs) are loaded correctly.
#
# Usage:
#     bash viz_sim/robocasa365_orch.sh                  # 3 models × default tasks × 20 eps
#     EPISODES=5 TASKS="PickPlaceCounterToStove" bash viz_sim/robocasa365_orch.sh
#     MODELS="pi05_robocasa365" bash viz_sim/robocasa365_orch.sh
set -uo pipefail

REPO=/home/edward/projects/openpi_vis
LOG=${REPO}/results/_logs
PY=/home/edward/miniconda3/envs/robocasa_sim/bin/python

EPISODES=${EPISODES:-20}
SEED_BASE=${SEED_BASE:-0}
GPU=${GPU:-2}
PORT=${PORT:-8000}
MAX_STEPS=${MAX_STEPS:-400}

# Default tasks: a 5-task subset spanning robocasa365's atomic_seen + composite_seen
# distributions. Override with the TASKS env var (space-separated).
DEFAULT_TASKS="PickPlaceCounterToStove PickPlaceCounterToSink OpenDrawer CloseDrawer TurnOnSinkFaucet"
TASKS_STR=${TASKS:-${DEFAULT_TASKS}}
read -r -a TASKS <<< "${TASKS_STR}"

DEFAULT_MODELS="pi05_robocasa365 pi05_libero pi05_droid"
MODELS_STR=${MODELS:-${DEFAULT_MODELS}}
read -r -a MODELS <<< "${MODELS_STR}"

mkdir -p "${LOG}"

start_server() {
    local cfg="$1"
    local ckpt="${REPO}/checkpoints/viz/${cfg}_pytorch"
    if [ ! -d "${ckpt}" ]; then
        echo "[orch365] missing ckpt for ${cfg}: ${ckpt}"
        return 1
    fi
    echo "[orch365] killing stale servers"
    pkill -f 'serve_policy_attn' 2>/dev/null || true
    sleep 3
    local logf="${LOG}/orch365_server_${cfg}.log"
    : > "${logf}"
    echo "[orch365] starting ${cfg} server on GPU ${GPU} port ${PORT}"
    (cd "${REPO}" && CUDA_VISIBLE_DEVICES=${GPU} uv run viz_sim/serve_policy_attn.py \
        --port=${PORT} --config=${cfg} --dir="${ckpt}" \
        > "${logf}" 2>&1) &
    SERVER_PID=$!
    echo "[orch365] server pid=${SERVER_PID}"
    for i in $(seq 1 120); do
        grep -q "websockets.server:server listening" "${logf}" 2>/dev/null && return 0
        if grep -qE "^Traceback|FileNotFoundError|ValueError|RuntimeError" "${logf}" 2>/dev/null; then
            echo "[orch365] server ${cfg} failed:"; tail -25 "${logf}"
            kill -TERM ${SERVER_PID} 2>/dev/null || true
            return 1
        fi
        sleep 5
    done
    echo "[orch365] server ${cfg} did not come up within 600s"
    tail -25 "${logf}"
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] && kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 3
    pkill -f 'serve_policy_attn' 2>/dev/null || true
    SERVER_PID=""
}

trap 'stop_server' EXIT

echo "[orch365] models=${MODELS[*]}"
echo "[orch365] tasks=${TASKS[*]}"
echo "[orch365] episodes=${EPISODES} seed_base=${SEED_BASE} max_steps=${MAX_STEPS}"

for model in "${MODELS[@]}"; do
    if ! start_server "${model}"; then
        echo "[orch365] skipping ${model} (server failed)"
        continue
    fi
    sleep 5  # extra warmup grace
    for task in "${TASKS[@]}"; do
        echo "[orch365] === ${model} / ${task} ==="
        "${PY}" "${REPO}/viz_sim/eval_robocasa365.py" \
            --model "${model}" --task "${task}" \
            --episodes "${EPISODES}" --seed_base "${SEED_BASE}" \
            --max_steps "${MAX_STEPS}" --port "${PORT}" \
            > "${LOG}/orch365_${model}_${task}.log" 2>&1 \
            || echo "[orch365] ${model}/${task} errored, see log"
    done
    stop_server
done

echo "[orch365] DONE"
