#!/bin/bash
# Re-run pi05_libero and pi05_robocasa365 with fixes applied:
# - libero obs_fn + OSC_POSE controller (eval_runner.MODEL_CFG)
# - robocasa365 assets/droid/norm_stats.json now exists
set -uo pipefail
REPO=/home/edward/projects/openpi_vis
LOG=${REPO}/results/_logs
EPISODES=${EPISODES:-20}
TASK=${TASK:-Lift}

run() {
    local model=$1 cfg=$2 ckpt=$3
    echo "[rerun] === ${model} (cfg=${cfg}) ==="
    pkill -f 'serve_policy_attn' 2>/dev/null || true
    sleep 3
    (cd "${REPO}" && CUDA_VISIBLE_DEVICES=2 uv run viz_sim/serve_policy_attn.py \
        --port=8000 --config="${cfg}" --dir="${ckpt}" \
        > "${LOG}/${model}.server.log" 2>&1) &
    SERVER_PID=$!
    for i in $(seq 1 90); do
        if grep -qE "websockets.server:server listening" "${LOG}/${model}.server.log" 2>/dev/null; then break; fi
        if grep -qE "Error|Traceback" "${LOG}/${model}.server.log" 2>/dev/null; then
            echo "[rerun] server failed; aborting"; return 1; fi
        sleep 5
    done
    sleep 5
    echo "[rerun] running eval"
    conda run -n robocasa_sim --no-capture-output \
        python "${REPO}/viz_sim/eval_runner.py" \
        --model "${model}" --task "${TASK}" --episodes "${EPISODES}" \
        > "${LOG}/${model}.eval.log" 2>&1
    echo "[rerun] eval done; killing server"
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 3
    pkill -f 'serve_policy_attn' 2>/dev/null || true
}

run pi05_libero        pi05_libero ${REPO}/checkpoints/viz/pi05_libero_pytorch
run pi05_robocasa365   pi05_droid  ${REPO}/checkpoints/viz/pi05_robocasa365_pytorch

echo "[rerun] all done"
