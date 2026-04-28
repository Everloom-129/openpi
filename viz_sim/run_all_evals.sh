#!/bin/bash
# Orchestrate overnight eval: bring up each policy server, run the eval client
# in robocasa_sim, take the server down, repeat.
#
# Logs go to results/_logs/{model}.{server,eval}.log so you can tail them.
# State lives in results/_state.json — re-running this script resumes whatever
# wasn't completed yet (per-episode granularity).
set -uo pipefail
REPO=/home/edward/projects/openpi_vis
LOG=${REPO}/results/_logs
mkdir -p "${LOG}"

EPISODES=${EPISODES:-20}
TASK=${TASK:-Lift}

# pi0.5 ckpts share the same server launcher; we vary CONFIG and CKPT.
PI05_DIR=${REPO}/checkpoints/viz
declare -A PI05_CKPT=(
    [pi05_droid]="${PI05_DIR}/pi05_droid_pytorch"
    [pi05_libero]="${PI05_DIR}/pi05_libero_pytorch"
    [pi05_robocasa365]="${PI05_DIR}/pi05_robocasa365_pytorch"
)
declare -A PI05_CONFIG=(
    [pi05_droid]="pi05_droid"
    [pi05_libero]="pi05_libero"
    [pi05_robocasa365]="pi05_droid"  # use droid config for arch/transforms
)

run_pi05() {
    local model=$1
    local cfg=${PI05_CONFIG[$model]}
    local ckpt=${PI05_CKPT[$model]}
    echo "[orch] === ${model} (cfg=${cfg}) ==="

    pkill -f serve_policy_attn || true
    sleep 2
    echo "[orch] starting server (config=${cfg}, ckpt=${ckpt})"
    (cd "${REPO}" && CUDA_VISIBLE_DEVICES=2 uv run viz_sim/serve_policy_attn.py \
        --port=8000 --config="${cfg}" --dir="${ckpt}" \
        > "${LOG}/${model}.server.log" 2>&1) &
    SERVER_PID=$!

    # wait until server responds
    for i in $(seq 1 90); do
        if grep -qE "Serving|listening|on :8000" "${LOG}/${model}.server.log" 2>/dev/null; then break; fi
        sleep 5
    done
    sleep 8  # extra grace for first-call warmup
    echo "[orch] server up (pid=${SERVER_PID}); running eval"

    conda run -n robocasa_sim --no-capture-output \
        python "${REPO}/viz_sim/eval_runner.py" \
        --model "${model}" --task "${TASK}" --episodes "${EPISODES}" \
        > "${LOG}/${model}.eval.log" 2>&1
    echo "[orch] eval done; killing server"
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 3
    pkill -f serve_policy_attn || true
}

run_gr00t() {
    local model=$1
    echo "[orch] === ${model} ==="
    pkill -f run_gr00t_server || true
    sleep 2
    CUDA_VISIBLE_DEVICES=2 MODEL="${REPO}/checkpoints/viz/gr00t_n15_robocasa365" \
        bash "${REPO}/viz_sim/run_gr00t_server.sh" \
        > "${LOG}/${model}.server.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 60); do
        if grep -qE "ready|Listening|server" "${LOG}/${model}.server.log" 2>/dev/null; then break; fi
        sleep 5
    done
    sleep 5
    conda run -n robocasa_sim --no-capture-output \
        python "${REPO}/viz_sim/eval_runner.py" \
        --model "${model}" --task "${TASK}" --episodes "${EPISODES}" \
        > "${LOG}/${model}.eval.log" 2>&1 || echo "[orch] gr00t eval failed (expected if shape mismatch)"
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 3
    pkill -f run_gr00t_server || true
}

for model in pi05_droid pi05_libero pi05_robocasa365; do
    run_pi05 "${model}"
done
run_gr00t gr00t_n15_robocasa365

echo "[orch] all done. summaries at /mnt/sda/edward/projects/robocasa_365/*/${TASK}/_summary.json"
