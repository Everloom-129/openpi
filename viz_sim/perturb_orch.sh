#!/bin/bash
# Orchestrate the layer-7 KV-perturbation experiment for pi05_robocasa365.
# 5 tasks x 7 conditions x N episodes (matched seeds). One server, no restarts.
set -uo pipefail
REPO=/home/edward/projects/openpi_vis
LOG=${REPO}/results/_logs
PY=/home/edward/miniconda3/envs/robocasa_sim/bin/python
EPISODES=${EPISODES:-10}
SEED_BASE=${SEED_BASE:-1000}
BLOCK_BASE=${BLOCK_BASE:-0}   # set to 1 to zero torso/base, force arm-only

mkdir -p "${LOG}"

TASKS=(Lift Stack Door PickPlaceCan NutAssemblySquare)
CONDITIONS=(baseline ext_zero_max ext_strengthen_max ext_zero_min wrist_zero_max wrist_strengthen_max wrist_zero_min)

echo "[perturb-orch] killing stale servers"
pkill -f 'serve_policy_attn' 2>/dev/null || true
sleep 3

echo "[perturb-orch] starting pi05_robocasa365 server on GPU 2"
(cd "${REPO}" && CUDA_VISIBLE_DEVICES=2 uv run viz_sim/serve_policy_attn.py \
    --port=8000 --config=pi05_robocasa365 \
    --dir="${REPO}/checkpoints/viz/pi05_robocasa365_pytorch" \
    > "${LOG}/perturb_server.log" 2>&1) &
SERVER_PID=$!
echo "[perturb-orch] server pid=${SERVER_PID}"

for i in $(seq 1 90); do
    grep -q "websockets.server:server listening" "${LOG}/perturb_server.log" 2>/dev/null && break
    if grep -qE "^Traceback|FileNotFoundError|ValueError" "${LOG}/perturb_server.log" 2>/dev/null; then
        echo "[perturb-orch] server failed:"; tail -25 "${LOG}/perturb_server.log"
        kill -TERM ${SERVER_PID} 2>/dev/null || true; exit 1
    fi
    sleep 5
done
sleep 8
echo "[perturb-orch] server up; running $((${#TASKS[@]} * ${#CONDITIONS[@]})) cells x ${EPISODES} eps"

for task in "${TASKS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        echo "[perturb-orch] === ${task} / ${cond} ==="
        EXTRA_ARGS=()
        [ "${BLOCK_BASE}" = "1" ] && EXTRA_ARGS+=(--block_base)
        "${PY}" "${REPO}/viz_sim/eval_perturb.py" \
            --task "${task}" --condition "${cond}" \
            --episodes "${EPISODES}" --seed_base "${SEED_BASE}" \
            "${EXTRA_ARGS[@]}" \
            > "${LOG}/perturb_${task}_${cond}.log" 2>&1 \
            || echo "[perturb-orch] ${task}/${cond} errored, see log"
    done
done

echo "[perturb-orch] killing server"
kill -TERM ${SERVER_PID} 2>/dev/null || true
sleep 3
pkill -f 'serve_policy_attn' 2>/dev/null || true

echo "[perturb-orch] rendering ep_000 comparison videos + per-task report"
cd "${REPO}" && uv run python viz_sim/render_perturb_video.py 2>&1 | tail -20
cd "${REPO}" && uv run python viz_sim/build_perturb_report.py 2>&1 | tail -20

echo "[perturb-orch] rendering attention videos for SUCCESSFUL episodes"
cd "${REPO}" && uv run python viz_sim/render_perturb_success.py 2>&1 | tail -40

echo "[perturb-orch] building A+B summary heatmap (results/perturb_summary.png)"
cd "${REPO}" && uv run python viz_sim/build_perturb_summary_fig.py 2>&1 | tail -10
echo "[perturb-orch] DONE"
