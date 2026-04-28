#!/bin/bash
# Multi-task eval: per pi0.5 ckpt, keep the server up while we sweep tasks.
# Skips gr00t (obs-shape mismatch with Panda fixed-base).
set -uo pipefail
REPO=/home/edward/projects/openpi_vis
LOG=${REPO}/results/_logs
EPISODES=${EPISODES:-20}

# task → prompt
declare -A TASK_PROMPT=(
    [Lift]="pick up the cube"
    [Stack]="stack the red cube on top of the green cube"
    [Door]="open the door"
    [PickPlaceCan]="pick up the can and place it in the bin"
    [NutAssemblySquare]="put the square nut onto the peg"
)
# explicit ordering
TASKS=(Lift Stack Door PickPlaceCan NutAssemblySquare)

declare -A PI05_CKPT=(
    [pi05_droid]="${REPO}/checkpoints/viz/pi05_droid_pytorch"
    [pi05_libero]="${REPO}/checkpoints/viz/pi05_libero_pytorch"
    [pi05_robocasa365]="${REPO}/checkpoints/viz/pi05_robocasa365_pytorch"
)
declare -A PI05_CONFIG=(
    [pi05_droid]="pi05_droid"
    [pi05_libero]="pi05_libero"
    [pi05_robocasa365]="pi05_droid"
)

run_model() {
    local model=$1
    local cfg=${PI05_CONFIG[$model]}
    local ckpt=${PI05_CKPT[$model]}

    echo "[multi] === ${model} (cfg=${cfg}) ==="
    pkill -f 'serve_policy_attn' 2>/dev/null || true
    sleep 3
    (cd "${REPO}" && CUDA_VISIBLE_DEVICES=2 uv run viz_sim/serve_policy_attn.py \
        --port=8000 --config="${cfg}" --dir="${ckpt}" \
        > "${LOG}/${model}.server.log" 2>&1) &
    SERVER_PID=$!

    for i in $(seq 1 90); do
        if grep -q "websockets.server:server listening" "${LOG}/${model}.server.log" 2>/dev/null; then
            break
        fi
        if grep -qE "^Traceback|FileNotFoundError|ValueError" "${LOG}/${model}.server.log" 2>/dev/null; then
            echo "[multi] ${model} server failed; skipping"
            kill -TERM ${SERVER_PID} 2>/dev/null || true
            return 1
        fi
        sleep 5
    done
    sleep 5
    echo "[multi] server up"

    for task in "${TASKS[@]}"; do
        local prompt="${TASK_PROMPT[$task]}"
        echo "[multi] ${model} / ${task} / '${prompt}'"
        conda run -n robocasa_sim --no-capture-output \
            python "${REPO}/viz_sim/eval_runner.py" \
            --model "${model}" --task "${task}" --episodes "${EPISODES}" \
            --prompt "${prompt}" \
            > "${LOG}/${model}.${task}.eval.log" 2>&1 || \
            echo "[multi] ${model}/${task} eval errored, see log"
    done

    echo "[multi] ${model} done; killing server"
    kill -TERM ${SERVER_PID} 2>/dev/null || true
    sleep 3
    pkill -f 'serve_policy_attn' 2>/dev/null || true
}

for model in pi05_droid pi05_libero pi05_robocasa365; do
    run_model "${model}" || true
done

echo "[multi] all done"
