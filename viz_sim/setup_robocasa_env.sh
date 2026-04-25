#!/usr/bin/env bash
# Create a dedicated conda env for the robocasa MuJoCo simulator.
#
# This env is intentionally separate from the openpi .venv:
#   - robocasa needs numpy==2.2.5 / mujoco==3.3.1
#   - openpi (JAX) needs numpy<2.0
#
# The simulator runs here; pi0/pi0.5 inference runs in the openpi .venv and is
# reached over a websocket via scripts/serve_policy.py (see viz_sim/README plan).

set -euo pipefail

ENV_NAME="${ENV_NAME:-robocasa_sim}"
PY_VER="${PY_VER:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Conda env
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

# Use `conda run` so we don't need to source conda's shell hook.
PIP=(conda run -n "${ENV_NAME}" --no-capture-output pip)

# 2. Core deps. Pin the versions robocasa requires.
"${PIP[@]}" install --upgrade pip wheel
"${PIP[@]}" install \
    "mujoco==3.3.1" \
    "numpy==2.2.5" \
    "numba==0.61.2" \
    "scipy==1.15.3" \
    pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi \
    gymnasium \
    "qpsolvers[quadprog]>=4.3.1" \
    requests

# tianshou/lerobot pins from robocasa's setup.py — install only if you need
# the dataset-collection tooling. Skipped here because they pull heavy deps
# (torch, etc.) that we do not need just to drive the viewer.
# "${PIP[@]}" install "tianshou==0.4.10" "lerobot==0.3.3"

# 3. Editable installs of the bundled simulators.
"${PIP[@]}" install -e "${REPO_ROOT}/third_party/robosuite"
"${PIP[@]}" install -e "${REPO_ROOT}/third_party/robocasa"

# 4. macros_private.py — robocasa reads DATASET_BASE_PATH from here.
MACROS_PRIVATE="${REPO_ROOT}/third_party/robocasa/robocasa/macros_private.py"
if [ ! -f "${MACROS_PRIVATE}" ]; then
    conda run -n "${ENV_NAME}" --no-capture-output \
        python "${REPO_ROOT}/third_party/robocasa/robocasa/scripts/setup_macros.py" || true
fi

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Then test with:       python ${REPO_ROOT}/viz_sim/test_viewer.py"
