#!/usr/bin/env bash
# Install the minimal client-side deps for talking to the GR00T server
# (gr00t/policy/server_client.py uses ZMQ REQ/REP + msgpack) into the
# robocasa_sim conda env. We don't install gr00t itself — viz_sim ships
# a vendored mini-client (gr00t_client.py) that only needs pyzmq + msgpack.
set -euo pipefail

ENV_NAME="${ENV_NAME:-robocasa_sim}"
PIP=(conda run -n "${ENV_NAME}" --no-capture-output pip)

"${PIP[@]}" install "pyzmq>=25" "msgpack>=1.0.5" "scipy" "Pillow"

echo "Done. gr00t_client.PolicyClient is now usable in ${ENV_NAME}."
