#!/usr/bin/env bash
# Install the openpi websocket client into the robocasa_sim conda env.
# openpi-client pins numpy<2 (overly conservative); we install with --no-deps
# and add only the runtime deps that are numpy-version-agnostic.
set -euo pipefail

ENV_NAME="${ENV_NAME:-robocasa_sim}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP=(conda run -n "${ENV_NAME}" --no-capture-output pip)

"${PIP[@]}" install "msgpack>=1.0.5" "websockets>=11.0" "typing_extensions"
"${PIP[@]}" install --no-deps -e "${REPO_ROOT}/packages/openpi-client"

echo "Done. WebsocketClientPolicy is now importable in ${ENV_NAME}."
