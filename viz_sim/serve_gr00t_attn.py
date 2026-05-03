"""GR00T inference server with attention capture — analog of serve_policy_attn.py.

Wraps `gr00t.policy.gr00t_policy.Gr00tPolicy.get_action` so that, on every
inference call, the in-RAM buffers from `viz.gr00t_attn` are armed, populated
by the patched Qwen3 backbone, snapshotted into the response `info` dict, and
cleared in a finally block.

Run inside the Isaac-GR00T uv venv (NOT the openpi .venv or robocasa_sim
conda):

    REPO=/home/edward/projects/openpi_vis
    cd "$REPO/third_party/Isaac-GR00T"
    uv run python "$REPO/viz_sim/serve_gr00t_attn.py" \
        --port 5555 \
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
        --model-path nvidia/GR00T-N1.7-DROID

Or via the launcher:
    CUDA_VISIBLE_DEVICES=0 ATTN=1 bash viz_sim/run_gr00t_server.sh

The launcher routes ATTN=1 here; the default (no ATTN) still launches the
plain `gr00t/eval/run_gr00t_server.py` for users who don't need attention.

Wire format
-----------
The response from `get_action` is `(action_dict, info_dict)`. We attach
attention as `info_dict["attn"]`:

    info["attn"] = {
        "vlm": {                          # one ndarray per Qwen3 layer
            "0":  ndarray(B, H, seq, seq),  float32,
            "1":  ndarray(B, H, seq, seq),
            ...
        },
        "vlm_meta": {
            "input_ids":      ndarray(B, seq),  int64
            "image_mask":     ndarray(B, seq),  bool
            "attention_mask": ndarray(B, seq),  bool
            "image_token_id": int,
        },
        # When DiT capture is implemented (deferred):
        "dit":          dict[str, ndarray] | None,
        "dit_steps":    list[dict[str, ndarray]] | None,
        "action_traj":  list[ndarray] | None,
    }

Layer indices are stringified because msgpack stringifies int dict keys; the
client converts back at parse time. Numpy arrays travel via the existing
`__ndarray_class__` hook in `MsgSerializer`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `viz.gr00t_attn` importable from inside Isaac-GR00T's working dir.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Install attention hooks BEFORE Gr00tPolicy loads the model — so the
# Qwen3Backbone class has the patched forward when it's instantiated.
from viz import gr00t_attn as _gattn  # noqa: E402

_gattn.install_hooks()

# Now safe to import gr00t — the patched class is already in place.
from dataclasses import dataclass  # noqa: E402

import tyro  # noqa: E402

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from gr00t.policy.server_client import PolicyServer  # noqa: E402


@dataclass
class ServerConfig:
    """Mirrors `gr00t/eval/run_gr00t_server.py`'s ServerConfig (model-only path)."""

    model_path: str = "nvidia/GR00T-N1.7-DROID"
    """Path to the model checkpoint directory (or HF id)."""

    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    """Embodiment tag (case-insensitive)."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    port: int = 5555

    strict: bool = True

    capture_dit_steps: int = 1
    """Number of early flow-matching steps to capture and average for DiT.
    Currently a no-op (DiT hooks pending) but kept for API parity."""


def _attn_capturing_wrapper(policy: Gr00tPolicy, capture_dit_steps: int):
    """Return a get_action callable that arms/clears buffers around inference."""

    orig_get_action = policy.get_action

    def get_action_with_attn(observation: dict, options: dict | None = None):
        _gattn.enable_all(dit_capture_steps=capture_dit_steps)
        try:
            result = orig_get_action(observation, options=options)
            # `result` is (action_dict, info_dict). info_dict comes from the
            # underlying Gr00tPolicy._get_action which returns ({}, {}) — empty.
            if isinstance(result, tuple) and len(result) == 2:
                action_dict, info_dict = result
            else:
                # defensive: some policies return just an action dict
                action_dict, info_dict = result, {}
            if not isinstance(info_dict, dict):
                info_dict = {}

            attn_payload: dict = {}
            vlm_buf = _gattn.get_vlm_attn_buffer()
            if vlm_buf:
                # Stringify int keys for msgpack round-tripping.
                attn_payload["vlm"] = {str(k): v for k, v in vlm_buf.items()}
            vlm_meta = _gattn.get_vlm_token_meta()
            if vlm_meta:
                attn_payload["vlm_meta"] = vlm_meta
            dit_buf = _gattn.get_dit_attn_buffer()
            if dit_buf:
                attn_payload["dit"] = {str(k): v for k, v in dit_buf.items()}
            dit_steps = _gattn.get_dit_attn_steps_buffer()
            if dit_steps:
                attn_payload["dit_steps"] = [
                    {str(k): v for k, v in step.items()} for step in dit_steps
                ]
            action_traj = _gattn.get_action_traj_buffer()
            if action_traj:
                attn_payload["action_traj"] = action_traj  # list[ndarray]

            if attn_payload:
                info_dict["attn"] = attn_payload
            return action_dict, info_dict
        finally:
            _gattn.clear_all()

    return get_action_with_attn


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server (attention-capture variant)...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path:     {config.model_path}")
    print(f"  Device:         {config.device}")
    print(f"  Host/Port:      {config.host}:{config.port}")
    print(f"  capture_dit_steps={config.capture_dit_steps}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )

    # Replace policy.get_action with the attention-capturing wrapper.
    # PolicyServer.__init__ snapshots `policy.get_action` into its endpoint
    # registry, so we patch BEFORE constructing the server.
    policy.get_action = _attn_capturing_wrapper(policy, config.capture_dit_steps)

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    print(f"\n✓ Server ready (with attention capture) — listening on "
          f"{config.host}:{config.port}\n")
    server.run()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
