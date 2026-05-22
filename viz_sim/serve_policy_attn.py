"""Policy server that captures and returns text→image attention.

Wraps `scripts/serve_policy.py`'s checkpoint-loading path with an
`AttnCapturingPolicy` that arms the in-RAM attention buffer in
`gemma_pytorch` around each `infer()` call, reduces the captured tensor to
a compact `(L, H, 512)` float32 (L=18 layers, H=8 heads, 256 ext + 256
wrist patches per head), and ships it back in the websocket response
under the key `text_to_img_attn`. The client picks layer + head-aggregation
mode at display time.

The reduction (per layer, per head):
    - mean over the *real* text tokens (mask-aware, via the policy's own
      input-transform tokenization) → (512,)
    - heads kept separate so the client can switch min/max/avg live

If the buffer is empty (e.g. JAX backend), the field is simply omitted —
clients should fall back to a placeholder.

Run inside the openpi `.venv`:
    bash viz_sim/run_pi0_policy_server.sh
"""
from __future__ import annotations

import dataclasses
import logging
import os
import socket
import sys
from typing import Any

import numpy as np
import tyro

# Match scripts/serve_policy.py JAX setup.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_jax_config import configure_jax_single_gpu  # noqa: E402

from openpi_client import base_policy as _base_policy  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402


TEXT_START_IDX = 768
TOTAL_IMAGE_TOKENS = 512


class AttnCapturingPolicy(_base_policy.BasePolicy):
    """Wraps a Policy and adds `text_to_img_attn` to each infer() result."""

    def __init__(self, inner: _base_policy.BasePolicy):
        self._inner = inner
        # Lazy-import to avoid loading torch in the wrong process.
        from openpi.models_pytorch import gemma_pytorch as _gpt
        self._gpt = _gpt
        from openpi.models.tokenizer import PaligemmaTokenizer  # noqa: F401
        self._has_tokenizer = True

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._inner, "metadata", {})

    def _real_text_count(self, obs: dict) -> int | None:
        """Run the policy's input transform once to get the true text-token count."""
        try:
            transformed = self._inner._input_transform({**obs})  # type: ignore[attr-defined]
            mask = np.asarray(transformed["tokenized_prompt_mask"])
            return int(mask.sum())
        except Exception:
            return None

    def _reduce_layer(self, attn: np.ndarray, n_real_text: int | None, seq_len: int) -> np.ndarray:
        """attn: (1, n_heads, seq, seq) → (n_heads, 512) float32. Heads kept separate."""
        if attn.ndim == 4:
            attn = attn[0]                              # (n_heads, seq, seq)
        if n_real_text and n_real_text > 0:
            t_end = min(TEXT_START_IDX + n_real_text, attn.shape[1])
            t2i = attn[:, TEXT_START_IDX:t_end, :TOTAL_IMAGE_TOKENS]   # (h, n_text, 512)
        else:
            t2i = attn[:, TEXT_START_IDX:seq_len, :TOTAL_IMAGE_TOKENS]
        # mean over the real text tokens; keep heads.
        return t2i.mean(axis=1).astype(np.float32)                     # (n_heads, 512)

    def infer(self, obs: dict) -> dict:
        # Pop perturbation request (if any) before forwarding to the inner
        # policy — avoids unknown-key errors in the input transform.
        perturb = obs.pop("_perturb", None) if isinstance(obs, dict) else None
        if perturb is not None:
            self._gpt.set_perturbation(
                mode=perturb.get("mode"),
                camera=perturb.get("camera"),
                layer=int(perturb.get("layer", 7)),
            )

        # DeLock contrastive prompt guidance: client may set `prompt_neg`
        # (trained prompt that captures post-training bias) and `cpg_w`
        # (guidance scale). When both are present, route through the
        # CPG-capable Policy.infer_cpg path; otherwise fall back to the
        # standard infer().
        prompt_neg = obs.pop("prompt_neg", None) if isinstance(obs, dict) else None
        cpg_w = obs.pop("cpg_w", None) if isinstance(obs, dict) else None
        use_cpg = prompt_neg is not None and cpg_w is not None

        # Attention buffer mixes both forwards under CPG (τ⁺ and τ⁻ both write
        # to the same global buffer); the resulting `text_to_img_attn` would
        # be incoherent. Skip the capture in CPG mode — to compare τ⁺ vs τ⁻
        # attention, run two non-CPG rollouts (one per prompt) and diff client
        # side. This matches paper Fig 4(a)'s methodology anyway.
        capture_attn = not use_cpg
        if capture_attn:
            self._gpt.enable_attn_buffer()
        # Always capture the per-denoising-step action trajectory: it's a
        # `(num_steps, action_horizon, action_dim)` artifact that's small and
        # cheap, and the CPG sweep runner needs it to build the export npz.
        self._gpt.enable_action_traj_buffer()
        try:
            if use_cpg:
                obs_neg = {**obs, "prompt": prompt_neg}
                result = self._inner.infer_cpg(obs, obs_neg, cpg_w=float(cpg_w))
            else:
                result = self._inner.infer(obs)
            buf = self._gpt.get_attn_buffer() if capture_attn else {}
            buf = buf or {}
            traj_buf = self._gpt.get_action_traj_buffer() or []
        finally:
            if capture_attn:
                self._gpt.clear_attn_buffer()
            self._gpt.clear_action_traj_buffer()
            if perturb is not None:
                self._gpt.clear_perturbation()

        if traj_buf:
            try:
                # (num_steps, action_horizon, action_dim) float32
                result["action_trajectory"] = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in traj_buf], axis=0,
                )
            except Exception as e:  # fail-soft, never break inference
                logging.warning("action trajectory stacking failed: %s", e)

        if buf:
            try:
                layers = sorted(buf.keys())
                seq_len = int(buf[layers[0]].shape[-1])
                n_real = self._real_text_count(obs)
                stacked = np.stack(
                    [self._reduce_layer(buf[i], n_real, seq_len) for i in layers],
                    axis=0,
                )  # (L, H, 512)
                result["text_to_img_attn"] = stacked
                result["text_to_img_meta"] = {
                    "layers": layers,
                    "n_heads": int(stacked.shape[1]),
                    "seq_len": seq_len,
                    "n_real_text": int(n_real or 0),
                }
            except Exception as e:  # fail-soft
                logging.warning("attn reduction failed: %s", e)
        return result


@dataclasses.dataclass
class Args:
    config: str = "pi05_droid"
    dir: str = "checkpoints/viz/pi05_droid_pytorch"
    port: int = 8000
    default_prompt: str | None = "pick up the cube"


def main(args: Args) -> None:
    configure_jax_single_gpu()
    logging.basicConfig(level=logging.INFO, force=True)

    train_cfg = _config.get_config(args.config)
    inner = _policy_config.create_trained_policy(
        train_cfg, args.dir, default_prompt=args.default_prompt
    )
    policy = AttnCapturingPolicy(inner)

    hostname = socket.gethostname()
    logging.info("Serving %s from %s on :%d (returning all-layer/all-head text→img attn)",
                 args.config, args.dir, args.port)
    logging.info("Hostname: %s", hostname)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
