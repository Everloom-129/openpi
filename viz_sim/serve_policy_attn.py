"""Policy server that captures and returns text→image attention.

Wraps `scripts/serve_policy.py`'s checkpoint-loading path with an
`AttnCapturingPolicy` that arms the in-RAM attention buffer in
`gemma_pytorch` around each `infer()` call, reduces the captured tensor to
a compact `(512,)` float32 (256 ext patches + 256 wrist patches), and
ships it back in the websocket response under the key
`text_to_img_attn`.

The reduction:
    - pick `--attn_layer` (default 7, mid-network — most spatially meaningful)
    - mean over heads
    - mean over the *real* text tokens (mask-aware, via the policy's own
      input-transform tokenization)

If the buffer is empty (e.g. JAX backend), the field is simply omitted —
clients should fall back to a placeholder.

Run inside the openpi `.venv`:
    bash viz_sim/run_policy_server.sh
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

    def __init__(self, inner: _base_policy.BasePolicy, *, layer: int = 7):
        self._inner = inner
        self._layer = layer
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

    def _reduce(self, attn: np.ndarray, n_real_text: int | None, seq_len: int) -> np.ndarray:
        """attn: (1, n_heads, seq, seq) → (512,) float32."""
        if attn.ndim == 4:
            attn = attn[0]                              # (n_heads, seq, seq)
        if n_real_text and n_real_text > 0:
            t_end = min(TEXT_START_IDX + n_real_text, attn.shape[1])
            t2i = attn[:, TEXT_START_IDX:t_end, :TOTAL_IMAGE_TOKENS]   # (h, n_text, 512)
        else:
            t2i = attn[:, TEXT_START_IDX:seq_len, :TOTAL_IMAGE_TOKENS]
        # mean over heads then over text tokens
        return t2i.mean(axis=0).mean(axis=0).astype(np.float32)         # (512,)

    def infer(self, obs: dict) -> dict:
        self._gpt.enable_attn_buffer()
        try:
            result = self._inner.infer(obs)
            buf = self._gpt.get_attn_buffer() or {}
        finally:
            self._gpt.clear_attn_buffer()

        if buf and self._layer in buf:
            try:
                attn = buf[self._layer]
                seq_len = int(attn.shape[-1])
                n_real = self._real_text_count(obs)
                result["text_to_img_attn"] = self._reduce(attn, n_real, seq_len)
                result["text_to_img_meta"] = {
                    "layer": self._layer,
                    "seq_len": seq_len,
                    "n_real_text": int(n_real or 0),
                }
            except Exception as e:  # fail-soft
                logging.warning("attn reduction failed: %s", e)
        elif buf:
            logging.warning(
                "attn layer %d not in buffer (have: %s)", self._layer, sorted(buf.keys())
            )
        return result


@dataclasses.dataclass
class Args:
    config: str = "pi05_droid"
    dir: str = "checkpoints/viz/pi05_droid_pytorch"
    port: int = 8000
    default_prompt: str | None = "pick up the cube"
    attn_layer: int = 7


def main(args: Args) -> None:
    configure_jax_single_gpu()
    logging.basicConfig(level=logging.INFO, force=True)

    train_cfg = _config.get_config(args.config)
    inner = _policy_config.create_trained_policy(
        train_cfg, args.dir, default_prompt=args.default_prompt
    )
    policy = AttnCapturingPolicy(inner, layer=args.attn_layer)

    hostname = socket.gethostname()
    logging.info("Serving %s from %s on :%d (attn layer=%d)",
                 args.config, args.dir, args.port, args.attn_layer)
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
