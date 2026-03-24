"""Online inference: load Pi0.5 model once, run forward pass, return attention slices.

The model is cached via @st.cache_resource so it's loaded once per checkpoint
and reused across Streamlit reruns.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import streamlit as st


def _infer_config_name(checkpoint_dir: str) -> str:
    """Infer the training config name from the checkpoint directory name.

    Strips a trailing ``_pytorch`` suffix, then tries progressively shorter
    underscore-delimited names against the registered configs.  Falls back to
    ``pi05_droid`` / ``pi0_droid`` when nothing matches.
    """
    import os as _os
    from openpi.training import config as _cfg

    raw = _os.path.basename(_os.path.normpath(checkpoint_dir))
    candidate = raw[: -len("_pytorch")] if raw.endswith("_pytorch") else raw

    while candidate:
        try:
            _cfg.get_config(candidate)
            return candidate
        except (ValueError, KeyError):
            pass
        idx = candidate.rfind("_")
        if idx == -1:
            break
        candidate = candidate[:idx]

    return "pi05_droid" if "pi05" in raw.lower() else "pi0_droid"


@st.cache_resource
def load_model(checkpoint_dir: str, device: str = "cuda:0"):
    """Load Pi0 or Pi0.5 policy once, cache across reruns.

    The config is inferred from the checkpoint directory name:
    'pi05' → pi05_droid, 'pi0' → pi0_droid.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        sys.path.insert(0, os.path.join(project_root, "src"))

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    config_name = _infer_config_name(checkpoint_dir)
    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)
    return policy


def run_inference(policy, example: dict) -> dict:
    """Run one forward pass with in-RAM attention capture and return a slice dict.

    Uses the gemma_pytorch attention buffer (same mechanism as pipeline.py) so no
    files are written to disk. Returns a dict with schema meta/images/prefix for
    direct use by the views. Full attention matrix is stored for all 18 layers.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    for p in [project_root, os.path.join(project_root, "src"), os.path.join(project_root, "viz")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from openpi.models_pytorch import gemma_pytorch as _gpt
    from viz.dashboard.loader import TEXT_START_IDX, TOTAL_IMAGE_TOKENS

    # ── Capture attention into RAM buffer ────────────────────────────────────
    _gpt.enable_attn_buffer()
    try:
        _ = policy.infer(example)
        buf = _gpt.get_attn_buffer()
    finally:
        _gpt.clear_attn_buffer()

    if not buf:
        st.warning("Attention buffer is empty — model may not have run the prefix forward pass.")
        return {}

    # ── Detect dimensions ────────────────────────────────────────────────────
    first = next(iter(buf.values()))
    seq_len = int(first.shape[-1])
    n_text = seq_len - TEXT_START_IDX

    # ── Tokenize instruction for token labels ────────────────────────────────
    token_texts: list[str] = [f"tok_{i}" for i in range(n_text)]
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer()
        joint_pos = example.get("observation/joint_position", np.zeros(7))
        gripper_pos = np.atleast_1d(example.get("observation/gripper_position", np.zeros(1)))
        state = np.concatenate([np.atleast_1d(joint_pos), gripper_pos])
        instruction = example.get("prompt", "").strip()
        disc = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, disc))
        full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "
        ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in ids]
    except Exception as e:
        st.warning(f"Tokenizer failed: {e}. Using generic token labels.")

    n_text_actual = min(n_text, len(token_texts))

    # ── Resize images to 224×224 ─────────────────────────────────────────────
    def _to_224(img: np.ndarray | None) -> np.ndarray | None:
        if img is None:
            return None
        from PIL import Image as _PIL
        return np.array(
            _PIL.fromarray(img.astype(np.uint8)).resize((224, 224), _PIL.BILINEAR),
            dtype=np.uint8,
        )

    # ── Build prefix dict from buffer ────────────────────────────────────────
    prefix: dict[str, Any] = {}
    for layer_idx, attn in buf.items():
        if attn.ndim == 4:
            attn = attn[0]          # drop batch dim → (n_heads, seq, seq)
        attn = attn.astype(np.float32)

        t2i = attn[:, TEXT_START_IDX : TEXT_START_IDX + n_text_actual, :TOTAL_IMAGE_TOKENS]
        layer_data: dict[str, np.ndarray] = {"text_to_img": t2i, "full": attn}

        prefix[f"layer_{layer_idx}"] = layer_data

    return {
        "meta": {
            "prefix_len": TEXT_START_IDX,
            "seq_len": seq_len,
            "n_real_tokens": n_text_actual,
            "instruction": example.get("prompt", ""),
            "token_texts": token_texts[:n_text_actual],
        },
        "images": {
            "exterior": _to_224(example.get("observation/exterior_image_1_left")),
            "wrist":    _to_224(example.get("observation/wrist_image_left")),
        },
        "prefix": prefix,
    }


def list_online_checkpoints(checkpoint_root: str = "checkpoints/viz") -> list[str]:
    """List available checkpoint directories for online inference."""
    if not os.path.exists(checkpoint_root):
        return []
    return sorted(
        d for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d))
    )
