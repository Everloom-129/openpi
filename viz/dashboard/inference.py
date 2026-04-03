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
    files are written to disk. Returns a dict with schema meta/images/prefix/joint
    for direct use by the views. Full attention matrix is stored for all 18 layers.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    for p in [project_root, os.path.join(project_root, "src"), os.path.join(project_root, "viz")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from openpi.models_pytorch import gemma_pytorch as _gpt
    from viz.dashboard.loader import TEXT_START_IDX, TOTAL_IMAGE_TOKENS

    # ── Capture attention into RAM buffer ────────────────────────────────────
    _gpt.enable_attn_buffer()
    _gpt.enable_suffix_attn_buffer()
    try:
        result = policy.infer(example)
        buf = _gpt.get_attn_buffer()
        suffix_buf = _gpt.get_suffix_attn_buffer()
    finally:
        _gpt.clear_attn_buffer()
        _gpt.clear_suffix_attn_buffer()

    if not buf:
        st.warning("Attention buffer is empty — model may not have run the prefix forward pass.")
        return {}

    # ── Detect dimensions ────────────────────────────────────────────────────
    first = next(iter(buf.values()))
    seq_len = int(first.shape[-1])
    n_text = seq_len - TEXT_START_IDX

    # ── Tokenize instruction for token labels ────────────────────────────────
    # Run the policy's own input transform pipeline (which includes quantile
    # normalization + prompt cleaning + discretization) on a shallow copy so we
    # get the *exact* token IDs the model saw, then decode them to text pieces.
    token_texts: list[str] = [f"tok_{i}" for i in range(n_text)]
    n_text_actual = n_text
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        # Shallow-copy the dict so TokenizePrompt's pop("prompt") doesn't clobber the original.
        inputs_copy = {**example}
        transformed = policy._input_transform(inputs_copy)
        token_ids = np.asarray(transformed["tokenized_prompt"])
        token_mask_arr = np.asarray(transformed["tokenized_prompt_mask"])
        n_real = int(token_mask_arr.sum())
        real_ids = token_ids[:n_real].tolist()
        tokenizer = PaligemmaTokenizer()
        token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in real_ids]
        n_text_actual = min(n_text, len(token_texts))
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

    # ── Build joint dict from suffix buffer ──────────────────────────────────
    joint: dict[str, Any] = {}
    for layer_idx, sa in suffix_buf.items():
        if sa.ndim == 4:
            sa = sa[0]              # drop batch dim → (n_heads, 8, k)
        sa = sa.astype(np.float32)
        a2i = sa[:, :, :TOTAL_IMAGE_TOKENS]                               # (n_heads, 8, 512)
        a2t = sa[:, :, TEXT_START_IDX : TEXT_START_IDX + n_text_actual]  # (n_heads, 8, n_text)
        a2a = sa[:, :, seq_len:]                                          # (n_heads, 8, ≤8)
        joint[f"layer_{layer_idx}"] = {
            "action_to_img":    a2i,
            "action_to_text":   a2t,
            "action_to_action": a2a,
        }

    pred_action = result.get("actions")    # (8, 8) float32
    gt_action   = example.get("gt_action") # (8, 8) float32 or None

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
        "prefix":      prefix,
        "joint":       joint or None,
        "pred_action": pred_action,
        "gt_action":   gt_action,
    }


def list_online_checkpoints(checkpoint_root: str = "checkpoints/viz") -> list[str]:
    """List available checkpoint directories for online inference."""
    if not os.path.exists(checkpoint_root):
        return []
    return sorted(
        d for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d))
    )


def detect_backend(checkpoint_dir: str) -> str:
    """Detect whether a checkpoint is PyTorch or JAX.

    Returns "pytorch" if ``model.safetensors`` exists, "jax" if ``params/``
    exists, or "unknown".
    """
    if os.path.isfile(os.path.join(checkpoint_dir, "model.safetensors")):
        return "pytorch"
    if os.path.isdir(os.path.join(checkpoint_dir, "params")):
        return "jax"
    return "unknown"


def list_online_checkpoints_by_backend(
    checkpoint_root: str = "checkpoints/viz", backend: str = "all"
) -> list[str]:
    """List checkpoints filtered by backend type.

    Args:
        checkpoint_root: Root directory containing checkpoint subdirs.
        backend: "pytorch", "jax", or "all".
    """
    all_ckpts = list_online_checkpoints(checkpoint_root)
    if backend == "all":
        return all_ckpts
    return [
        d for d in all_ckpts
        if detect_backend(os.path.join(checkpoint_root, d)) == backend
    ]
