"""JAX model inference with attention capture for the dashboard.

Supports both offline (single forward pass) and live (from websocket server)
attention visualization. The returned dict uses the same schema as inference.py
so all existing views work unchanged.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np
import streamlit as st


def _ensure_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    for p in [project_root, os.path.join(project_root, "src"), os.path.join(project_root, "viz")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _infer_config_name(checkpoint_dir: str) -> str:
    """Infer the training config name from the checkpoint directory name.

    Same logic as inference.py but works for JAX checkpoint dirs (no _pytorch suffix).
    """
    from openpi.training import config as _cfg

    raw = os.path.basename(os.path.normpath(checkpoint_dir))
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


def load_jax_model(config_name: str, checkpoint_path: str):
    """Load a JAX Pi0 model with checkpoint weights.

    Returns (model, config) where model is the Pi0 NNX module.
    """
    _ensure_paths()

    from openpi.models import model as _model
    from openpi.shared import download
    from openpi.training import config as _config

    config = _config.get_config(config_name)
    model_config = config.model
    ckpt = pathlib.Path(str(download.maybe_download(checkpoint_path)))
    # restore_params expects the directory containing _METADATA.
    # If the caller passed the top-level checkpoint dir, append "params/".
    params_path = ckpt / "params" if (ckpt / "params").is_dir() else ckpt
    params = _model.restore_params(params_path)
    model = model_config.load(params)
    model.eval()
    return model, config


@st.cache_resource
def load_jax_model_cached(checkpoint_dir: str):
    """Load a JAX model with Streamlit caching. Infers config from dir name.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory (must contain ``params/``).

    Returns (model, config).
    """
    config_name = _infer_config_name(checkpoint_dir)
    return load_jax_model(config_name, checkpoint_dir)


def run_jax_inference(model, example: dict, *, is_pi05: bool = True) -> dict:
    """Run a JAX forward pass with attention capture.

    Uses Pi0.forward_with_attention() which returns prefix and suffix attention
    from all 18 layers. Returns a dict with the same schema as inference.py's
    run_inference() so all dashboard views work unchanged.

    Args:
        model: A Pi0 NNX model (from load_jax_model).
        example: The input dict (same format as policy.infer input).
        is_pi05: Whether the model is pi0.5 (affects tokenization).

    Returns:
        Dict with keys: meta, images, prefix, joint, pred_action, gt_action.
    """
    _ensure_paths()

    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from viz.dashboard.loader import TEXT_START_IDX
    from viz.dashboard.loader import TOTAL_IMAGE_TOKENS

    rng = jax.random.key(0)

    # Build observation from example dict (assumes input transform already applied).
    images = {}
    image_masks = {}
    for img_key in ["observation/exterior_image_1_left", "observation/wrist_image_left"]:
        if img_key in example:
            img = example[img_key]
            if img.ndim == 3:
                img = img[None]
            img = jnp.array(img, dtype=jnp.float32) / 255.0
            short_key = img_key.split("/")[-1]
            if "exterior" in short_key:
                images["base_0_rgb"] = img
                image_masks["base_0_rgb"] = jnp.ones((img.shape[0],), dtype=jnp.bool_)
            if "wrist" in short_key:
                images["left_wrist_0_rgb"] = img
                image_masks["left_wrist_0_rgb"] = jnp.ones((img.shape[0],), dtype=jnp.bool_)

    if "right_wrist_0_rgb" not in images:
        dummy = jnp.zeros_like(next(iter(images.values())))
        images["right_wrist_0_rgb"] = dummy
        image_masks["right_wrist_0_rgb"] = jnp.zeros((dummy.shape[0],), dtype=jnp.bool_)

    state = jnp.array(
        example.get("observation/joint_position", np.zeros(32)), dtype=jnp.float32,
    )
    if state.ndim == 1:
        state = state[None]

    # Tokenize prompt.
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer()
    prompt = example.get("prompt", "")
    max_token_len = getattr(model, "max_token_len", 200)

    token_ids, token_mask = tokenizer.tokenize(prompt, state=state[0] if is_pi05 else None)
    token_ids = np.array(token_ids[:max_token_len], dtype=np.int32)
    token_mask = np.array(token_mask[:max_token_len], dtype=bool)

    n_real_tokens = int(token_mask.sum())
    token_texts = [tokenizer._tokenizer.id_to_piece(int(i)) for i in token_ids[:n_real_tokens]]  # noqa: SLF001

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=jnp.array(token_ids[None]),
        tokenized_prompt_mask=jnp.array(token_mask[None], dtype=bool),
    )

    # Run forward with attention capture.
    actions, prefix_attn, suffix_attn = model.forward_with_attention(rng, observation)

    actions_np = np.array(actions)

    # Build prefix dict (same schema as inference.py).
    prefix: dict[str, Any] = {}
    seq_len = 0
    if prefix_attn:
        first = next(iter(prefix_attn.values()))
        seq_len = int(first.shape[-1])
        for layer_idx, attn in prefix_attn.items():
            attn_np = np.array(attn)
            if attn_np.ndim == 4:
                attn_np = attn_np[0]  # drop batch
            attn_np = attn_np.astype(np.float32)
            t2i = attn_np[:, TEXT_START_IDX: TEXT_START_IDX + n_real_tokens, :TOTAL_IMAGE_TOKENS]
            prefix[f"layer_{layer_idx}"] = {"text_to_img": t2i, "full": attn_np}

    # Build joint (suffix) dict.
    joint: dict[str, Any] = {}
    for layer_idx, attn in suffix_attn.items():
        attn_np = np.array(attn)
        if attn_np.ndim == 4:
            attn_np = attn_np[0]
        attn_np = attn_np.astype(np.float32)
        a2i = attn_np[:, :, :TOTAL_IMAGE_TOKENS]
        a2t = attn_np[:, :, TEXT_START_IDX: TEXT_START_IDX + n_real_tokens]
        a2a = attn_np[:, :, seq_len:]
        joint[f"layer_{layer_idx}"] = {
            "action_to_img": a2i,
            "action_to_text": a2t,
            "action_to_action": a2a,
        }

    def _to_224(img):
        if img is None:
            return None
        from PIL import Image as PILImage
        if isinstance(img, np.ndarray) and img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return np.array(PILImage.fromarray(img).resize((224, 224), PILImage.BILINEAR), dtype=np.uint8)

    return {
        "meta": {
            "prefix_len": TEXT_START_IDX,
            "seq_len": seq_len,
            "n_real_tokens": n_real_tokens,
            "instruction": prompt,
            "token_texts": token_texts,
        },
        "images": {
            "exterior": _to_224(example.get("observation/exterior_image_1_left")),
            "wrist": _to_224(example.get("observation/wrist_image_left")),
        },
        "prefix": prefix,
        "joint": joint or None,
        "pred_action": actions_np,
        "gt_action": example.get("gt_action"),
    }


def attn_dict_from_weights(attn_weights_stack, seq_len: int, n_real_tokens: int) -> dict[str, Any]:
    """Convert stacked attention weights (n_layers, B, N, T, S) to dashboard prefix dict.

    Useful for converting raw JAX attention output without running full inference.
    """
    from viz.dashboard.loader import TEXT_START_IDX
    from viz.dashboard.loader import TOTAL_IMAGE_TOKENS

    prefix: dict[str, Any] = {}
    attn_np = np.array(attn_weights_stack)
    for layer_idx in range(attn_np.shape[0]):
        layer_attn = attn_np[layer_idx, 0].astype(np.float32)
        t2i = layer_attn[:, TEXT_START_IDX: TEXT_START_IDX + n_real_tokens, :TOTAL_IMAGE_TOKENS]
        prefix[f"layer_{layer_idx}"] = {"text_to_img": t2i, "full": layer_attn}
    return prefix
