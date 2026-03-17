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


@st.cache_resource
def load_model(checkpoint_dir: str, device: str = "cuda:0"):
    """Load Pi0.5 policy once, cache across reruns."""
    # Add project root to path so openpi imports work
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        sys.path.insert(0, os.path.join(project_root, "src"))

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    config = _config.get_config("pi05_droid")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)
    return policy


def run_inference(
    policy,
    example: dict,
    layers_prefix_output_dir: str = "results/layers_prefix",
) -> dict:
    """Run one forward pass and return attention slice dict.

    The model writes attention maps to `layers_prefix_output_dir` as side effects.
    We then read them back and build an in-memory slice dict.

    Returns a slice dict with keys: meta, images, prefix (same schema as HDF5).
    """
    import os

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

    from viz.dashboard.loader import make_slice_dict_from_npy

    # Run inference — side-effect: writes attn_map_layer_*.npy
    _ = policy.infer(example)

    # Tokenize to get token texts
    token_texts = None
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        import numpy as _np

        tokenizer = PaligemmaTokenizer()
        joint_pos = example.get("observation/joint_position", _np.zeros(7))
        gripper_pos = example.get("observation/gripper_position", _np.zeros(1))
        state = _np.concatenate([joint_pos, gripper_pos])
        instruction = example.get("prompt", "").strip().replace("_", " ").replace("\n", " ")
        discretized_state = _np.digitize(state, bins=_np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "
        token_ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in token_ids]
    except Exception as e:
        st.warning(f"Tokenizer failed: {e}. Token labels will be generic.")

    # Resize images to 224×224
    ext_img = example.get("observation/exterior_image_1_left")
    wrist_img = example.get("observation/wrist_image_left")

    instruction = example.get("prompt", "")

    slice_dict = make_slice_dict_from_npy(
        layers_prefix_dir=layers_prefix_output_dir,
        token_texts=token_texts,
        instruction=instruction,
        ext_img=ext_img,
        wrist_img=wrist_img,
    )
    return slice_dict


def list_online_checkpoints(checkpoint_root: str = "checkpoints/viz") -> list[str]:
    """List available checkpoint directories for online inference."""
    if not os.path.exists(checkpoint_root):
        return []
    return sorted(
        d for d in os.listdir(checkpoint_root)
        if os.path.isdir(os.path.join(checkpoint_root, d))
    )
