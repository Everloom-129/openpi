"""HDF5 load helpers with Streamlit cache.

All functions take h5_path as first argument so the cache key encodes
(checkpoint, episode, frame) without any mutable state.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512   # ext (0:256) + wrist (256:512)
TEXT_START_IDX = 768
NUM_LAYERS = 18


# ── Directory helpers ──────────────────────────────────────────────────────────

def list_checkpoints(attn_h5_root: str = "attn_h5") -> list[str]:
    root = Path(attn_h5_root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def list_episodes(checkpoint: str, attn_h5_root: str = "attn_h5") -> list[str]:
    ep_dir = Path(attn_h5_root) / checkpoint
    if not ep_dir.exists():
        return []
    return sorted(d.name for d in ep_dir.iterdir() if d.is_dir())


def list_frames(checkpoint: str, episode: str, attn_h5_root: str = "attn_h5") -> list[int]:
    frame_dir = Path(attn_h5_root) / checkpoint / episode
    if not frame_dir.exists():
        return []
    frames = []
    for f in sorted(frame_dir.glob("*.h5")):
        try:
            frames.append(int(f.stem))
        except ValueError:
            pass
    return frames


def h5_path(checkpoint: str, episode: str, frame: int, attn_h5_root: str = "attn_h5") -> str:
    return str(Path(attn_h5_root) / checkpoint / episode / f"{frame:05d}.h5")


def list_layers_in_h5(path: str) -> list[int]:
    """Return sorted list of layer indices available in prefix group."""
    if not os.path.exists(path):
        return []
    with h5py.File(path, "r") as f:
        prefix = f.get("prefix", {})
        layers = []
        for k in prefix:
            if k.startswith("layer_"):
                try:
                    layers.append(int(k.split("_")[1]))
                except ValueError:
                    pass
    return sorted(layers)


@st.cache_data(ttl=300)
def load_gt_action(path: str) -> np.ndarray | None:
    """Load ground-truth actions: float32[OPEN_LOOP_HORIZON, action_dim].

    action_dim = 8: [joint_velocity×7, gripper_position×1].
    Last rows may be NaN if the frame is within OPEN_LOOP_HORIZON of episode end.
    Returns None if the file predates GT action capture.
    """
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        if "gt_action" not in f:
            return None
        return f["gt_action"][()].astype(np.float32)


@st.cache_data(ttl=300)
def load_pred_action(path: str) -> np.ndarray | None:
    """Load Pi0.5 predicted actions: float32[OPEN_LOOP_HORIZON, action_dim].

    action_dim = 8: [joint_velocity×7, gripper_position×1].
    Returns None if the file predates predicted action capture.
    """
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        if "pred_action" not in f:
            return None
        return f["pred_action"][()].astype(np.float32)


@st.cache_data(ttl=300)
def load_action_to_img(path: str, layer: int) -> np.ndarray | None:
    """Load action-token → image attention: (n_heads, 8_steps, 512_patches) float32.

    Written by pipeline.py via the suffix attention buffer.
    Returns None if the file predates suffix capture or the layer is absent.
    """
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        key = f"suffix/layer_{layer}/action_to_img"
        if key not in f:
            return None
        return f[key][()].astype(np.float32)


def has_full_matrix(path: str, layer: int) -> bool:
    if not os.path.exists(path):
        return False
    with h5py.File(path, "r") as f:
        return f"prefix/layer_{layer}/full" in f


@st.cache_data(ttl=300)
def load_suffix_denoising(path: str, layer: int) -> dict | None:
    """Load per-NFE-step denoising attention for one layer.

    Written by pipeline.py via ``suffix_steps_buffer`` → ``/suffix_denoising/``.

    Returns dict with:
      n_steps            int
      action_to_img_steps  float32(n_steps, 8_action, 512_patches)  — mean over heads
      group_masses         float32(n_steps, 4)  — [ext, wrist, text, action_self]
    Returns None if the file has no denoising data (pre-dates this feature).
    """
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        grp_key = f"suffix_denoising/layer_{layer}"
        if grp_key not in f:
            return None
        n_steps = int(f["suffix_denoising/n_steps"][()])
        return {
            "n_steps":              n_steps,
            "action_to_img_steps":  f[f"{grp_key}/action_to_img_steps"][()].astype(np.float32),
            "group_masses":         f[f"{grp_key}/group_masses"][()].astype(np.float32),
        }


@st.cache_data(ttl=300)
def load_suffix_denoising_trajectory(
    paths: tuple[str, ...],
    layer: int,
) -> dict | None:
    """Load group_masses across multiple frames (a trajectory) for one layer.

    Returns dict with:
      frame_indices  list[int]          — frames that had denoising data
      group_masses   float32(F, S, 4)   — F frames, S denoising steps, 4 groups
      n_steps        int
    """
    all_masses = []
    frame_indices = []
    n_steps = None

    for path in paths:
        d = load_suffix_denoising(path, layer)
        if d is None:
            continue
        all_masses.append(d["group_masses"])     # (n_steps, 4)
        frame_indices.append(path)
        if n_steps is None:
            n_steps = d["n_steps"]

    if not all_masses:
        return None

    return {
        "frame_paths":  frame_indices,
        "group_masses": np.stack(all_masses, axis=0),   # (F, n_steps, 4)
        "n_steps":      n_steps,
    }


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_meta(path: str) -> dict[str, Any]:
    """Load metadata only — fast, no attention data."""
    if not os.path.exists(path):
        return {}
    with h5py.File(path, "r") as f:
        meta = f.get("meta", {})
        result: dict[str, Any] = {}
        for k in ("prefix_len", "frame_idx", "seq_len", "n_real_tokens"):
            if k in meta:
                result[k] = int(meta[k][()])
        if "instruction" in meta:
            result["instruction"] = meta["instruction"][()].decode("utf-8") if isinstance(meta["instruction"][()], bytes) else str(meta["instruction"][()])
        if "token_texts" in meta:
            raw = meta["token_texts"][()]
            result["token_texts"] = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in raw]
        if "token_ids" in meta:
            result["token_ids"] = meta["token_ids"][()].tolist()
        return result


@st.cache_data(ttl=300)
def load_images(path: str) -> dict[str, np.ndarray | None]:
    """Load exterior and wrist camera images (224×224 uint8)."""
    result: dict[str, np.ndarray | None] = {"exterior": None, "wrist": None}
    if not os.path.exists(path):
        return result
    with h5py.File(path, "r") as f:
        imgs = f.get("images", {})
        if "exterior" in imgs:
            result["exterior"] = imgs["exterior"][()].astype(np.uint8)
        if "wrist" in imgs:
            result["wrist"] = imgs["wrist"][()].astype(np.uint8)
    return result


@st.cache_data(ttl=300)
def load_text_to_img(path: str, layer: int) -> np.ndarray | None:
    """Load text→image attention slice: (8, n_text, 512) float32."""
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        key = f"prefix/layer_{layer}/text_to_img"
        if key not in f:
            return None
        return f[key][()].astype(np.float32)


@st.cache_data(ttl=300)
def load_full_matrix(path: str, layer: int, head: int) -> np.ndarray | None:
    """Load full attention matrix for one head: (seq_len, seq_len) float32."""
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        key = f"prefix/layer_{layer}/full"
        if key not in f:
            return None
        return f[key][head].astype(np.float32)


@st.cache_data(ttl=300)
def load_full_matrix_all_heads(path: str, layer: int) -> np.ndarray | None:
    """Load full attention matrix for all heads: (8, seq_len, seq_len) float32."""
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        key = f"prefix/layer_{layer}/full"
        if key not in f:
            return None
        return f[key][()].astype(np.float32)


# ── Counterfactual helpers ─────────────────────────────────────────────────────

def prompt_to_slug(prompt: str, max_len: int = 24) -> str:
    """Convert a prompt string to a safe filename slug."""
    import re
    slug = prompt.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")[:max_len].rstrip("_")
    return slug or "prompt"


def h5_path_cf(
    checkpoint: str,
    episode: str,
    frame: int,
    prompt_slug: str,
    attn_h5_root: str = "attn_h5",
) -> str:
    """Path for a counterfactual rollout HDF5 file."""
    return str(Path(attn_h5_root) / checkpoint / episode / f"{frame:05d}_{prompt_slug}.h5")


def list_cf_slugs(
    checkpoint: str,
    episode: str,
    frame: int,
    attn_h5_root: str = "attn_h5",
) -> list[str]:
    """List all prompt slugs saved for a given (checkpoint, episode, frame)."""
    frame_dir = Path(attn_h5_root) / checkpoint / episode
    if not frame_dir.exists():
        return []
    prefix = f"{frame:05d}_"
    slugs = []
    for f in sorted(frame_dir.glob(f"{prefix}*.h5")):
        slug = f.stem[len(prefix):]
        if slug:
            slugs.append(slug)
    return slugs


# ── In-memory data from online inference ──────────────────────────────────────

def make_slice_dict_from_npy(
    layers_prefix_dir: str,
    token_texts: list[str] | None = None,
    instruction: str = "",
    ext_img: np.ndarray | None = None,
    wrist_img: np.ndarray | None = None,
) -> dict:
    """Build a slice dict (same schema as HDF5) from a directory of .npy files.

    Used by online inference mode to avoid disk I/O.
    """
    from PIL import Image as PILImage

    # Detect seq_len
    seq_len = None
    for i in range(NUM_LAYERS):
        p = os.path.join(layers_prefix_dir, f"attn_map_layer_{i}.npy")
        if os.path.exists(p):
            arr = np.load(p, mmap_mode="r")
            seq_len = arr.shape[-1]
            break
    if seq_len is None:
        return {}

    n_text = seq_len - TEXT_START_IDX
    if token_texts is None:
        token_texts = [f"tok_{i}" for i in range(n_text)]

    # Resize images to 224×224
    def to_224(img):
        if img is None:
            return None
        if img.shape[:2] != (224, 224):
            pil = PILImage.fromarray(img).resize((224, 224), PILImage.BILINEAR)
            img = np.array(pil, dtype=np.uint8)
        return img

    result: dict = {
        "meta": {
            "prefix_len": TEXT_START_IDX,
            "seq_len": seq_len,
            "instruction": instruction,
            "token_texts": token_texts[:n_text],
        },
        "images": {
            "exterior": to_224(ext_img),
            "wrist": to_224(wrist_img),
        },
        "prefix": {},
    }

    for i in range(NUM_LAYERS):
        p = os.path.join(layers_prefix_dir, f"attn_map_layer_{i}.npy")
        if not os.path.exists(p):
            continue
        attn = np.load(p)
        if attn.ndim == 4:
            attn = attn[0]  # (8, seq, seq)
        attn = attn.astype(np.float32)

        t2i = attn[:, TEXT_START_IDX:TEXT_START_IDX + n_text, :TOTAL_IMAGE_TOKENS]
        layer_data: dict = {"text_to_img": t2i}

        if i in {1, 4, 5, 7, 10}:
            layer_data["full"] = attn

        result["prefix"][f"layer_{i}"] = layer_data

    return result
