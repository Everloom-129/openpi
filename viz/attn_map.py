"""Shared utilities for the attention visualization pipelines.

Historically this module also held a JAX-era visualization layer (``visualize_attention``,
``visualize_heads``, ``visualize_tokenizer``, ``vis_example``) and a
``process_episode`` driver that read ``attn/{device_id}/layers_prefix/*.npy``
files dropped by the model. All of that has been deleted (2026-04-28) — the
RAM-buffer + HDF5 pipeline in ``viz/pipeline.py`` replaced it long ago. What
remains here are only the helpers still imported by the active pipelines and
the dashboard:

    - load_duck_example  : single-frame DROID-format example loader
    - get_keyframes      : open-loop keyframe stride
    - infer_config_name  : checkpoint-dir → openpi config name
    - get_policy         : create a trained policy from a checkpoint dir
    - select_best_gpu    : pick the CUDA device with most free memory
"""

import os

import numpy as np
import torch
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def load_duck_example(camera: str = "left", index: int = 0):
    """Load a single example from the local duck dataset (data/example/duck)."""

    if camera == "left":
        camera = "varied_camera_1"
    elif camera == "right":
        camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    if not 0 <= index <= 90:
        raise ValueError("index must be between 0 and 90")

    _attn_map_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_attn_map_dir, ".."))
    data_dir = os.path.join(_project_root, "data/example/duck/frames")
    traj_path = os.path.join(_project_root, "data/example/duck/trajectory.h5")

    ext_path = os.path.join(data_dir, camera, f"{index:05d}.jpg")
    hand_path = os.path.join(data_dir, "hand_camera", f"{index:05d}.jpg")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    import h5py
    with h5py.File(traj_path, "r") as f:
        joint_position = f["observation/robot_state/joint_positions"][index].astype(np.float64)
        gripper_position = f["observation/robot_state/gripper_position"][index : index + 1].astype(np.float64)

    instruction = "place the duck toy into the pink bowl"
    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": instruction,
    }


def get_keyframes(total_frames: int, horizon: int) -> list[int]:
    """Indices where open-loop inference actually happens (every `horizon` frames)."""
    return list(range(0, total_frames, horizon))


def infer_config_name(checkpoint_dir: str) -> str:
    """Infer the openpi training-config name from a checkpoint directory name.

    Resolution order:
      1. Strip a trailing ``_pytorch`` suffix (added by the conversion scripts).
      2. Try the result against the registered configs.
      3. If unknown, drop trailing ``_<word>`` segments and retry.
      4. Fall back to ``pi05_droid`` / ``pi0_droid`` based on the family hint.
    """
    raw = os.path.basename(os.path.normpath(checkpoint_dir))
    candidate = raw[: -len("_pytorch")] if raw.endswith("_pytorch") else raw

    while candidate:
        try:
            _config.get_config(candidate)
            return candidate
        except (ValueError, KeyError):
            pass
        idx = candidate.rfind("_")
        if idx == -1:
            break
        candidate = candidate[:idx]

    return "pi05_droid" if "pi05" in raw.lower() else "pi0_droid"


def get_policy(checkpoint_dir: str, device: str = "cuda:0", config_name: str | None = None):
    if config_name is None:
        config_name = infer_config_name(checkpoint_dir)
    print(f"[policy] Using config '{config_name}' for checkpoint '{checkpoint_dir}'")
    config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)


def select_best_gpu() -> str:
    """Pick the CUDA device with the most free memory (or 'cpu' if no GPU)."""
    if not torch.cuda.is_available():
        return "cpu"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return "cuda:0"

    max_free_memory = 0
    best_gpu = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        free_memory = torch.cuda.mem_get_info()[0]
        print(f"GPU {i}: {free_memory / 1e9:.2f} GB free")
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i

    selected_device = f"cuda:{best_gpu}"
    print(f"Auto-selected {selected_device} with {max_free_memory / 1e9:.2f} GB free")
    return selected_device
