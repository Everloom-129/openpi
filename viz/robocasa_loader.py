"""Load RoboCasa episodes in LeRobot format and return DROID-compatible observation dicts.

The output dict uses the same keys as ``viz/pipeline.py:load_example()`` so that
the existing ``DroidInputs`` transform, ``infer_and_save()``, and the Streamlit
dashboard can consume robocasa data without modification.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

# ── LeRobot camera key mapping ────────────────────────────────────────────────

_CAMERA_KEYS = {
    "left": "observation.images.robot0_agentview_left",
    "right": "observation.images.robot0_agentview_right",
    "wrist": "observation.images.robot0_eye_in_hand",
}


# ── Frame extraction ─────────────────────────────────────────────────────────


def extract_frame(
    video_path: Path,
    frame_index: int,
    width: int = 256,
    height: int = 256,
    fps: int = 20,
) -> np.ndarray:
    """Extract a single RGB frame from an MP4 video using ffmpeg.

    Uses input-level seeking (``-ss`` before ``-i``) with a ``select`` filter
    for frame-exact extraction from H.264 streams.
    """
    seek_time = frame_index / fps
    cmd = [
        "ffmpeg",
        "-ss", f"{seek_time:.6f}",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_index})",
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    buf = result.stdout
    expected = width * height * 3
    if len(buf) != expected:
        raise ValueError(
            f"Expected {expected} bytes from {video_path} frame {frame_index}, got {len(buf)}"
        )
    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)


# ── Metadata helpers ─────────────────────────────────────────────────────────


def load_episode_metadata(lerobot_root: Path) -> list[dict]:
    """Parse ``meta/episodes.jsonl`` → list of episode dicts."""
    episodes_path = lerobot_root / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def get_episode_info(lerobot_root: Path, episode_index: int) -> dict:
    """Return the metadata dict for a single episode."""
    for ep in load_episode_metadata(lerobot_root):
        if ep["episode_index"] == episode_index:
            return ep
    raise ValueError(f"Episode {episode_index} not found in {lerobot_root / 'meta' / 'episodes.jsonl'}")


def get_dataset_info(lerobot_root: Path) -> dict:
    """Read ``meta/info.json`` and return the parsed dict."""
    with open(lerobot_root / "meta" / "info.json") as f:
        return json.load(f)


# ── Video path resolution ────────────────────────────────────────────────────


def _video_path(lerobot_root: Path, episode_index: int, camera_key: str) -> Path:
    """Resolve the MP4 path for a given episode and camera."""
    info = get_dataset_info(lerobot_root)
    template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = episode_index // chunks_size
    path = lerobot_root / template.format(
        episode_chunk=episode_chunk,
        video_key=camera_key,
        episode_index=episode_index,
    )
    return path


# ── State loading ─────────────────────────────────────────────────────────────


def _load_state_from_parquet(lerobot_root: Path, episode_index: int, frame_index: int) -> np.ndarray:
    """Read the 16-dim observation.state for a single frame from the episode parquet."""
    import pyarrow.parquet as pq

    info = get_dataset_info(lerobot_root)
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = episode_index // chunks_size
    parquet_path = lerobot_root / data_template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    table = pq.read_table(parquet_path, columns=["observation.state", "frame_index"])
    frame_indices = table.column("frame_index").to_pylist()
    row = frame_indices.index(frame_index)
    state = np.array(table.column("observation.state")[row].as_py(), dtype=np.float64)
    return state


def _map_state_to_droid(state_16: np.ndarray, mode: str = "ee") -> tuple[np.ndarray, np.ndarray]:
    """Map 16-dim RoboCasa state to (joint_position[7], gripper_position[1]).

    Modes:
        "ee":    Use ee_pos_rel[7:10] + ee_rot_rel[10:14] → 7-dim, gripper_qpos[14] → 1-dim.
        "zeros": Return zeros (avoids normalization artifacts with DROID checkpoint).
    """
    if mode == "zeros":
        return np.zeros(7, dtype=np.float64), np.zeros(1, dtype=np.float64)
    # mode == "ee"
    joint_position = state_16[7:14].astype(np.float64)   # ee_pos_rel(3) + ee_rot_rel(4)
    gripper_position = state_16[14:15].astype(np.float64)  # gripper_qpos[0]
    return joint_position, gripper_position


# ── Main loader ───────────────────────────────────────────────────────────────


def load_robocasa_example(
    lerobot_root: Path,
    episode_index: int,
    frame_index: int,
    ext_camera: str = "left",
    state_mode: str = "ee",
) -> dict:
    """Load one frame from a RoboCasa LeRobot-format dataset.

    Returns a dict with the same keys as ``viz/pipeline.py:load_example()``
    so it can be passed directly through ``DroidInputs`` → ``policy.infer()``.

    Args:
        lerobot_root: Path to the LeRobot dataset root (contains ``meta/``, ``videos/``, ``data/``).
        episode_index: Episode number (0-based).
        frame_index: Frame within the episode (0-based).
        ext_camera: Which exterior camera to use: ``"left"`` or ``"right"``.
        state_mode: ``"ee"`` maps end-effector state to 8-dim, ``"zeros"`` uses zeros.
    """
    lerobot_root = Path(lerobot_root)
    info = get_dataset_info(lerobot_root)
    fps = info.get("fps", 20)

    # Resolve video dimensions from info.json
    wrist_feat = info["features"].get("observation.images.robot0_eye_in_hand", {})
    vid_shape = wrist_feat.get("shape", [256, 256, 3])
    height, width = vid_shape[0], vid_shape[1]

    # Extract frames from MP4
    ext_camera_key = _CAMERA_KEYS[ext_camera]
    wrist_camera_key = _CAMERA_KEYS["wrist"]

    ext_video = _video_path(lerobot_root, episode_index, ext_camera_key)
    wrist_video = _video_path(lerobot_root, episode_index, wrist_camera_key)

    ext_img = extract_frame(ext_video, frame_index, width=width, height=height, fps=fps)
    wrist_img = extract_frame(wrist_video, frame_index, width=width, height=height, fps=fps)

    # Load instruction from episodes.jsonl
    ep_info = get_episode_info(lerobot_root, episode_index)
    tasks = ep_info.get("tasks", [])
    instruction = tasks[0] if tasks else ""

    # Load state from parquet (optional — fallback to zeros)
    try:
        state_16 = _load_state_from_parquet(lerobot_root, episode_index, frame_index)
        joint_position, gripper_position = _map_state_to_droid(state_16, mode=state_mode)
    except Exception:
        joint_position, gripper_position = _map_state_to_droid(np.zeros(16), mode="zeros")

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": instruction,
        "gt_action": None,
    }
