"""Closed-loop GR00T-N1.7 inference inside a robosuite/robocasa MuJoCo sim.

Companion to `run_policy_sim.py` (which targets pi0.5-DROID over websocket).
Differences:

* Server: GR00T `gr00t/eval/run_gr00t_server.py` on ZMQ REP, default port 5555.
  Launch via `viz_sim/run_gr00t_server.sh` in the Isaac-GR00T uv venv.
* Action space: GR00T DROID embodiment outputs
    - `joint_position`: (T, 7) RELATIVE deltas (added to current qpos each step)
    - `gripper_position`: (T, 1) ABSOLUTE in [0, 1]
    - `eef_9d`: ignored here (we drive the arm via joint targets)
  We use robosuite's JOINT_POSITION controller and apply
  `target_qpos = current_qpos + Δ`, mapping the gripper to the GRIP signal.
* Obs format: nested dict with `video`, `state`, `language` matching
  `oxe_droid_relative_eef_relative_joint` (see
  third_party/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:28).
  Images are 180×320, padded with `resize_with_pad` (matches main_gr00t.py).
* video.delta_indices = [-15, 0] → we keep a 16-frame ring buffer and send
  (frame[t-15], frame[t]) stacked as (B=1, T=2, H, W, 3).

Run (after `bash viz_sim/run_gr00t_server.sh` is up in the GR00T env):
    conda activate robocasa_sim
    python viz_sim/run_policy_sim_gr00t.py --task Lift --prompt "pick up the cube"
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import robosuite
from PIL import Image
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation

# gr00t_client lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gr00t_client import PolicyClient

# GR00T DROID image resolution (H, W) — see examples/DROID/main_gr00t.py:49.
GR00T_RES_H = 180
GR00T_RES_W = 320

# Video temporal context. The static registry for OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
# uses delta_indices=[-15, 0] (T=2), but each fine-tuned checkpoint can override this
# via its experiment.json. We query the server at startup to get the actual deltas.
DEFAULT_VIDEO_DELTA = [0]

# Egocentric frame correction matching the OXE DROID training pipeline.
# Copied from examples/DROID/main_gr00t.py:53.
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)

# Display layout (same as run_policy_sim.py).
SIM_VIEW_SIZE = 448
TILE_SIZE = 224
PROMPT_STRIP_HEIGHT = 36


def build_env(task: str):
    """Robosuite Panda env with JOINT_POSITION control (GR00T outputs joint targets)."""
    controller_cfg = load_composite_controller_config(robot="Panda")
    # Pass raw qpos targets through unscaled: matching input/output ranges make
    # the controller's affine remap an identity, so action[i] is interpreted
    # directly as the target joint angle in radians.
    controller_cfg["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_max": 3.14,
        "input_min": -3.14,
        "output_max": 3.14,
        "output_min": -3.14,
        "kp": 50,
        "damping_ratio": 1,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }

    return robosuite.make(
        env_name=task,
        robots="Panda",
        controller_configs=controller_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["frontview", "agentview", "robot0_eye_in_hand"],
        camera_heights=[SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE],
        camera_widths=[SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE],
        ignore_done=True,
        control_freq=20,
    )


def _normalize_robosuite_image(img) -> np.ndarray:
    """Robosuite returns float[0,1] images flipped vertically. Return uint8 RGB."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[::-1])


def _resize_with_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """tf.image.resize_with_pad equivalent (matches examples/DROID/utils.py)."""
    if img.shape[:2] == (h, w):
        return img
    pil = Image.fromarray(img)
    cw, ch = pil.size
    ratio = max(cw / w, ch / h)
    rh, rw = int(ch / ratio), int(cw / ratio)
    pil_resized = pil.resize((rw, rh), resample=Image.BILINEAR)
    canvas = Image.new(pil_resized.mode, (w, h), 0)
    canvas.paste(pil_resized, (max(0, (w - rw) // 2), max(0, (h - rh) // 2)))
    return np.asarray(canvas)


def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a robosuite (xyzw) quaternion to GR00T's 6D rotation rep.

    Mirrors `compute_eef_9d` in examples/DROID/main_gr00t.py:59 — applies the
    DROID egocentric correction and takes the top two rows of the matrix.
    """
    rot_robot = Rotation.from_quat(quat_xyzw).as_matrix()
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    return rot_mat[:2, :].reshape(6)


def _make_eef_9d(env_obs: dict) -> np.ndarray:
    pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    rot6d = _quat_xyzw_to_rot6d(np.asarray(env_obs["robot0_eef_quat"], dtype=np.float64))
    return np.concatenate([pos, rot6d]).astype(np.float32)


def _gripper_position_scalar(env_obs: dict) -> np.ndarray:
    """Map robosuite 2-finger qpos to a 1-D gripper position in [0, 1] (1=closed).

    Robosuite's Panda gripper qpos is ~[0.04, -0.04] open and [0, 0] closed,
    so |q[0]-q[1]| is the finger separation. We invert and clip to roughly
    match the DROID gripper_position convention (1=closed, 0=open).
    """
    q = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)
    sep = float(abs(q[0] - q[1]))
    # Panda max separation ≈ 0.08; closed ≈ 0.0.
    closed = 1.0 - np.clip(sep / 0.08, 0.0, 1.0)
    return np.array([closed], dtype=np.float32)


def make_gr00t_obs(frame_buf: deque, video_deltas: list[int], env_obs: dict,
                   instruction: str) -> dict:
    """Build the nested obs dict the GR00T DROID embodiment expects.

    `video_deltas` is the model's `video.delta_indices` (e.g. [0] or [-15, 0]).
    We sample `frame_buf[delta]` for each entry; the buffer is sized so that
    index -1 is "now" and earlier negative indices reach into history.
    """
    def _sample(key: str) -> np.ndarray:
        # frame_buf[-1] is "now". delta=0 → -1, delta=-15 → -16. Clamp to oldest
        # frame on startup before history has filled.
        n = len(frame_buf)
        frames = []
        for d in video_deltas:
            idx = -1 + min(d, 0)         # negative or zero
            idx = max(idx, -n)           # clamp to oldest available
            frames.append(frame_buf[idx][key])
        return np.stack(frames)[None, ...]  # (1, T, H, W, 3)

    ext_stack = _sample("ext")
    wrist_stack = _sample("wrist")

    state = {
        "eef_9d": _make_eef_9d(env_obs)[None, None, ...],
        "gripper_position": _gripper_position_scalar(env_obs)[None, None, ...],
        "joint_position": np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32)[None, None, ...],
    }

    return {
        "video": {
            "exterior_image_1_left": ext_stack,
            "wrist_image_left": wrist_stack,
        },
        "state": state,
        "language": {
            "annotation.language.language_instruction": [[instruction]],
        },
    }


# ---------- canvas helpers (lifted from run_policy_sim.py) ----------

def _label(tile: np.ndarray, text: str) -> np.ndarray:
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (max(80, 8 * len(text)), 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _prompt_strip(width: int, text: str) -> np.ndarray:
    strip = np.full((PROMPT_STRIP_HEIGHT, width, 3), 20, dtype=np.uint8)
    if not text:
        return strip
    max_chars = max(10, width // 11)
    shown = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    (tw, th), _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(strip, shown, ((width - tw) // 2, (PROMPT_STRIP_HEIGHT + th) // 2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return strip


def _to_tile(img: np.ndarray, size: int = TILE_SIZE) -> np.ndarray:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def compose_canvas(sim: np.ndarray, ext: np.ndarray, wrist: np.ndarray,
                   prompt: str = "") -> np.ndarray:
    sim = _label(sim, "sim (frontview)")
    pending = np.full((TILE_SIZE, TILE_SIZE, 3), 40, dtype=np.uint8)
    cv2.putText(pending, "no attn (gr00t)", (24, TILE_SIZE // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    right_top = np.hstack([_label(_to_tile(ext), "ext"), _label(pending, "—")])
    right_bot = np.hstack([_label(_to_tile(wrist), "wrist"), _label(pending, "—")])
    right = np.vstack([right_top, right_bot])
    grid = np.hstack([sim, right])
    return np.vstack([grid, _prompt_strip(grid.shape[1], prompt)])


# ---------- main loop ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift", help="robosuite env name")
    ap.add_argument("--prompt", default="pick up the red cube")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=8,
                    help="actions to execute from each chunk before re-querying "
                         "(GR00T returns 40-step chunks; 8 keeps loop reactive)")
    ap.add_argument("--gripper-thresh", type=float, default=0.5,
                    help="binarize gripper>thresh to closed (GRIP=+1), else open (-1)")
    args = ap.parse_args()

    print(f"Connecting to GR00T policy at tcp://{args.host}:{args.port} ...")
    policy = PolicyClient(host=args.host, port=args.port)
    print("Ping:", policy.ping())

    # Fetch the actual modality spec from the server — different checkpoints
    # can override video horizon (e.g. N1.7-DROID uses T=1, not the registry's T=2).
    modality_cfg = policy.get_modality_config()
    video_deltas = list(modality_cfg["video"]["delta_indices"]) or DEFAULT_VIDEO_DELTA
    hist_span = max(-min(video_deltas), 0) + 1   # how far back we need to remember
    print(f"Server video.delta_indices={video_deltas} → buffer length {hist_span}")

    env = build_env(args.task)
    print(f"Env action_dim={env.action_dim}  (expecting 8: 7 qpos targets + 1 gripper)")
    obs = env.reset()

    # Seed the video history buffer with the first frame so we have something
    # to send for the t-15 slot until 15 real steps have elapsed.
    def _resize_pair(env_obs):
        ext_full = _normalize_robosuite_image(env_obs["agentview_image"])
        wrist_full = _normalize_robosuite_image(env_obs["robot0_eye_in_hand_image"])
        return {
            "ext": _resize_with_pad(ext_full, GR00T_RES_H, GR00T_RES_W),
            "wrist": _resize_with_pad(wrist_full, GR00T_RES_H, GR00T_RES_W),
        }

    frame_buf: deque = deque(maxlen=hist_span)
    initial = _resize_pair(obs)
    for _ in range(hist_span):
        frame_buf.append(initial)

    chunk: np.ndarray | None = None  # shape (T, 8): 7 qpos targets + 1 gripper
    t0 = time.time()
    n_infer = 0

    for step in range(args.steps):
        frame_buf.append(_resize_pair(obs))

        if chunk is None or step % args.horizon == 0:
            request_obs = make_gr00t_obs(frame_buf, video_deltas, obs, args.prompt)
            action_dict, _info = policy.get_action(request_obs)
            # Server returns each key as (T, D) (no batch dim on the way back).
            jp_rel = np.asarray(action_dict["joint_position"], dtype=np.float32)  # (T, 7) RELATIVE
            grip = np.asarray(action_dict["gripper_position"], dtype=np.float32)  # (T, 1) ABS [0,1]
            if jp_rel.ndim == 3:  # defensive: in case server keeps a batch dim
                jp_rel = jp_rel[0]
                grip = grip[0]
            # Convert relative joint deltas to absolute targets using the qpos
            # at the moment of inference. (Could be re-anchored each step, but
            # the open-loop chunking matches main_gr00t.py.)
            base_qpos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
            chunk_qpos = base_qpos[None, :] + jp_rel              # (T, 7)
            chunk = np.concatenate([chunk_qpos, grip], axis=1)    # (T, 8)
            n_infer += 1

        action = chunk[step % args.horizon].copy()
        # Robosuite GRIP expects [-1, 1], +1 = close. Binarize to match main_gr00t.
        action[-1] = 1.0 if action[-1] > args.gripper_thresh else -1.0
        if action.shape[0] != env.action_dim:
            a = np.zeros(env.action_dim, dtype=np.float32)
            a[: min(len(action), env.action_dim)] = action[: env.action_dim]
            action = a

        obs, _reward, _done, _info = env.step(action)

        sim_view = _normalize_robosuite_image(obs["frontview_image"])
        sim_view = cv2.resize(sim_view, (SIM_VIEW_SIZE, SIM_VIEW_SIZE), interpolation=cv2.INTER_AREA)
        ext = _normalize_robosuite_image(obs["agentview_image"])
        wrist = _normalize_robosuite_image(obs["robot0_eye_in_hand_image"])

        canvas = compose_canvas(sim_view, ext, wrist, prompt=args.prompt)
        cv2.imshow("openpi sim (gr00t)", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    dt = time.time() - t0
    print(f"Done. {args.steps} sim steps, {n_infer} policy queries in {dt:.1f}s.")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
