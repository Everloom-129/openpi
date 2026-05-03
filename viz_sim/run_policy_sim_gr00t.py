"""Closed-loop GR00T-N1.7-DROID inference inside a robosuite/robocasa MuJoCo sim.

Companion to `run_pi0_policy_sim.py` (pi0/pi0.5 over websocket); this script
targets the GR00T DROID checkpoint over ZMQ.

Architecture
------------
* This script runs in the `robocasa_sim` conda env.
* The GR00T policy runs in the Isaac-GR00T uv venv and is launched by
  `viz_sim/run_gr00t_server.sh` as a ZMQ REP server on localhost:5555.
* Robot can be either:
    - `Panda`        — fixed-base, the natural training distribution for
                       GR00T-N1.7-DROID (DROID dataset is all Panda)
    - `PandaOmron`   — mobile manipulator used by robocasa365 tasks. GR00T
                       has no base/torso head, so torso + base + control_mode
                       are pinned to 0 / -1 (i.e. always `--block_base`).
* Action space: GR00T DROID embodiment outputs
    - `joint_position`  (T, 7) ABSOLUTE joint targets in radians (despite
                        the embodiment tag containing "RELATIVE_JOINT").
                        Verified against the model's bundled statistics.json:
                        `action/joint_position/{min,max,mean}` covers Panda's
                        joint limits with mean ≈ home pose, not small deltas.
                        Upstream `main_gr00t.py` sends these straight to
                        `RobotEnv(action_space="joint_position")` which also
                        treats them as absolute targets.
    - `gripper_position`(T, 1) ABSOLUTE in [0, 1] (1=closed)
    - `eef_9d`          ignored — we drive the arm via joint targets
  We use a JOINT_POSITION (identity-passthrough) arm controller and send
  `target = action[:7]` directly — no `current_qpos + Δ` anchoring. (Earlier
  versions misread "RELATIVE" in the tag name and added the action to the
  current qpos, commanding ~6-rad targets that exploded the robot.)
* Obs: nested {video, state, language} matching
  `oxe_droid_relative_eef_relative_joint`. Cameras letterboxed to 180×320.
* control_freq = 15 Hz to match `DROID_CONTROL_FREQUENCY` in upstream
  `examples/DROID/main_gr00t.py`. The previous 20 Hz made each chunked
  delta land 33% too aggressively.

Run (after `bash viz_sim/run_gr00t_server.sh` is up):
    conda activate robocasa_sim
    python viz_sim/run_policy_sim_gr00t.py --task Lift --robot Panda \
        --prompt "pick up the cube"
    # or on a robocasa mobile-manipulator task:
    python viz_sim/run_policy_sim_gr00t.py --task PnPCounterToCab \
        --robot PandaOmron --prompt "pick up the can and place it in the cabinet"
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import robosuite
from PIL import Image
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# gr00t_client lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gr00t_client import PolicyClient  # noqa: E402

# GR00T DROID image resolution (H, W) — see examples/DROID/main_gr00t.py:49.
GR00T_RES_H = 180
GR00T_RES_W = 320

# Default video.delta_indices if the server doesn't return any.
DEFAULT_VIDEO_DELTA = [0]

# Egocentric frame correction matching the OXE DROID training pipeline.
# Copied from examples/DROID/main_gr00t.py:53.
DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)

# DROID training control frequency (upstream main_gr00t.py).
DROID_CONTROL_FREQ = 15

# Display layout (same as run_pi0_policy_sim.py).
SIM_VIEW_SIZE = 448
TILE_SIZE = 224
PROMPT_STRIP_HEIGHT = 36


# ---------- env ----------

def _droid_arm_cfg() -> dict:
    """JOINT_POSITION absolute-target controller for GR00T-DROID.

    GR00T's `action/joint_position` is ABSOLUTE joint targets in radians
    (verified against statistics.json — mean ≈ Panda home pose, range =
    Panda joint limits). Robosuite's JointPositionController defaults to
    `input_type="delta"`, which interprets the action as a delta added
    to current qpos — that mismatch is what makes the arm "rotate
    forever like velocity control": a ±3-rad absolute target gets
    treated as a ±3-rad delta per control tick, the controller's rate
    limit saturates, and the joint integrates indefinitely.

    Setting `input_type="absolute"` makes set_goal() do
    `self.goal_qpos = action` directly (joint_pos.py:227-228), bypassing
    `scale_action`. With this, input/output ranges are irrelevant — we
    keep them at ±3.14 just for documentation. Note: absolute mode
    requires `impedance_mode="fixed"` (the default), see joint_pos.py:172.
    """
    return {
        "type": "JOINT_POSITION",
        "input_type": "absolute",   # ← THE fix: absolute targets, not deltas
        "input_max": 3.14,
        "input_min": -3.14,
        "output_max": 3.14,
        "output_min": -3.14,
        "kp": 50,
        "damping_ratio": 1,
        "impedance_mode": "fixed",  # required by absolute mode
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }


def build_env(task: str, robot: str):
    """Robosuite env with JOINT_POSITION arm control for GR00T-DROID.

    For PandaOmron the composite controller exposes torso + base body parts;
    we override only the arm. The model has no base head, so we always pin
    those to 0 in the action loop — `--block_base` is implicit.
    """
    controller_cfg = load_composite_controller_config(robot=robot)
    controller_cfg["body_parts"]["right"] = _droid_arm_cfg()

    cam_names = ["frontview", "agentview", "robot0_eye_in_hand"]
    cam_heights = [SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE]
    cam_widths = [SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE]

    def _make(extra_cams: list[str]):
        names = cam_names + extra_cams
        heights = cam_heights + [TILE_SIZE] * len(extra_cams)
        widths = cam_widths + [TILE_SIZE] * len(extra_cams)
        return robosuite.make(
            env_name=task,
            robots=robot,
            controller_configs=controller_cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=names,
            camera_heights=heights,
            camera_widths=widths,
            ignore_done=True,
            control_freq=DROID_CONTROL_FREQ,
        )

    if robot == "PandaOmron":
        try:
            return _make(["robot0_agentview_left"])
        except ValueError as e:
            if "robot0_agentview_left" in str(e):
                print(f"[build_env] task '{task}' has no robot0_agentview_left "
                      "camera; falling back to agentview")
            else:
                raise
    return _make([])


# ---------- image / obs helpers ----------

def _normalize_robosuite_image(img) -> np.ndarray:
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
    """Mirrors `compute_eef_9d` in examples/DROID/main_gr00t.py:59."""
    rot_robot = Rotation.from_quat(quat_xyzw).as_matrix()
    rot_mat = rot_robot @ DROID_EEF_ROTATION_CORRECT
    return rot_mat[:2, :].reshape(6)


def _make_eef_9d(env_obs: dict) -> np.ndarray:
    pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    rot6d = _quat_xyzw_to_rot6d(np.asarray(env_obs["robot0_eef_quat"], dtype=np.float64))
    return np.concatenate([pos, rot6d]).astype(np.float32)


def _gripper_position_scalar(env_obs: dict) -> np.ndarray:
    """Robosuite 2-finger qpos → 1-D position in [0,1] (1=closed)."""
    q = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)
    sep = float(abs(q[0] - q[1]))
    closed = 1.0 - np.clip(sep / 0.08, 0.0, 1.0)
    return np.array([closed], dtype=np.float32)


def _ext_image_key(env_obs: dict) -> str:
    """Prefer the robocasa robot-mounted side camera if present."""
    return ("robot0_agentview_left_image"
            if "robot0_agentview_left_image" in env_obs
            else "agentview_image")


def make_gr00t_obs(frame_buf: deque, video_deltas: list[int],
                   env_obs: dict, instruction: str) -> dict:
    def _sample(key: str) -> np.ndarray:
        n = len(frame_buf)
        frames = []
        for d in video_deltas:
            idx = -1 + min(d, 0)
            idx = max(idx, -n)
            frames.append(frame_buf[idx][key])
        return np.stack(frames)[None, ...]  # (1, T, H, W, 3)

    state = {
        "eef_9d": _make_eef_9d(env_obs)[None, None, ...],
        "gripper_position": _gripper_position_scalar(env_obs)[None, None, ...],
        "joint_position": np.asarray(env_obs["robot0_joint_pos"],
                                     dtype=np.float32)[None, None, ...],
    }
    return {
        "video": {
            "exterior_image_1_left": _sample("ext"),
            "wrist_image_left": _sample("wrist"),
        },
        "state": state,
        "language": {
            "annotation.language.language_instruction": [[instruction]],
        },
    }


# ---------- canvas helpers (lifted from run_pi0_policy_sim.py) ----------

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


def _placeholder(text: str = "no attn (gr00t)", size: int = TILE_SIZE) -> np.ndarray:
    tile = np.full((size, size, 3), 40, dtype=np.uint8)
    cv2.putText(tile, text, (24, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return tile


# ---------- attention overlay helpers ----------

def _agg_heads(layer_attn: np.ndarray, mode: str) -> np.ndarray:
    """layer_attn: (H, seq, seq) → (seq, seq) reduced over heads."""
    if mode == "max":
        return layer_attn.max(axis=0)
    if mode == "min":
        return layer_attn.min(axis=0)
    return layer_attn.mean(axis=0)


def _attn_overlay_for_image(img_uint8: np.ndarray, attn_1d: np.ndarray,
                            grid_hw: tuple[int, int] | None,
                            *, alpha: float = 0.45,
                            size: int = TILE_SIZE) -> np.ndarray:
    """attn_1d: (n_tokens,) per-image-token attention.

    grid_hw: (H_patches, W_patches) if known (from image_grid_thw); else None
    and we fall back to ⌈√n⌉×⌈n/⌈√n⌉⌉ which matches square images.
    """
    n = int(attn_1d.shape[0])
    if grid_hw is not None and grid_hw[0] * grid_hw[1] == n:
        h_p, w_p = grid_hw
    else:
        side = int(np.ceil(np.sqrt(max(n, 1))))
        h_p, w_p = side, int(np.ceil(n / side))
    grid = np.full((h_p * w_p,), float(attn_1d.min()), dtype=np.float32)
    grid[:n] = attn_1d.astype(np.float32)
    grid = grid.reshape(h_p, w_p)
    up = cv2.resize(grid, (size, size), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(up.min()), float(up.max())
    norm = (up - lo) / (hi - lo + 1e-8)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    base = img_uint8 if img_uint8.shape[:2] == (size, size) else cv2.resize(
        img_uint8, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)


def parse_attn_payload(info: dict | None) -> dict:
    """Pull info["attn"] into normalized shape: int layer keys + ndarrays.

    Returns {"vlm": {layer_idx: (H, seq, seq)}, "image_mask": (seq,) bool,
              "text_mask": (seq,) bool, "image_grid_thw": ndarray|None}
    Empty dict when payload is missing or malformed.
    """
    if not isinstance(info, dict):
        return {}
    attn = info.get("attn") or {}
    vlm_raw = attn.get("vlm") or {}
    if not vlm_raw:
        return {}
    vlm: dict[int, np.ndarray] = {}
    for k, v in vlm_raw.items():
        try:
            arr = np.asarray(v, dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 4:    # (B, H, seq, seq) → (H, seq, seq)
            arr = arr[0]
        vlm[int(k)] = arr
    if not vlm:
        return {}

    meta = attn.get("vlm_meta") or {}
    image_mask = np.asarray(meta.get("image_mask"), dtype=bool) if "image_mask" in meta else None
    text_mask = None
    if image_mask is not None and image_mask.ndim == 2:
        image_mask = image_mask[0]
    if "attention_mask" in meta:
        am = np.asarray(meta["attention_mask"], dtype=bool)
        if am.ndim == 2:
            am = am[0]
        text_mask = am & ~image_mask if image_mask is not None else am

    grid_thw = meta.get("image_grid_thw")
    if grid_thw is not None:
        grid_thw = np.asarray(grid_thw)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.reshape(1, -1)

    return {
        "vlm":            vlm,
        "image_mask":     image_mask,
        "text_mask":      text_mask,
        "image_grid_thw": grid_thw,
    }


def render_attn_tiles(parsed: dict, ext_img: np.ndarray, wrist_img: np.ndarray,
                      layer_idx: int, head_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """From a parsed attn payload → ext_attn tile + wrist_attn tile."""
    if not parsed:
        return _placeholder("no attn (gr00t)"), _placeholder("no attn (gr00t)")
    vlm = parsed["vlm"]
    if not vlm:
        return _placeholder("no attn (gr00t)"), _placeholder("no attn (gr00t)")

    image_mask = parsed.get("image_mask")
    text_mask = parsed.get("text_mask")
    if image_mask is None or text_mask is None:
        return _placeholder("missing mask"), _placeholder("missing mask")

    layer_keys = sorted(vlm.keys())
    layer_idx = max(0, min(layer_idx, len(layer_keys) - 1))
    layer_attn = vlm[layer_keys[layer_idx]]                # (H, seq, seq)
    attn_2d = _agg_heads(layer_attn, head_mode)            # (seq, seq)

    img_idx = np.where(image_mask)[0]
    txt_idx = np.where(text_mask)[0]
    if img_idx.size == 0 or txt_idx.size == 0:
        return _placeholder("empty t/i"), _placeholder("empty t/i")

    # text rows × image cols → mean over text rows = (n_image,)
    per_img = attn_2d[np.ix_(txt_idx, img_idx)].mean(axis=0)
    n_image = per_img.shape[0]

    # Split per-image attention into ext + wrist using image_grid_thw if
    # available; otherwise assume 50/50 in the order we sent (ext first).
    grid_thw = parsed.get("image_grid_thw")
    if grid_thw is not None and grid_thw.shape[0] >= 2:
        # Each row: (T, H, W). Token count per image: T * H * W (post-merge).
        counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).astype(int)
        if counts.sum() == n_image:
            ext_count = int(counts[0])
            ext_attn = per_img[:ext_count]
            wrist_attn = per_img[ext_count : ext_count + int(counts[1])]
            ext_grid = (int(grid_thw[0, 1]), int(grid_thw[0, 2]))
            wrist_grid = (int(grid_thw[1, 1]), int(grid_thw[1, 2]))
        else:
            ext_attn, wrist_attn = per_img[: n_image // 2], per_img[n_image // 2:]
            ext_grid = wrist_grid = None
    else:
        ext_attn, wrist_attn = per_img[: n_image // 2], per_img[n_image // 2:]
        ext_grid = wrist_grid = None

    ext_tile = _attn_overlay_for_image(ext_img, ext_attn, ext_grid)
    wrist_tile = _attn_overlay_for_image(wrist_img, wrist_attn, wrist_grid)
    return ext_tile, wrist_tile


def compose_canvas(sim: np.ndarray, ext: np.ndarray, wrist: np.ndarray,
                   ext_attn: np.ndarray | None = None,
                   wrist_attn: np.ndarray | None = None,
                   prompt: str = "") -> np.ndarray:
    sim = _label(sim, "sim (frontview)")
    ext_attn_tile = ext_attn if ext_attn is not None else _placeholder()
    wrist_attn_tile = wrist_attn if wrist_attn is not None else _placeholder()
    right_top = np.hstack([_label(_to_tile(ext), "ext"),
                           _label(_to_tile(ext_attn_tile), "ext attn")])
    right_bot = np.hstack([_label(_to_tile(wrist), "wrist"),
                           _label(_to_tile(wrist_attn_tile), "wrist attn")])
    right = np.vstack([right_top, right_bot])
    grid = np.hstack([sim, right])
    return np.vstack([grid, _prompt_strip(grid.shape[1], prompt)])


# ---------- main loop ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift", help="robosuite env name")
    ap.add_argument("--robot", default="Panda", choices=["Panda", "PandaOmron"],
                    help="Panda for DROID-style tasks; PandaOmron for robocasa "
                         "mobile-manipulator tasks (base/torso are pinned to 0 "
                         "since GR00T-DROID has no base head)")
    ap.add_argument("--prompt", default="pick up the red cube")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=8,
                    help="actions to execute from each chunk before re-querying")
    ap.add_argument("--gripper-thresh", type=float, default=0.5,
                    help="binarize gripper>thresh to closed (GRIP=+1), else open (-1)")
    ap.add_argument("--diag_dir", default="viz_sim/results/action_diag",
                    help="where to save the per-dim action diagnostic on exit")
    ap.add_argument("--no_diag", action="store_true",
                    help="skip writing the action diagnostic on exit")
    args = ap.parse_args()

    print(f"Connecting to GR00T policy at tcp://{args.host}:{args.port} ...")
    policy = PolicyClient(host=args.host, port=args.port)
    print("Ping:", policy.ping())

    modality_cfg = policy.get_modality_config()
    video_deltas = list(modality_cfg["video"]["delta_indices"]) or DEFAULT_VIDEO_DELTA
    hist_span = max(-min(video_deltas), 0) + 1
    print(f"Server video.delta_indices={video_deltas} → buffer length {hist_span}")

    env = build_env(args.task, args.robot)
    print(f"Env action_dim={env.action_dim}  (robot={args.robot})")
    if args.robot == "PandaOmron":
        try:
            split = env.robots[0].composite_controller._action_split_indexes
            print("[robocasa] composite action layout:",
                  ", ".join(f"{k}=[{v[0]}:{v[1]}]" for k, v in split.items()))
        except Exception as e:
            print("[robocasa] could not inspect composite action layout:", e)
        print("[robocasa] GR00T-DROID has no base/torso head → forcing torso=0, "
              "base=0, control_mode=-1 (arm-only)")

    obs = env.reset()

    def _resize_pair(env_obs):
        ext_full = _normalize_robosuite_image(env_obs[_ext_image_key(env_obs)])
        wrist_full = _normalize_robosuite_image(env_obs["robot0_eye_in_hand_image"])
        return {
            "ext": _resize_with_pad(ext_full, GR00T_RES_H, GR00T_RES_W),
            "wrist": _resize_with_pad(wrist_full, GR00T_RES_H, GR00T_RES_W),
        }

    frame_buf: deque = deque(maxlen=hist_span)
    initial = _resize_pair(obs)
    for _ in range(hist_span):
        frame_buf.append(initial)

    # `chunk` stores (T, 8) = [Δjoint(7), gripper_abs(1)]. The Δjoint stays
    # delta — we re-anchor at current qpos every step in the loop.
    chunk: np.ndarray | None = None
    t0 = time.time()
    n_infer = 0
    recorded_actions: list[np.ndarray] = []  # raw model output (Δq, gripper_abs)
    interrupted = False

    def _on_sigint(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _on_sigint)

    pbar = tqdm(range(args.steps), desc="sim", dynamic_ncols=True)

    WINDOW = "openpi sim (gr00t)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    # Live attention controls (mirrors run_pi0_policy_sim.py).
    head_modes = ["mean", "max", "min"]
    state = {"layer": 0, "head_mode": 0, "n_layers": 1}
    cv2.createTrackbar("layer", WINDOW, 0, max(1, state["n_layers"] - 1),
                       lambda v: state.update(layer=v))
    cv2.createTrackbar("head: 0=mean 1=max 2=min", WINDOW, 0, len(head_modes) - 1,
                       lambda v: state.update(head_mode=v))

    last_attn_payload: dict = {}  # parse_attn_payload result

    for step in pbar:
        frame_buf.append(_resize_pair(obs))

        if chunk is None or step % args.horizon == 0:
            request_obs = make_gr00t_obs(frame_buf, video_deltas, obs, args.prompt)
            action_dict, info = policy.get_action(request_obs)
            jp_rel = np.asarray(action_dict["joint_position"], dtype=np.float32)  # (T,7) Δ
            grip = np.asarray(action_dict["gripper_position"], dtype=np.float32)  # (T,1) abs
            if jp_rel.ndim == 3:
                jp_rel = jp_rel[0]
                grip = grip[0]
            chunk = np.concatenate([jp_rel, grip], axis=1)  # (T, 8)
            n_infer += 1

            # Refresh attention payload + retune layer trackbar to actual L.
            last_attn_payload = parse_attn_payload(info)
            if last_attn_payload.get("vlm"):
                L = len(last_attn_payload["vlm"])
                if L != state["n_layers"]:
                    state["n_layers"] = L
                    try:
                        cv2.setTrackbarMax("layer", WINDOW, max(0, L - 1))
                    except cv2.error:
                        pass

        model_action = chunk[step % args.horizon].astype(np.float32, copy=True)
        recorded_actions.append(model_action.copy())

        # `joint_position` is ABSOLUTE in radians (verified against the model's
        # statistics.json — the action stats span Panda's full joint range with
        # mean ≈ home pose). Send straight through; no current+Δ anchoring.
        target_qpos = model_action[:7]
        gripper_abs = float(model_action[7])
        gripper_cmd = 1.0 if gripper_abs > args.gripper_thresh else -1.0

        # Route into the env's flat action vector. Layouts:
        #   Panda:       env[0:7]=qpos, env[7]=gripper                          (8-D)
        #   PandaOmron:  env[0:7]=qpos, env[7]=torso, env[8:11]=base(x,y,yaw),
        #                env[11]=right_gripper, env[12]=control_mode            (13-D
        #                with HYBRID_MOBILE_BASE; smaller without).
        env_action = np.zeros(env.action_dim, dtype=np.float32)
        env_action[0:7] = target_qpos
        if args.robot == "PandaOmron":
            # torso, base, control_mode all forced (no GR00T head for them).
            if env.action_dim > 7:
                env_action[7] = 0.0                      # torso
            if env.action_dim > 10:
                env_action[8:11] = 0.0                   # base x/y/yaw
            grip_idx = 11 if env.action_dim > 11 else (env.action_dim - 1)
            env_action[grip_idx] = gripper_cmd
            if env.action_dim > 12:
                env_action[12] = -1.0                    # arm-only mode
        else:
            env_action[7] = gripper_cmd

        # JOINT_POSITION is identity-passthrough in radians; no clip on the
        # arm dims. Clip torso/base/gripper/cmode which live in [-1, 1].
        if args.robot == "PandaOmron":
            env_action[7:] = np.clip(env_action[7:], -1.0, 1.0)
        else:
            env_action[7:] = np.clip(env_action[7:], -1.0, 1.0)

        arm_str = np.array2string(target_qpos, precision=3, suppress_small=True, separator=" ")
        pbar.set_postfix_str(
            f"step={step} qpos={arm_str} grip_raw={gripper_abs:+.3f} grip_cmd={gripper_cmd:+.1f}",
            refresh=False,
        )

        obs, _reward, _done, _info = env.step(env_action)

        sim_view = _normalize_robosuite_image(obs["frontview_image"])
        sim_view = cv2.resize(sim_view, (SIM_VIEW_SIZE, SIM_VIEW_SIZE),
                              interpolation=cv2.INTER_AREA)
        ext = _normalize_robosuite_image(obs[_ext_image_key(obs)])
        wrist = _normalize_robosuite_image(obs["robot0_eye_in_hand_image"])

        # Live attention: re-render every step using the most recent payload
        # (refreshed every chunk = `args.horizon` sim steps). Trackbar state
        # for layer/head_mode is read each frame so scrubbing is real-time.
        if last_attn_payload.get("vlm"):
            ext_attn_tile, wrist_attn_tile = render_attn_tiles(
                last_attn_payload, ext, wrist,
                layer_idx=state["layer"],
                head_mode=head_modes[state["head_mode"]],
            )
        else:
            ext_attn_tile = wrist_attn_tile = None

        prompt_with_state = (
            f"{args.prompt}   [layer {state['layer']}/{state['n_layers']-1}, "
            f"head={head_modes[state['head_mode']]}]"
        )
        canvas = compose_canvas(sim_view, ext, wrist,
                                ext_attn=ext_attn_tile,
                                wrist_attn=wrist_attn_tile,
                                prompt=prompt_with_state)
        cv2.imshow(WINDOW, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if interrupted:
            print("\n[run] SIGINT received — finishing up...")
            break

    dt = time.time() - t0
    print(f"Done. {step + 1} sim steps, {n_infer} policy queries in {dt:.1f}s.")

    # Save action diagnostic (per-dim time series + histogram).
    if not args.no_diag and recorded_actions:
        try:
            from viz_sim.diagnose_actions import plot_action_timeseries
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from diagnose_actions import plot_action_timeseries  # type: ignore
        arr = np.stack(recorded_actions, axis=0)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"gr00t_droid_{args.robot}_{args.task}_{ts}"
        out_dir = Path(args.diag_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{stem}.npz", actions=arr,
                            prompt=args.prompt, task=args.task,
                            config="gr00t_droid", robot=args.robot)
        try:
            plot_action_timeseries(
                arr, out_dir / f"{stem}.png", config="gr00t_droid",
                norm_stats_path=None,
                title=f"gr00t-droid | {args.robot} | {args.task} | {arr.shape[0]} steps",
            )
        except Exception as e:
            print(f"[diag] plot failed: {e} (npz still saved)")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
