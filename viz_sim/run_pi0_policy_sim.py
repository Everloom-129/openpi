"""Closed-loop pi0.5-DROID inference inside a robosuite/robocasa MuJoCo sim.

Architecture
------------
* This script runs in the `robocasa_sim` conda env (mujoco 3.3.1, numpy 2.x).
* The pi0.5 policy runs in the openpi `.venv` and is exposed by
  `viz_sim/run_pi0_policy_server.sh` as a websocket server on localhost:8000.
* We open a `Panda` arm in robosuite with the JOINT_VELOCITY arm controller
  so the env's action space is exactly [joint_vel × 7, gripper × 1] — the
  same 8-dim layout as the DROID checkpoint output. No remapping needed.
* Each step we resize the agent + wrist cameras to 224×224, build a DROID
  obs dict, send it to the policy, and replay the first OPEN_LOOP_HORIZON
  actions before re-querying.

Run (after the policy server is up):
    conda activate robocasa_sim
    python viz_sim/run_pi0_policy_sim.py --task Lift --prompt "pick up the cube"
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import robosuite
from tqdm import tqdm
from robosuite.controllers import load_composite_controller_config

from openpi_client.websocket_client_policy import WebsocketClientPolicy

# pi0.5-DROID emits a chunk of 15 actions; the dashboard executes the first 8.
OPEN_LOOP_HORIZON = 8


SIM_VIEW_SIZE = 448  # third-person sim panel
TILE_SIZE = 224      # ext / wrist / attn tiles


def _droid_arm_cfg() -> dict:
    """JOINT_POSITION delta controller for pi0/pi0.5-DROID.

    The DROID checkpoints train on `action_dict.joint_position` (per
    `src/openpi/training/droid_rlds_dataset.py:34` — "We default to joint
    position actions, since they allow policy evaluation in simulation").
    Norm stats confirm: action[0:7] mean≈0, std≈0.15–0.30 — these are
    *delta* joint positions in radians per chunk step, not velocities.

    We map the policy's [-1, 1] output range to ±0.3 rad delta per step.
    The previous JOINT_VELOCITY config was incorrect and produced systematic
    drift: it interpreted radians as rad/s and integrated them at 20Hz.
    """
    return {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.3,
        "output_min": -0.3,
        "kp": 50,
        "velocity_limits": [-2, 2],
        "interpolation": None,
        "ramp_ratio": 0.2,
        "control_delta": True,
        "gripper": {"type": "GRIP"},
    }


def _libero_arm_cfg() -> dict:
    # OSC_POSE — LIBERO's default Panda controller. Action layout is
    # [dx, dy, dz, droll, dpitch, dyaw], all in [-1, 1] (output limits below
    # match LIBERO's robosuite defaults).
    return {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }


def _robocasa_arm_cfg() -> dict:
    """OSC_POSE delta arm controller for pi0.5-robocasa365.

    pi05_robocasa365 was trained on robocasa target tasks (PandaOmron + 4-DoF
    mobile base + composite controller; 12-D action, 16-D state).

    There are TWO different 12-dim layouts in the robocasa codebase:

    A. **LeRobot dataset layout** (`convert_hdf5_lerobot.py` + the
       `PandaOmron_modality.json` key map):
           action[0:4]   base_motion          (4 dims)
           action[4]     control_mode        (1)
           action[5:8]   end_effector_position (3)
           action[8:11]  end_effector_rotation (3)
           action[11]    gripper_close       (1)
       state (16): [base_pos(3), base_rot(4 quat), eef_pos_rel(3),
                    eef_rot_rel(4 quat), gripper_qpos(2)].

    B. **Gym-wrapper layout** (`robocasa/utils/env_utils.py:134 convert_action`):
           action[0:3]   end_effector_position
           action[3:6]   end_effector_rotation
           action[6]     gripper_close
           action[7:11]  base_motion
           action[11]    control_mode

    The pi05_robocasa365 *checkpoint's* `norm_stats.json` matches **layout B**:
    eef_pos has uniform std ≈ 0.32 in dims 0–2; eef_rot has std ≈ 0.10 in
    dims 3–5; the [-1,+1] gripper sits at dim 6 with std ≈ 0.99 (perfect
    Bernoulli signature); base motion in dims 7–10 (3 active + 1 zero);
    control_mode at dim 11 (std ≈ 0.70). So we trust the checkpoint and
    decode model output as layout B.

    `serve_policy_attn.py` runs with `--config=pi05_droid`, so `DroidOutputs`
    slices to the first 8 dims — under layout B these are exactly
    `[Δeef_pos(3), Δeef_rot(3), gripper(1), base_x(1)]`. The other 3 base
    dims and the control_mode flag are dropped.
    """
    return {
        "type": "OSC_POSE",
        "input_max": 1,
        "input_min": -1,
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "position_limits": None,
        "orientation_limits": None,
        "uncouple_pos_ori": True,
        "control_delta": True,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }


def build_env(task: str, config: str):
    """Create a robosuite env with the right robot + controller for `config`.

    - pi0/pi0.5-DROID    → Panda + JOINT_POSITION delta (8-dim action)
    - pi0.5-LIBERO       → Panda + OSC_POSE delta (7-dim action)
    - pi0.5-ROBOCASA365  → PandaOmron + OSC_POSE arm + composite mobile base
                           (uses first 8 dims of robocasa's 12-dim layout)

    Renders ext + wrist at 224 (policy input) and a third-person `frontview`
    at SIM_VIEW_SIZE (composited UI panel). No on-screen GLFW window — we
    composite everything into a single cv2 canvas instead.
    """
    if config == "pi05_robocasa365":
        # PandaOmron: 7-DoF Panda mounted on a 4-DoF mobile base. The default
        # composite controller for PandaOmron exposes the base as a separate
        # body_part — env.action_dim is correspondingly larger than 8.
        # We override the arm with OSC_POSE delta; the base controller is
        # left at its default so the model's base-motion dim still actuates
        # the mobile platform.
        controller_cfg = load_composite_controller_config(robot="PandaOmron")
        controller_cfg["body_parts"]["right"] = _robocasa_arm_cfg()
        robot_name = "PandaOmron"
    else:
        controller_cfg = load_composite_controller_config(robot="Panda")
        if config == "pi05_libero":
            controller_cfg["body_parts"]["right"] = _libero_arm_cfg()
        else:
            controller_cfg["body_parts"]["right"] = _droid_arm_cfg()
        robot_name = "Panda"

    # Per-camera sizes: frontview large, the two policy-input cams at 224.
    # robocasa365 was trained on `robot0_agentview_left` (a robot-mounted side
    # view that follows PandaOmron's mobile base). That camera is registered
    # by robocasa envs; for plain-robosuite tasks (e.g. PickPlaceSingle) we
    # fall back to `agentview` so env creation doesn't error out.
    cam_names = ["frontview", "agentview", "robot0_eye_in_hand"]
    cam_heights = [SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE]
    cam_widths = [SIM_VIEW_SIZE, TILE_SIZE, TILE_SIZE]

    def _make(extra_cams: list[str]):
        names = cam_names + extra_cams
        heights = cam_heights + [TILE_SIZE] * len(extra_cams)
        widths = cam_widths + [TILE_SIZE] * len(extra_cams)
        return robosuite.make(
            env_name=task,
            robots=robot_name,
            controller_configs=controller_cfg,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=names,
            camera_heights=heights,
            camera_widths=widths,
            ignore_done=True,
            control_freq=20,
        )

    if config == "pi05_robocasa365":
        try:
            return _make(["robot0_agentview_left"])
        except ValueError as e:
            if "robot0_agentview_left" in str(e):
                print(f"[build_env] task '{task}' has no robot0_agentview_left camera; "
                      f"falling back to agentview (training-distribution camera unavailable)")
            else:
                raise
    return _make([])


def to_uint8_rgb(img, size: int | None = TILE_SIZE) -> np.ndarray:
    """Robosuite returns float[0,1] BGR-ish images flipped vertically. Normalize.

    If `size` is provided and the image isn't already that size, resize to (size, size).
    Pass `size=None` to keep the camera's native resolution.
    """
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr[::-1])
    if size is not None and arr.shape[:2] != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr


def placeholder_attn(size: int = TILE_SIZE) -> np.ndarray:
    """Stand-in attention tile when no attention is available."""
    tile = np.full((size, size, 3), 40, dtype=np.uint8)
    cv2.putText(tile, "no attn", (size // 2 - 40, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return tile


def attn_overlay(img: np.ndarray, attn_256: np.ndarray, *,
                 alpha: float = 0.45, size: int = TILE_SIZE) -> np.ndarray:
    """Reshape (256,) → 16×16, upsample to `size`, jet-blend onto `img` (RGB).

    Per-camera min/max normalization (matches viz/dashboard/views/grid_heatmap.py).
    Returns RGB uint8 of shape (size, size, 3).
    """
    grid = np.asarray(attn_256, dtype=np.float32).reshape(16, 16)
    up = cv2.resize(grid, (size, size), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(up.min()), float(up.max())
    norm = (up - lo) / (hi - lo + 1e-8)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    base = img if img.shape[:2] == (size, size) else cv2.resize(img, (size, size))
    return cv2.addWeighted(base, 1 - alpha, color, alpha, 0)


def label(tile: np.ndarray, text: str) -> np.ndarray:
    """Draw a small label in the top-left corner of an RGB tile (in-place safe copy)."""
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (max(80, 8 * len(text)), 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


PROMPT_STRIP_HEIGHT = 36


def prompt_strip(width: int, text: str) -> np.ndarray:
    """Render a full-width caption bar showing the current instruction."""
    strip = np.full((PROMPT_STRIP_HEIGHT, width, 3), 20, dtype=np.uint8)
    if not text:
        return strip
    # Truncate if it would overflow the strip.
    max_chars = max(10, width // 11)
    shown = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    (tw, th), _ = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(strip, shown, ((width - tw) // 2, (PROMPT_STRIP_HEIGHT + th) // 2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return strip


def compose_canvas(sim: np.ndarray, ext: np.ndarray, wrist: np.ndarray,
                   ext_attn: np.ndarray, wrist_attn: np.ndarray,
                   prompt: str = "") -> np.ndarray:
    """Build the unified UI canvas.

    Layout:
        [ sim view (448x448) ] [ ext (224)   | ext_attn (224)   ]
        [                    ] [ wrist (224) | wrist_attn (224) ]
        [               instruction strip (full width)          ]
    Total: 896 wide x (448 + PROMPT_STRIP_HEIGHT) tall.
    """
    sim = label(sim, "sim (frontview)")
    right_top = np.hstack([label(ext, "ext"), label(ext_attn, "ext attn")])
    right_bot = np.hstack([label(wrist, "wrist"), label(wrist_attn, "wrist attn")])
    right = np.vstack([right_top, right_bot])  # 448 x 448
    grid = np.hstack([sim, right])              # 448 x 896
    return np.vstack([grid, prompt_strip(grid.shape[1], prompt)])


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Mirror examples/libero/main.py:_quat2axisangle (xyzw → 3-vec)."""
    quat = np.asarray(quat, dtype=np.float64)
    # robosuite returns xyzw; clip w for numerical stability.
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(1.0 - w * w))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] / den * angle).astype(np.float32)


def make_libero_obs(env_obs: dict, prompt: str) -> dict:
    """Map a robosuite obs dict to the LIBERO input format expected by pi0.5-libero.

    State is 8-dim: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)] — matches
    examples/libero/main.py.
    """
    base = to_uint8_rgb(env_obs["agentview_image"])
    wrist = to_uint8_rgb(env_obs["robot0_eye_in_hand_image"])
    state = np.concatenate([
        np.asarray(env_obs["robot0_eef_pos"], dtype=np.float32),
        _quat2axisangle(env_obs["robot0_eef_quat"]),
        np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32),
    ]).astype(np.float32)
    return {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def _resize_with_pad(img: np.ndarray, size: int = TILE_SIZE) -> np.ndarray:
    """Aspect-preserving resize with zero-padding to (size, size). Matches
    `openpi_client.image_tools.resize_with_pad` used in upstream robocasa eval."""
    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def make_robocasa_obs(env_obs: dict, prompt: str) -> dict:
    """Mirror of upstream `examples/robocasa/main.py`.

    State is a single 16-D vector concatenated as
        [eef_pos_rel(3), eef_rot_rel(4 quat xyzw),
         base_pos(3),    base_rot(4 quat xyzw),
         gripper_qpos(2)]
    Cameras are `robot0_agentview_left` (robot-mounted, follows the mobile
    base) and `robot0_eye_in_hand`, both letterboxed to 224×224 with
    `_resize_with_pad`.
    """
    ext_key = ("robot0_agentview_left_image"
               if "robot0_agentview_left_image" in env_obs
               else "agentview_image")
    ext_raw = to_uint8_rgb(env_obs[ext_key], size=None)
    wrist_raw = to_uint8_rgb(env_obs["robot0_eye_in_hand_image"], size=None)
    ext = _resize_with_pad(ext_raw, TILE_SIZE)
    wrist = _resize_with_pad(wrist_raw, TILE_SIZE)
    state = np.concatenate([
        np.asarray(env_obs["robot0_base_to_eef_pos"], dtype=np.float32).reshape(-1),
        np.asarray(env_obs["robot0_base_to_eef_quat"], dtype=np.float32).reshape(-1),
        np.asarray(env_obs["robot0_base_pos"], dtype=np.float32).reshape(-1),
        np.asarray(env_obs["robot0_base_quat"], dtype=np.float32).reshape(-1),
        np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
    ]).astype(np.float32)
    return {
        "observation/image": ext,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def make_droid_obs(env_obs: dict, prompt: str) -> dict:
    """Map a robosuite obs dict to the DROID input format expected by pi0.5."""
    ext = to_uint8_rgb(env_obs["agentview_image"])
    wrist = to_uint8_rgb(env_obs["robot0_eye_in_hand_image"])
    qpos = np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32)  # 7
    # Robosuite gripper qpos is 2 (two fingers); collapse to 1-dim normalized open/close.
    grip_qpos = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper = np.array([grip_qpos[0] - grip_qpos[1]], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": ext,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": qpos,
        "observation/gripper_position": gripper,
        "prompt": prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift", help="robosuite env name")
    ap.add_argument("--prompt", default="pick up the red cube")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=OPEN_LOOP_HORIZON,
                    help="how many actions from each chunk to execute before re-querying")
    ap.add_argument("--config", default="pi05_droid",
                    choices=["pi05_droid", "pi0_droid", "pi05_libero", "pi05_robocasa365"],
                    help="must match the policy server's CONFIG; controls obs/action layout")
    ap.add_argument("--diag_dir", default="viz_sim/results/action_diag",
                    help="where to save the per-dim action diagnostic plot/npz on exit")
    ap.add_argument("--no_diag", action="store_true",
                    help="skip writing the action diagnostic on exit")
    ap.add_argument("--block_base", action="store_true",
                    help="(robocasa only) zero env_action[7:11] (torso + base x/y/yaw) and "
                         "force control_mode=-1 so only the arm + gripper are driven. "
                         "Use to isolate manipulator behavior from mobile-base / torso noise.")
    ap.add_argument("--prompt_neg", default=None,
                    help="DeLock CPG: negative (trained) prompt that captures post-training bias. "
                         "When set together with --cpg_w, the server runs contrastive prompt guidance: "
                         "v_cpg = v_neg + w*(v_pos - v_neg), where v_pos is conditioned on --prompt.")
    ap.add_argument("--cpg_w", type=float, default=None,
                    help="DeLock CPG guidance scale. w=1 recovers vanilla --prompt sampling, "
                         "w=0 recovers --prompt_neg sampling, w>1 extrapolates along the contrast.")
    args = ap.parse_args()
    if (args.prompt_neg is None) != (args.cpg_w is None):
        ap.error("--prompt_neg and --cpg_w must be set together (CPG is opt-in).")

    is_libero = args.config == "pi05_libero"
    is_robocasa = args.config == "pi05_robocasa365"
    print(f"Connecting to policy at ws://{args.host}:{args.port} ... (config={args.config})")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected. Server metadata:", policy.get_server_metadata())

    env = build_env(args.task, args.config)
    if is_robocasa and args.block_base:
        print("[robocasa] --block_base: torso + base will be pinned to 0, "
              "control_mode forced to -1 (arm-only)")
    if is_robocasa:
        # Dump the composite controller's action layout so we can verify how
        # robosuite slices our 12-D action across body parts (arm/gripper/
        # torso/base/...). The training-side layout is layout B
        # [eef(6), grip(1), base(4), control_mode(1)] from
        # robocasa.utils.env_utils.convert_action — but raw robosuite (no
        # gym wrapper) uses whatever order the body_parts dict yields.
        try:
            split = env.robots[0].composite_controller._action_split_indexes
            print("[robocasa] composite action layout:",
                  ", ".join(f"{k}=[{v[0]}:{v[1]}]" for k, v in split.items()))
        except Exception as e:
            print("[robocasa] could not inspect composite action layout:", e)
        expected_model_dims = 12          # server returns full robocasa 12-D action
    elif is_libero:
        expected_model_dims = 7
    else:
        expected_model_dims = 8
    print(f"Env action_dim={env.action_dim}  (model returns {expected_model_dims} dims)")
    if is_robocasa and env.action_dim < 12:
        print(f"WARN: env.action_dim={env.action_dim} < 12; some robocasa action dims will be dropped")
    obs = env.reset()

    # Live attention controls: trackbars on the cv2 window let you scrub
    # layer (0..L-1) and head-aggregation (avg/min/max) without restarting.
    WINDOW = "openpi sim"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    head_modes = ["avg", "min", "max"]
    state = {"layer": 7, "head_mode": 0, "n_layers": 18, "n_heads": 8}
    cv2.createTrackbar("layer", WINDOW, state["layer"], state["n_layers"] - 1,
                       lambda v: state.update(layer=v))
    cv2.createTrackbar("head: 0=avg 1=min 2=max", WINDOW, state["head_mode"],
                       len(head_modes) - 1, lambda v: state.update(head_mode=v))

    def reduce_attn(stack: np.ndarray) -> np.ndarray:
        """stack: (L, H, 512) → (512,) using current trackbar state."""
        L, H, _ = stack.shape
        layer = min(state["layer"], L - 1)
        heads = stack[layer]  # (H, 512)
        mode = head_modes[state["head_mode"]]
        if mode == "avg":
            return heads.mean(axis=0)
        if mode == "min":
            return heads.min(axis=0)
        return heads.max(axis=0)

    t0 = time.time()
    n_infer = 0
    last_stack: np.ndarray | None = None  # (L, H, 512) attention from last infer
    recorded_actions: list[np.ndarray] = []  # per-step model_action (post-unnorm, pre-clip)
    pbar = tqdm(range(args.steps), desc="sim", dynamic_ncols=True)
    interrupted = False

    def _save_action_diag():
        if args.no_diag or not recorded_actions:
            return
        from datetime import datetime
        from pathlib import Path
        try:
            from viz_sim.diagnose_actions import plot_action_timeseries
        except ImportError:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from diagnose_actions import plot_action_timeseries  # type: ignore
        arr = np.stack(recorded_actions, axis=0)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"{args.config}_{args.task}_{ts}"
        out_dir = Path(args.diag_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{stem}.npz"
        png_path = out_dir / f"{stem}.png"
        np.savez_compressed(npz_path, actions=arr, prompt=args.prompt, task=args.task,
                            config=args.config)
        # Find norm_stats.json next to the ckpt for training-distribution overlay.
        repo_root = Path(__file__).resolve().parents[1]
        norm_path = repo_root / "checkpoints" / "viz" / f"{args.config}_pytorch" / "assets" / "droid" / "norm_stats.json"
        plot_action_timeseries(
            arr, png_path, config=args.config,
            norm_stats_path=norm_path if norm_path.exists() else None,
            title=f"{args.config} | {args.task} | {arr.shape[0]} steps",
        )

    import signal as _signal

    def _on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
    _signal.signal(_signal.SIGINT, _on_sigint)

    for step in pbar:
        # Query the policy for a fresh action chunk every `horizon` steps.
        if step % args.horizon == 0:
            # Obs format follows the server's `--config`. Each builder ships
            # the exact keys that config's input transform expects:
            #   pi05_libero       → libero 8-D state
            #   pi05_robocasa365  → robocasa 16-D state (base + base→eef + gripper)
            #   pi0/pi0.5-droid   → DROID joint-position state
            if is_libero:
                policy_obs = make_libero_obs(obs, args.prompt)
            elif is_robocasa:
                policy_obs = make_robocasa_obs(obs, args.prompt)
            else:
                policy_obs = make_droid_obs(obs, args.prompt)
            if args.prompt_neg is not None and args.cpg_w is not None:
                policy_obs["prompt_neg"] = args.prompt_neg
                policy_obs["cpg_w"] = float(args.cpg_w)
            result = policy.infer(policy_obs)
            chunk = np.asarray(result["actions"])  # (N, 7) libero, (N, 8) droid/robocasa365
            n_infer += 1
            attn_field = result.get("text_to_img_attn")
            if attn_field is not None:
                arr = np.asarray(attn_field, dtype=np.float32)
                if arr.ndim == 3:  # (L, H, 512) — current server format
                    last_stack = arr
                    L, H = arr.shape[:2]
                    if L != state["n_layers"] or H != state["n_heads"]:
                        state["n_layers"], state["n_heads"] = L, H
                        cv2.setTrackbarMax("layer", WINDOW, L - 1)
                elif arr.ndim == 1 and arr.shape[0] == 512:  # legacy (512,)
                    last_stack = arr[None, None]

        model_action = chunk[step % args.horizon].astype(np.float32, copy=True)
        recorded_actions.append(model_action.copy())

        # Per-config routing into the env's flat action vector.
        env_action = np.zeros(env.action_dim, dtype=np.float32)
        if is_robocasa:
            # Model layout (robocasa gym-wrapper layout B):
            #   [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
            # where base_motion = [base_x, base_y, base_yaw, torso_z]
            # (see robocasa/wrappers/gym_wrapper.py:121-124).
            #
            # Raw robosuite PandaOmron HYBRID_MOBILE_BASE layout. Iteration
            # order over part_controller_config is `[right, torso, base,
            # right_gripper]` because robot.py:957-958 *appends* "right_gripper"
            # AFTER the body_parts dict has been flattened. Plus 1 cmode dim
            # at the end for HYBRID_MOBILE_BASE:
            #   env[0:6]  arm OSC_POSE
            #   env[6]    torso
            #   env[7:10] base (x, y, yaw)
            #   env[10]   right_gripper
            #   env[11]   control_mode
            env_action[0:6] = model_action[0:6]            # arm
            if args.block_base:
                # Pin torso + base to 0 and force arm-only mode. Keep gripper
                # passthrough so manipulation is still observable.
                env_action[6]    = 0.0                     # torso
                env_action[7:10] = 0.0                     # base
                env_action[10]   = float(model_action[6])  # right_gripper ← model gripper
                env_action[11]   = -1.0                    # arm-only mode
            else:
                env_action[6]    = float(model_action[10]) # torso        ← base_motion[3]
                env_action[7:10] = model_action[7:10]      # base x,y,yaw ← base_motion[0:3]
                env_action[10]   = float(model_action[6])  # right_gripper ← gripper
                env_action[11]   = float(model_action[11]) # control_mode
            raw_gripper = float(model_action[6])
            arm_label = "Δpose"
        elif is_libero:
            # OSC_POSE Panda; gripper passthrough (already ∈ [-1, +1]).
            n = min(len(model_action), env.action_dim)
            env_action[:n] = model_action[:n]
            raw_gripper = float(env_action[-1])
            arm_label = "Δpose"
        else:
            # JOINT_POSITION delta Panda; DROID gripper [0,1] → binarize ±1.
            n = min(len(model_action), env.action_dim)
            env_action[:n] = model_action[:n]
            raw_gripper = float(env_action[-1])
            env_action[-1] = 1.0 if raw_gripper > 0.5 else -1.0
            arm_label = "Δjoint"

        env_action = np.clip(env_action, -1.0, 1.0)

        # Pretty-print: arm slice + gripper + (optional) base slice.
        if is_robocasa:
            def _fmt(v):
                return ",".join(f"{x:+.2f}" for x in v)
            pos_str = _fmt(env_action[:3])
            rot_str = _fmt(env_action[3:6])
            torso = float(env_action[6])  if env.action_dim > 6 else float("nan")
            base_str = _fmt(env_action[7:10]) if env.action_dim > 9 else ""
            grip_cmd = float(env_action[10]) if env.action_dim > 10 else float("nan")
            cmode = float(env_action[11])    if env.action_dim > 11 else float("nan")
            pbar.set_postfix_str(
                f"s={step} p={pos_str}|r={rot_str} t={torso:+.2f} b={base_str} g={grip_cmd:+.1f} m={cmode:+.1f}",
                refresh=False,
            )
        else:
            arm_str = np.array2string(env_action[:-1], precision=3, suppress_small=True, separator=" ")
            pbar.set_postfix_str(
                f"step={step} {arm_label}={arm_str} grip_raw={raw_gripper:+.3f} grip_cmd={env_action[-1]:+.2f}",
                refresh=False,
            )

        obs, _reward, _done, _info = env.step(env_action)

        sim = to_uint8_rgb(obs["frontview_image"], size=SIM_VIEW_SIZE)
        ext = to_uint8_rgb(obs["agentview_image"])
        wrist = to_uint8_rgb(obs["robot0_eye_in_hand_image"])
        if last_stack is not None:
            reduced = reduce_attn(last_stack)
            ext_attn = attn_overlay(ext, reduced[:256])
            wrist_attn = attn_overlay(wrist, reduced[256:])
        else:
            ext_attn = placeholder_attn()
            wrist_attn = placeholder_attn()

        prompt_with_state = (
            f"{args.prompt}   [layer {state['layer']}/{state['n_layers']-1}, "
            f"head={head_modes[state['head_mode']]}]"
        )
        canvas = compose_canvas(sim, ext, wrist, ext_attn, wrist_attn, prompt=prompt_with_state)
        cv2.imshow(WINDOW, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if interrupted:
            print("\n[run] SIGINT received — finishing up...")
            break

    dt = time.time() - t0
    print(f"Done. {step + 1} sim steps, {n_infer} policy queries in {dt:.1f}s.")
    _save_action_diag()
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
