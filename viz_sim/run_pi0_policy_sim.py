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
    return {
        "type": "JOINT_VELOCITY",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.5,
        "output_min": -0.5,
        "kp": 3.0,
        "velocity_limits": [-1, 1],
        "interpolation": None,
        "ramp_ratio": 0.2,
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


def build_env(task: str, config: str):
    """Create a robosuite Panda env with the right controller for `config`.

    - pi0/pi0.5-DROID  → JOINT_VELOCITY, action [qvel(7), gripper(1)]   (8-dim)
    - pi0.5-LIBERO     → OSC_POSE,       action [dpose(6), gripper(1)]  (7-dim)

    Renders ext + wrist at 224 (policy input) and a third-person `frontview`
    at SIM_VIEW_SIZE (composited UI panel). No on-screen GLFW window — we
    composite everything into a single cv2 canvas instead.
    """
    controller_cfg = load_composite_controller_config(robot="Panda")
    # After loading, robosuite flattens body_parts.arms.right -> body_parts.right.
    if config == "pi05_libero":
        controller_cfg["body_parts"]["right"] = _libero_arm_cfg()
    else:
        controller_cfg["body_parts"]["right"] = _droid_arm_cfg()

    # Per-camera sizes: frontview large, the two policy-input cams at 224.
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
                    choices=["pi05_droid", "pi0_droid", "pi05_libero"],
                    help="must match the policy server's CONFIG; controls obs/action layout")
    args = ap.parse_args()

    is_libero = args.config == "pi05_libero"
    print(f"Connecting to policy at ws://{args.host}:{args.port} ... (config={args.config})")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected. Server metadata:", policy.get_server_metadata())

    env = build_env(args.task, args.config)
    expected = 7 if is_libero else 8
    print(f"Env action_dim={env.action_dim}  (expecting {expected})")
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
    pbar = tqdm(range(args.steps), desc="sim", dynamic_ncols=True)
    for step in pbar:
        # Query the policy for a fresh action chunk every `horizon` steps.
        if step % args.horizon == 0:
            policy_obs = make_libero_obs(obs, args.prompt) if is_libero else make_droid_obs(obs, args.prompt)
            result = policy.infer(policy_obs)
            chunk = np.asarray(result["actions"])  # (N, 7) libero, (N, 8) droid
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

        action = chunk[step % args.horizon].astype(np.float32, copy=True)
        # Defensive: pad/truncate to env.action_dim in case action_dim != 8.
        if action.shape[0] != env.action_dim:
            a = np.zeros(env.action_dim, dtype=np.float32)
            a[: min(len(action), env.action_dim)] = action[: env.action_dim]
            action = a

        # Gripper convention remap.
        # - pi0/pi0.5-DROID outputs gripper in [0, 1] (0=open, 1=close); see
        #   examples/droid/main.py and src/openpi/policies/droid_policy.py.
        # - pi0.5-LIBERO outputs gripper already in [-1, 1] (-1=open, +1=close)
        #   so it can pass through robosuite's GRIP controller untouched.
        raw_gripper = float(action[-1])
        if is_libero:
            action[-1] = raw_gripper  # already in [-1, 1]
        else:
            action[-1] = 1.0 if raw_gripper > 0.5 else -1.0
        action = np.clip(action, -1.0, 1.0)

        arm_label = "dpose" if is_libero else "qvel"
        arm_str = np.array2string(
            action[:-1], precision=3, suppress_small=True, separator=" "
        )
        pbar.set_postfix_str(
            f"step={step} {arm_label}={arm_str} grip_raw={raw_gripper:+.3f} grip_cmd={action[-1]:+.2f}",
            refresh=False,
        )

        obs, _reward, _done, _info = env.step(action)

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

    dt = time.time() - t0
    print(f"Done. {args.steps} sim steps, {n_infer} policy queries in {dt:.1f}s.")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
