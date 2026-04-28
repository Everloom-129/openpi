"""Headless evaluation runner for openpi/GR00T policies on robosuite tasks.

Designed to run resumably under a /loop cron:
- State of truth is `results/_state.json` ({model: {task: [ep_indices_done]}}).
- Per episode: writes one .npz with success, steps, prompt, attn snapshots
  (subsampled across the rollout) and a few RGB frames (also subsampled).
- No cv2 window. Talks to whatever policy server is up on localhost:--port.

Usage (must run inside the robocasa_sim conda env, after the right server is up):
    python viz_sim/eval_runner.py --model pi05_droid --task Lift --episodes 20

The orchestration script `viz_sim/run_all_evals.sh` brings each server up/down.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import traceback

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
ATTN_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365")
STATE_PATH = RESULTS_DIR / "_state.json"

OPEN_LOOP_HORIZON = 8
TILE = 224

# ---------------------------------------------------------------- state I/O ---


def load_state() -> dict:
    if not STATE_PATH.exists():
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({"completed": {}, "errors": {}}, indent=2))
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def mark_done(model: str, task: str, ep: int) -> None:
    s = load_state()
    s.setdefault("completed", {}).setdefault(model, {}).setdefault(task, [])
    if ep not in s["completed"][model][task]:
        s["completed"][model][task].append(ep)
    save_state(s)


def mark_error(model: str, task: str, ep: int, msg: str) -> None:
    s = load_state()
    s.setdefault("errors", {}).setdefault(model, {})[f"{task}/{ep}"] = msg[:500]
    save_state(s)


def episode_already_done(model: str, task: str, ep: int) -> bool:
    # Disk is the ground truth: a written .npz means the episode produced data.
    if (ATTN_DIR / model / task / f"ep_{ep:03d}.npz").exists():
        return True
    s = load_state()
    return ep in s.get("completed", {}).get(model, {}).get(task, [])


# ---------------------------------------------------------------- sim env ----


def build_env(task: str, controller: str = "JOINT_VELOCITY"):
    """Headless robosuite Panda env. No on-screen renderer."""
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    ctrl = load_composite_controller_config(robot="Panda")
    if controller == "JOINT_VELOCITY":
        ctrl["body_parts"]["right"] = {
            "type": "JOINT_VELOCITY",
            "input_max": 1, "input_min": -1,
            "output_max": 0.5, "output_min": -0.5,
            "kp": 3.0,
            "velocity_limits": [-1, 1],
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    elif controller == "OSC_POSE":
        ctrl["body_parts"]["right"] = {
            "type": "OSC_POSE",
            "input_max": 1, "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None,
            "uncouple_pos_ori": True, "control_delta": True,
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
    return robosuite.make(
        env_name=task,
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=TILE, camera_widths=TILE,
        ignore_done=True,
        control_freq=20,
    )


def to_uint8_rgb(img) -> np.ndarray:
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[::-1])


# -------------------------------------------------------------- obs builders -


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(1.0 - w * w))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] / den * angle).astype(np.float32)


def make_droid_obs(env_obs, prompt):
    grip = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": to_uint8_rgb(env_obs["agentview_image"]),
        "observation/wrist_image_left": to_uint8_rgb(env_obs["robot0_eye_in_hand_image"]),
        "observation/joint_position": np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32),
        "observation/gripper_position": np.array([grip[0] - grip[1]], dtype=np.float32),
        "prompt": prompt,
    }


def make_libero_obs(env_obs, prompt):
    state = np.concatenate([
        np.asarray(env_obs["robot0_eef_pos"], dtype=np.float32),
        _quat2axisangle(env_obs["robot0_eef_quat"]),
        np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32),
    ]).astype(np.float32)
    return {
        "observation/image": to_uint8_rgb(env_obs["agentview_image"]),
        "observation/wrist_image": to_uint8_rgb(env_obs["robot0_eye_in_hand_image"]),
        "observation/state": state,
        "prompt": prompt,
    }


# --------------------------------------------------------------- main ---------


MODEL_CFG = {
    # Each entry: server config, controller, obs builder, action handler.
    "pi05_droid": {
        "controller": "JOINT_VELOCITY",
        "obs_fn": "droid",
        "action_dim": 8,
        "default_prompt": "pick up the cube",
        "transport": "ws",
    },
    "pi05_libero": {
        "controller": "OSC_POSE",
        "obs_fn": "libero",  # server-side transform requires observation/{image,wrist_image,state}
        "action_dim": 7,
        "default_prompt": "pick up the cube and lift it",
        "transport": "ws",
    },
    "pi05_robocasa365": {
        "controller": "JOINT_VELOCITY",
        "obs_fn": "droid",  # same shape as droid; ckpt was trained on robocasa data so behavior may be off-distribution but attn captures
        "action_dim": 8,
        "default_prompt": "pick up the cube",
        "transport": "ws",
    },
    "gr00t_n15_robocasa365": {
        "controller": "JOINT_POSITION",
        "obs_fn": "gr00t_robocasa",
        "action_dim": 7,
        "default_prompt": "pick up the cube",
        "transport": "zmq",
    },
}


def build_obs(name: str, env_obs, prompt: str):
    if name == "droid":
        return make_droid_obs(env_obs, prompt)
    if name == "libero":
        return make_libero_obs(env_obs, prompt)
    if name == "gr00t_robocasa":
        # Best-effort obs for the GR00T-N1.5 robocasa ckpt. The ckpt expects
        # 3 cameras + mobile base + relative eef + gripper_qpos(2). We only
        # have a fixed-base Panda and 2 cameras, so we synthesize the missing
        # modalities. The server will likely reject this; we record the error
        # and move on (the gr00t inference path also doesn't expose attention,
        # so this branch produces no comparable data either way).
        from scipy.spatial.transform import Rotation as R
        import cv2

        def _resize(img, h=256, w=256):
            arr = to_uint8_rgb(img)
            return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)

        ext = _resize(env_obs["agentview_image"])
        wrist = _resize(env_obs["robot0_eye_in_hand_image"])
        eef_pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(env_obs["robot0_eef_quat"], dtype=np.float32)  # xyzw
        grip = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
        joint = np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32)
        return {
            "video.robot0_eye_in_hand": wrist[None, ...],
            "video.robot0_agentview_left": ext[None, ...],
            "video.robot0_agentview_right": ext[None, ...],  # duplicate; real env has 1 side cam
            "state.base_position": np.zeros((1, 3), dtype=np.float32),
            "state.base_rotation": np.array([[0, 0, 0, 1]], dtype=np.float32),
            "state.end_effector_position_relative": eef_pos[None, ...],
            "state.end_effector_rotation_relative": eef_quat[None, ...],
            "state.gripper_qpos": grip[None, ...],
            "annotation.human.action.task_description": [prompt],
        }
    raise ValueError(name)


def adapt_action(action: np.ndarray, env_action_dim: int, model: str) -> np.ndarray:
    """First-N slice + DROID gripper [0,1]→[-1,1] binarization."""
    a = np.zeros(env_action_dim, dtype=np.float32)
    n = min(len(action), env_action_dim)
    a[:n] = action[:n]
    # gripper: pi0/pi0.5-DROID + libero ckpt outputs (we feed DROID obs); binarize.
    a[-1] = 1.0 if a[-1] > 0.5 else -1.0
    return np.clip(a, -1.0, 1.0)


def run_episode(env, policy, model: str, task: str, ep_idx: int, prompt: str,
                max_steps: int = 200, attn_subsample: int = 5):
    """One rollout. Returns dict for npz."""
    cfg = MODEL_CFG[model]
    obs = env.reset()
    success = False
    last_reward = 0.0

    attn_snapshots = []  # list of (step, layer-stack) — we keep last only by default to save space
    last_attn_stack = None
    frames = []  # subsampled

    chunk = None
    for step in range(max_steps):
        if step % OPEN_LOOP_HORIZON == 0:
            policy_obs = build_obs(cfg["obs_fn"], obs, prompt)
            try:
                result = policy.infer(policy_obs)
            except Exception as e:
                raise RuntimeError(f"infer failed at step {step}: {e}") from e
            chunk = np.asarray(result["actions"])
            attn_field = result.get("text_to_img_attn")
            if attn_field is not None:
                last_attn_stack = np.asarray(attn_field, dtype=np.float32)

        action = adapt_action(chunk[step % OPEN_LOOP_HORIZON], env.action_dim, model)
        obs, reward, _done, _info = env.step(action)
        last_reward = float(reward)
        if reward > 0.5:
            success = True

        if step % attn_subsample == 0 and last_attn_stack is not None:
            attn_snapshots.append((step, last_attn_stack.copy()))
            frames.append((step,
                           to_uint8_rgb(obs["agentview_image"]),
                           to_uint8_rgb(obs["robot0_eye_in_hand_image"])))

    return {
        "model": model,
        "task": task,
        "episode": ep_idx,
        "prompt": prompt,
        "success": success,
        "final_reward": last_reward,
        "steps": step + 1,
        "attn_steps": np.array([s for s, _ in attn_snapshots], dtype=np.int32),
        "attn_stacks": np.stack([a for _, a in attn_snapshots], axis=0)
            if attn_snapshots else np.zeros((0,), dtype=np.float32),
        "frame_steps": np.array([s for s, _, _ in frames], dtype=np.int32),
        "ext_frames": np.stack([e for _, e, _ in frames], axis=0)
            if frames else np.zeros((0,), dtype=np.uint8),
        "wrist_frames": np.stack([w for _, _, w in frames], axis=0)
            if frames else np.zeros((0,), dtype=np.uint8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG.keys()))
    ap.add_argument("--task", default="Lift")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    cfg = MODEL_CFG[args.model]
    prompt = args.prompt or cfg["default_prompt"]
    port = args.port or (5555 if cfg["transport"] == "zmq" else 8000)

    out_dir = ATTN_DIR / args.model / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model={args.model} task={args.task} eps={args.episodes} prompt={prompt!r}")
    print(f"[eval] connecting {cfg['transport']}://{args.host}:{port}")

    if cfg["transport"] == "ws":
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=args.host, port=port)
        print("[eval] server metadata:", policy.get_server_metadata())
    else:
        from gr00t_client import PolicyClient
        policy = PolicyClient(host=args.host, port=port)
        print("[eval] gr00t ping:", policy.ping())

    env = build_env(args.task, controller=cfg["controller"])
    print(f"[eval] env action_dim={env.action_dim}")

    summary_rows = []
    t0 = time.time()
    for ep in range(args.episodes):
        if episode_already_done(args.model, args.task, ep):
            print(f"[eval] ep {ep:03d} already done, skipping")
            continue
        ep_t0 = time.time()
        try:
            data = run_episode(env, policy, args.model, args.task, ep, prompt,
                               max_steps=args.max_steps)
            np.savez_compressed(out_dir / f"ep_{ep:03d}.npz", **data)
            mark_done(args.model, args.task, ep)
            row = {k: data[k] for k in ("episode", "success", "final_reward", "steps")}
            summary_rows.append(row)
            print(f"[eval] ep {ep:03d} success={data['success']} reward={data['final_reward']:.3f} "
                  f"steps={data['steps']} ({time.time()-ep_t0:.1f}s)")
        except Exception as e:
            tb = traceback.format_exc()
            mark_error(args.model, args.task, ep, str(e))
            print(f"[eval] ep {ep:03d} ERROR: {e}\n{tb}")
            # bail on persistent errors (e.g., server down)
            if "infer failed" in str(e) or "timeout" in str(e).lower():
                break

    # write per-model summary
    summary_path = out_dir / "_summary.json"
    summary = {
        "model": args.model, "task": args.task, "prompt": prompt,
        "episodes": summary_rows,
        "wall_time_s": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x)))
    print(f"[eval] wrote {summary_path}")

    env.close()

    # Auto-render the cross-ckpt comparison animation for this task.
    # Skips silently if optional deps are missing or no other ckpts have run yet.
    try:
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from render_example_video import render_for_task
        render_for_task(args.task)
    except Exception as e:
        print(f"[eval] post-render skipped: {e}")


if __name__ == "__main__":
    main()
