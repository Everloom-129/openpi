"""Headless eval for the attention-perturbation experiment.

Per (task, condition) pair: run N matched-seed episodes against an already-up
pi05_robocasa365 server. Conditions are encoded in `obs["_perturb"]` so the
server applies the requested layer-7 prefix-V scaling at inference time.

Output layout:
    /mnt/sda/edward/projects/robocasa_365_perturb/{task}/{condition}/ep_NNN.npz

Each npz contains:
    success, final_reward, max_reward, steps, prompt, condition, seed,
    eef_traj (T,3), gripper_traj (T,2), reward_traj (T,),
    pred_chunks (n_calls, 8 or 15, action_dim), action_taken (T, action_dim),
    attn_steps, attn_stacks (k, 18, 8, 512),
    frame_steps, ext_frames (k, 224, 224, 3), wrist_frames (k, 224, 224, 3)

Usage:
    python viz_sim/eval_perturb.py --task Lift --condition baseline --episodes 10 --seed_base 0
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_ROOT = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_perturb")

OPEN_LOOP_HORIZON = 8
TILE = 224
ATTN_SUBSAMPLE = 5

TASK_PROMPT = {
    "Lift": "pick up the cube",
    "Stack": "stack the red cube on top of the green cube",
    "Door": "open the door",
    "PickPlaceCan": "pick up the can and place it in the bin",
    "NutAssemblySquare": "put the square nut onto the peg",
}
# Per-task rollout horizon, chosen to roughly match robocasa target horizons
# (200 / 300 / 400 / 500 / 600). Robosuite's own default is 1000, but our
# horizon=200 was too short and capped every baseline at succ=0.
TASK_HORIZON = {
    "Lift": 200,
    "Door": 300,
    "Stack": 400,
    "PickPlaceCan": 500,
    "NutAssemblySquare": 600,
}

CONDITIONS = {
    "baseline":              {"mode": None,             "camera": None},
    "ext_zero_max":          {"mode": "zero_max",       "camera": "ext"},
    "ext_strengthen_max":    {"mode": "strengthen_max", "camera": "ext"},
    "ext_zero_min":          {"mode": "zero_min",       "camera": "ext"},
    "wrist_zero_max":        {"mode": "zero_max",       "camera": "wrist"},
    "wrist_strengthen_max":  {"mode": "strengthen_max", "camera": "wrist"},
    "wrist_zero_min":        {"mode": "zero_min",       "camera": "wrist"},
}


def to_uint8_rgb(img):
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[::-1])


def _resize_with_pad(img, size=TILE):
    import cv2
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


def make_robocasa_obs(env_obs, prompt):
    """16-D state in upstream order: [eef_pos_rel(3), eef_rot_rel(4 quat),
    base_pos(3), base_rot(4 quat), gripper_qpos(2)]; cameras use
    `robot0_agentview_left` if available (training distribution),
    else `agentview` fallback."""
    ext_key = "robot0_agentview_left_image" if "robot0_agentview_left_image" in env_obs else "agentview_image"
    ext = _resize_with_pad(to_uint8_rgb(env_obs[ext_key]))
    wrist = _resize_with_pad(to_uint8_rgb(env_obs["robot0_eye_in_hand_image"]))
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


def adapt_action(action, env_action_dim, block_base: bool = False):
    """Map 12-D model output (layout B) to robosuite PandaOmron's actual
    composite-controller layout `[arm 6, torso 1, base 3, right_gripper 1, cmode 1]`.
    The gripper-name append in robosuite/robots/robot.py:957 puts
    `right_gripper` AFTER base, so `env[10]` is gripper, not env[6].
    """
    a = np.zeros(env_action_dim, dtype=np.float32)
    if env_action_dim != 12 or len(action) < 12:
        # Non-PandaOmron fallback: naive copy + binarize last dim.
        n = min(len(action), env_action_dim)
        a[:n] = action[:n]
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0
        return np.clip(a, -1.0, 1.0)
    a[0:6] = action[0:6]               # arm OSC_POSE
    if block_base:
        a[6]    = 0.0                  # torso
        a[7:10] = 0.0                  # base x/y/yaw
        a[10]   = float(action[6])     # right_gripper ← model gripper
        a[11]   = -1.0                 # arm-only mode
    else:
        a[6]    = float(action[10])    # torso          ← base_motion[3]
        a[7:10] = action[7:10]         # base x/y/yaw   ← base_motion[0:3]
        a[10]   = float(action[6])     # right_gripper  ← gripper
        a[11]   = float(action[11])    # control_mode
    return np.clip(a, -1.0, 1.0)


def build_env(task: str, seed: int):
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    # PandaOmron + the default HYBRID_MOBILE_BASE composite (arm OSC_POSE +
    # torso JOINT_POSITION + base JOINT_VELOCITY + cmode = 12-D action).
    ctrl = load_composite_controller_config(robot="PandaOmron")

    cam_names = ["agentview", "robot0_eye_in_hand"]
    cam_h = [TILE, TILE]
    cam_w = [TILE, TILE]

    def _make(extra):
        return robosuite.make(
            env_name=task, robots="PandaOmron", controller_configs=ctrl,
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=cam_names + extra,
            camera_heights=cam_h + [TILE] * len(extra),
            camera_widths=cam_w + [TILE] * len(extra),
            ignore_done=True, control_freq=20, seed=seed,
        )
    try:
        return _make(["robot0_agentview_left"])
    except ValueError as e:
        if "robot0_agentview_left" in str(e):
            return _make([])
        raise


def run_episode(env, policy, prompt, perturb_payload, max_steps=200, *, block_base=False):
    obs = env.reset()
    success = False
    last_attn_stack = None
    attn_snaps, frames = [], []
    eef_traj, grip_traj, reward_traj, action_taken = [], [], [], []
    pred_chunks = []
    chunk = None

    ext_obs_key = None  # resolved on first obs
    for step in range(max_steps):
        if ext_obs_key is None:
            ext_obs_key = ("robot0_agentview_left_image"
                           if "robot0_agentview_left_image" in obs else "agentview_image")
        if step % OPEN_LOOP_HORIZON == 0:
            policy_obs = make_robocasa_obs(obs, prompt)
            if perturb_payload is not None:
                policy_obs["_perturb"] = perturb_payload
            result = policy.infer(policy_obs)
            chunk = np.asarray(result["actions"])
            pred_chunks.append(chunk.astype(np.float32))
            attn = result.get("text_to_img_attn")
            if attn is not None:
                last_attn_stack = np.asarray(attn, dtype=np.float32)

        action = adapt_action(chunk[step % OPEN_LOOP_HORIZON],
                              env.action_dim, block_base=block_base)
        obs, reward, _done, _info = env.step(action)

        eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        grip_traj.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
        reward_traj.append(float(reward))
        action_taken.append(action.copy())
        if reward > 0.5:
            success = True

        if step % ATTN_SUBSAMPLE == 0 and last_attn_stack is not None:
            attn_snaps.append((step, last_attn_stack.copy()))
            frames.append((step,
                           to_uint8_rgb(obs[ext_obs_key]),
                           to_uint8_rgb(obs["robot0_eye_in_hand_image"])))

    reward_arr = np.asarray(reward_traj, dtype=np.float32)
    return {
        "success": success,
        "final_reward": float(reward_arr[-1]) if reward_arr.size else 0.0,
        "max_reward": float(reward_arr.max()) if reward_arr.size else 0.0,
        "steps": len(reward_traj),
        "eef_traj": np.stack(eef_traj, axis=0).astype(np.float32),
        "gripper_traj": np.stack(grip_traj, axis=0).astype(np.float32),
        "reward_traj": reward_arr,
        "action_taken": np.stack(action_taken, axis=0).astype(np.float32),
        "pred_chunks": np.stack(pred_chunks, axis=0).astype(np.float32) if pred_chunks else np.zeros((0,), np.float32),
        "attn_steps": np.array([s for s, _ in attn_snaps], dtype=np.int32),
        "attn_stacks": np.stack([a for _, a in attn_snaps], axis=0)
            if attn_snaps else np.zeros((0,), dtype=np.float32),
        "frame_steps": np.array([s for s, _, _ in frames], dtype=np.int32),
        "ext_frames": np.stack([e for _, e, _ in frames], axis=0)
            if frames else np.zeros((0,), dtype=np.uint8),
        "wrist_frames": np.stack([w for _, _, w in frames], axis=0)
            if frames else np.zeros((0,), dtype=np.uint8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_PROMPT.keys()))
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override per-task horizon; default uses TASK_HORIZON[task].")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--block_base", action="store_true",
                    help="zero torso/base and force arm-only mode (gripper still active).")
    args = ap.parse_args()

    prompt = TASK_PROMPT[args.task]
    max_steps = args.max_steps if args.max_steps is not None else TASK_HORIZON[args.task]
    perturb = CONDITIONS[args.condition]
    perturb_payload = (
        {"mode": perturb["mode"], "camera": perturb["camera"], "layer": 7}
        if perturb["mode"] is not None else None
    )

    out_dir = OUT_ROOT / args.task / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[perturb] {args.task}/{args.condition}: connecting {args.host}:{args.port}")
    print(f"[perturb] payload={perturb_payload}  prompt={prompt!r}  max_steps={max_steps}")

    summary = []
    for ep in range(args.episodes):
        seed = args.seed_base + ep
        out_path = out_dir / f"ep_{ep:03d}.npz"
        if out_path.exists():
            print(f"[perturb] ep {ep:03d} already done, skipping")
            continue
        np.random.seed(seed); random.seed(seed)
        env = build_env(args.task, seed=seed)
        t0 = time.time()
        try:
            data = run_episode(env, policy, prompt, perturb_payload,
                                max_steps=max_steps, block_base=args.block_base)
        except Exception as e:
            import traceback
            print(f"[perturb] ep {ep:03d} ERROR: {e}\n{traceback.format_exc()}")
            env.close()
            continue
        env.close()

        data["task"] = args.task
        data["condition"] = args.condition
        data["seed"] = int(seed)
        data["prompt"] = prompt
        np.savez_compressed(out_path, **data)
        summary.append({
            "ep": ep, "seed": seed,
            "success": bool(data["success"]),
            "final_reward": float(data["final_reward"]),
            "max_reward": float(data["max_reward"]),
            "steps": int(data["steps"]),
        })
        print(f"[perturb] ep {ep:03d} seed={seed} succ={data['success']} "
              f"r_max={data['max_reward']:.3f} steps={data['steps']} ({time.time()-t0:.1f}s)")

    (out_dir / "_summary.json").write_text(json.dumps({
        "task": args.task, "condition": args.condition,
        "perturb": perturb_payload, "prompt": prompt,
        "episodes": summary,
    }, indent=2))
    print(f"[perturb] done {args.task}/{args.condition}")


if __name__ == "__main__":
    main()
