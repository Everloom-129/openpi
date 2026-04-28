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


def make_droid_obs(env_obs, prompt):
    grip = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32)
    return {
        "observation/exterior_image_1_left": to_uint8_rgb(env_obs["agentview_image"]),
        "observation/wrist_image_left": to_uint8_rgb(env_obs["robot0_eye_in_hand_image"]),
        "observation/joint_position": np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32),
        "observation/gripper_position": np.array([grip[0] - grip[1]], dtype=np.float32),
        "prompt": prompt,
    }


def adapt_action(action, env_action_dim):
    a = np.zeros(env_action_dim, dtype=np.float32)
    n = min(len(action), env_action_dim)
    a[:n] = action[:n]
    a[-1] = 1.0 if a[-1] > 0.5 else -1.0
    return np.clip(a, -1.0, 1.0)


def build_env(task: str, seed: int):
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    ctrl = load_composite_controller_config(robot="Panda")
    ctrl["body_parts"]["right"] = {
        "type": "JOINT_VELOCITY",
        "input_max": 1, "input_min": -1,
        "output_max": 0.5, "output_min": -0.5,
        "kp": 3.0,
        "velocity_limits": [-1, 1],
        "interpolation": None, "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }
    return robosuite.make(
        env_name=task, robots="Panda", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=TILE, camera_widths=TILE,
        ignore_done=True, control_freq=20, seed=seed,
    )


def run_episode(env, policy, prompt, perturb_payload, max_steps=200):
    obs = env.reset()
    success = False
    last_attn_stack = None
    attn_snaps, frames = [], []
    eef_traj, grip_traj, reward_traj, action_taken = [], [], [], []
    pred_chunks = []
    chunk = None

    for step in range(max_steps):
        if step % OPEN_LOOP_HORIZON == 0:
            policy_obs = make_droid_obs(obs, prompt)
            if perturb_payload is not None:
                policy_obs["_perturb"] = perturb_payload
            result = policy.infer(policy_obs)
            chunk = np.asarray(result["actions"])
            pred_chunks.append(chunk.astype(np.float32))
            attn = result.get("text_to_img_attn")
            if attn is not None:
                last_attn_stack = np.asarray(attn, dtype=np.float32)

        action = adapt_action(chunk[step % OPEN_LOOP_HORIZON], env.action_dim)
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
                           to_uint8_rgb(obs["agentview_image"]),
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
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    prompt = TASK_PROMPT[args.task]
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
    print(f"[perturb] payload={perturb_payload}  prompt={prompt!r}")

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
            data = run_episode(env, policy, prompt, perturb_payload, max_steps=args.max_steps)
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
