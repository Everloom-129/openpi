"""Headless eval on real robocasa kitchen envs (no perturbation).

Builds a `robosuite.make(task, robots="PandaOmron", composite controller)` env
that exposes the training-distribution camera `robot0_agentview_left` and the
12-D HYBRID_MOBILE_BASE composite action. Runs N matched-seed episodes per
(model, task) cell against an already-up websocket server on `--port`.

Each model's chunk shape differs; a per-model adapter packs it into the
12-D PandaOmron env action layout `[arm 6, torso 1, base 3, right_gripper 1, cmode 1]`:

  * pi05_robocasa365 — full 12-D layout B; slot remap (arm/eef in 0:6,
    torso ← model[10], base ← model[7:10], gripper ← model[6], cmode ← model[11]).
  * pi05_libero — 7-D `[Δeef_pos(3), Δeef_rot(3), gripper(1)]`; arm slot is
    OSC_POSE so it fits 0:6 directly. Base/torso zeroed, cmode = -1 (arm-only),
    env[10] ← model[6] gripper.
  * pi05_droid — 8-D `[joint_vel(7), gripper(1)]`. The PandaOmron arm
    controller is OSC_POSE not JOINT_VELOCITY, so the joint-velocity components
    are stuffed (clipped) into env[0:6] purely as a "model still produces
    output, but it's interpreted by an incompatible controller" baseline. The
    expected outcome is near-zero success — recording happens regardless so
    we still get the attention captures.

Per-episode language is read from `env.get_ep_meta()["lang"]` (robocasa builds
a templated instruction during reset that mentions the sampled object/receptacle).

Output:
    /mnt/sda/edward/projects/robocasa_365_eval/{model}/{task}/ep_NNN.npz

Each npz contains: success, final_reward, max_reward, steps, prompt, model,
task, seed, action_taken (T, 12), pred_chunks (n_calls, 15, action_dim_native),
attn_steps, attn_stacks (k, 18, 8, 512), frame_steps, ext_frames, wrist_frames.

Usage:
    python viz_sim/eval_robocasa365.py --model pi05_robocasa365 \
        --task PickPlaceCounterToStove --episodes 20 --seed_base 0
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import time
import traceback

import numpy as np

OUT_ROOT = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_eval")

OPEN_LOOP_HORIZON = 8
TILE = 224
ATTN_SUBSAMPLE = 5
DEFAULT_HORIZON = 400  # robocasa kitchen tasks need longer rollouts than raw robosuite

MODELS = ("pi05_droid", "pi05_libero", "pi05_robocasa365")


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


# ---------------------------------------------------- per-model obs builders ---


def _ext_image(env_obs):
    key = "robot0_agentview_left_image" if "robot0_agentview_left_image" in env_obs else "agentview_image"
    return _resize_with_pad(to_uint8_rgb(env_obs[key]))


def _wrist_image(env_obs):
    return _resize_with_pad(to_uint8_rgb(env_obs["robot0_eye_in_hand_image"]))


def make_robocasa_obs(env_obs, prompt):
    """16-D state in upstream robocasa order."""
    ext = _ext_image(env_obs)
    wrist = _wrist_image(env_obs)
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


def make_droid_obs(env_obs, prompt):
    """8-D state for DROID schema: [joint_pos(7), gripper_pos(1)]. We send
    a single gripper scalar (mean of 2-finger qpos) since the DROID server
    expects `state.gripper_position` of shape (1,)."""
    ext = _ext_image(env_obs)
    wrist = _wrist_image(env_obs)
    joint = np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)[:7]
    grip2 = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    grip = np.array([grip2.mean()], dtype=np.float32)
    state = np.concatenate([joint, grip]).astype(np.float32)  # 8-D
    return {
        "observation/exterior_image_1_left": ext,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": joint,
        "observation/gripper_position": grip,
        "observation/state": state,  # fallback; some builds expect it
        "prompt": prompt,
    }


def make_libero_obs(env_obs, prompt):
    """LIBERO obs: external + wrist images (ext as `image`, wrist as `wrist_image`),
    8-D state [eef_pos(3), eef_rot_axisangle(3), gripper_qpos(2)]."""
    from scipy.spatial.transform import Rotation as R
    ext = _ext_image(env_obs)
    wrist = _wrist_image(env_obs)
    eef_pos = np.asarray(env_obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(env_obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)  # xyzw
    eef_rot = R.from_quat(eef_quat).as_rotvec().astype(np.float32)
    grip = np.asarray(env_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    state = np.concatenate([eef_pos, eef_rot, grip]).astype(np.float32)  # 8-D
    return {
        "observation/image": ext,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


OBS_BUILDERS = {
    "pi05_robocasa365": make_robocasa_obs,
    "pi05_droid": make_droid_obs,
    "pi05_libero": make_libero_obs,
}


# ---------------------------------------------------- per-model action adapters


def adapt_action_robocasa(action_chunk_step):
    """12-D layout B → 12-D PandaOmron env_action with the gripper/torso/base
    slot remap from robocasa-benchmark/openpi (robosuite robot.py:957 appends
    right_gripper after body_parts)."""
    a = np.zeros(12, dtype=np.float32)
    m = np.asarray(action_chunk_step, dtype=np.float32)
    if m.size < 12:
        # Defensive: pad if model emits fewer dims than expected.
        m_full = np.zeros(12, dtype=np.float32)
        m_full[:m.size] = m
        m = m_full
    a[0:6] = m[0:6]                # arm OSC_POSE
    a[6]   = float(m[10])          # torso ← base_motion[3]
    a[7:10] = m[7:10]              # base x/y/yaw ← base_motion[0:3]
    a[10]  = float(m[6])           # right_gripper ← gripper
    a[11]  = float(m[11])          # control_mode passthrough
    return np.clip(a, -1.0, 1.0)


def adapt_action_libero(action_chunk_step):
    """7-D libero [eef_pos(3), eef_rot(3), gripper(1)] → 12-D PandaOmron env.
    Arm slot is OSC_POSE so eef fits 0:6 directly. Force arm-only mode."""
    a = np.zeros(12, dtype=np.float32)
    m = np.asarray(action_chunk_step, dtype=np.float32)
    a[0:6] = m[:6]                 # arm OSC_POSE delta
    a[6]   = 0.0                   # torso
    a[7:10] = 0.0                  # base
    a[10]  = float(m[6])           # gripper passthrough (LIBERO already in ±1)
    a[11]  = -1.0                  # control_mode = arm-only
    return np.clip(a, -1.0, 1.0)


def adapt_action_droid(action_chunk_step):
    """8-D DROID [joint_vel(7), gripper(1)] → 12-D PandaOmron env. Arm slot is
    OSC_POSE not JOINT_VELOCITY — joint velocities don't translate. We stuff
    the first 6 components into the OSC_POSE slot purely so the model's output
    is non-trivially ingested; expected ~0% success on robocasa kitchens.
    Gripper: DROID's [0,1] convention → robosuite ±1 with 0.5 binarization."""
    a = np.zeros(12, dtype=np.float32)
    m = np.asarray(action_chunk_step, dtype=np.float32)
    a[0:6] = m[:6]                 # joint vels jammed into OSC_POSE slot
    a[6]   = 0.0                   # torso
    a[7:10] = 0.0                  # base
    a[10]  = 1.0 if m[7] > 0.5 else -1.0  # gripper [0,1] → ±1
    a[11]  = -1.0                  # control_mode = arm-only
    return np.clip(a, -1.0, 1.0)


ACTION_ADAPTERS = {
    "pi05_robocasa365": adapt_action_robocasa,
    "pi05_libero": adapt_action_libero,
    "pi05_droid": adapt_action_droid,
}


# ---------------------------------------------------- env construction --------


def build_env(task: str, seed: int):
    import robocasa  # noqa: F401  registers kitchen envs
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    ctrl = load_composite_controller_config(robot="PandaOmron")
    cam_names = ["robot0_agentview_left", "robot0_eye_in_hand"]
    return robosuite.make(
        env_name=task, robots="PandaOmron", controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cam_names,
        camera_heights=[TILE, TILE],
        camera_widths=[TILE, TILE],
        ignore_done=True, control_freq=20, seed=seed,
    )


def get_episode_prompt(env, fallback: str = "complete the task") -> str:
    try:
        meta = env.get_ep_meta()
        lang = meta.get("lang") if isinstance(meta, dict) else None
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
    except Exception:
        pass
    return fallback


# ---------------------------------------------------- rollout -----------------


def run_episode(env, policy, model: str, prompt: str, max_steps: int):
    obs = env.reset()
    prompt = get_episode_prompt(env, fallback=prompt)
    obs_builder = OBS_BUILDERS[model]
    action_adapter = ACTION_ADAPTERS[model]

    last_attn_stack = None
    attn_snaps, frames = [], []
    eef_traj, grip_traj, reward_traj, action_taken = [], [], [], []
    pred_chunks = []
    chunk = None
    success = False

    ext_obs_key = "robot0_agentview_left_image" if "robot0_agentview_left_image" in obs else "agentview_image"

    for step in range(max_steps):
        if step % OPEN_LOOP_HORIZON == 0:
            policy_obs = obs_builder(obs, prompt)
            try:
                result = policy.infer(policy_obs)
            except Exception as e:
                raise RuntimeError(f"infer failed at step {step}: {e}") from e
            chunk = np.asarray(result["actions"], dtype=np.float32)
            pred_chunks.append(chunk.copy())
            attn = result.get("text_to_img_attn")
            if attn is not None:
                last_attn_stack = np.asarray(attn, dtype=np.float32)

        env_action = action_adapter(chunk[step % OPEN_LOOP_HORIZON])
        obs, reward, _done, _info = env.step(env_action)

        eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        grip_traj.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
        reward_traj.append(float(reward))
        action_taken.append(env_action.copy())
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
        "prompt": prompt,
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
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.model / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[eval365] model={args.model} task={args.task} eps={args.episodes} "
          f"seed_base={args.seed_base} ws={args.host}:{args.port}")

    summary = []
    for ep in range(args.episodes):
        seed = args.seed_base + ep
        out_path = out_dir / f"ep_{ep:03d}.npz"
        if out_path.exists():
            print(f"[eval365] ep {ep:03d} already done, skipping")
            continue
        np.random.seed(seed); random.seed(seed)
        try:
            env = build_env(args.task, seed=seed)
        except Exception as e:
            print(f"[eval365] ep {ep:03d} build_env ERROR: {e}\n{traceback.format_exc()}")
            continue

        t0 = time.time()
        try:
            data = run_episode(env, policy, args.model, prompt="complete the task",
                               max_steps=args.max_steps)
        except Exception as e:
            print(f"[eval365] ep {ep:03d} ERROR: {e}\n{traceback.format_exc()}")
            env.close()
            if "infer failed" in str(e) or "timeout" in str(e).lower():
                break
            continue
        env.close()

        data["model"] = args.model
        data["task"] = args.task
        data["seed"] = int(seed)
        np.savez_compressed(out_path, **data)
        summary.append({
            "ep": ep, "seed": seed,
            "success": bool(data["success"]),
            "final_reward": float(data["final_reward"]),
            "max_reward": float(data["max_reward"]),
            "steps": int(data["steps"]),
            "prompt": data["prompt"],
        })
        print(f"[eval365] ep {ep:03d} seed={seed} succ={data['success']} "
              f"r_max={data['max_reward']:.3f} ({time.time()-t0:.1f}s) "
              f"prompt={data['prompt']!r}")

    (out_dir / "_summary.json").write_text(json.dumps({
        "model": args.model, "task": args.task,
        "episodes": summary,
    }, indent=2))
    print(f"[eval365] done {args.model}/{args.task}")


if __name__ == "__main__":
    main()
