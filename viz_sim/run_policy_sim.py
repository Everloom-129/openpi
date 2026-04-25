"""Closed-loop pi0.5-DROID inference inside a robosuite/robocasa MuJoCo sim.

Architecture
------------
* This script runs in the `robocasa_sim` conda env (mujoco 3.3.1, numpy 2.x).
* The pi0.5 policy runs in the openpi `.venv` and is exposed by
  `viz_sim/run_policy_server.sh` as a websocket server on localhost:8000.
* We open a `Panda` arm in robosuite with the JOINT_VELOCITY arm controller
  so the env's action space is exactly [joint_vel × 7, gripper × 1] — the
  same 8-dim layout as the DROID checkpoint output. No remapping needed.
* Each step we resize the agent + wrist cameras to 224×224, build a DROID
  obs dict, send it to the policy, and replay the first OPEN_LOOP_HORIZON
  actions before re-querying.

Run (after the policy server is up):
    conda activate robocasa_sim
    python viz_sim/run_policy_sim.py --task Lift --prompt "pick up the cube"
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from openpi_client.websocket_client_policy import WebsocketClientPolicy

# pi0.5-DROID emits a chunk of 15 actions; the dashboard executes the first 8.
OPEN_LOOP_HORIZON = 8


def build_env(task: str, render_camera: str = "agentview"):
    """Create a robosuite env with JOINT_VELOCITY arm control on a Panda.

    Action layout: [arm_qvel(7), gripper(1)] — matches DROID 8-dim output.
    """
    controller_cfg = load_composite_controller_config(robot="Panda")
    # After loading, robosuite flattens body_parts.arms.right -> body_parts.right.
    controller_cfg["body_parts"]["right"] = {
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

    return robosuite.make(
        env_name=task,
        robots="Panda",
        controller_configs=controller_cfg,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        render_camera=render_camera,
        ignore_done=True,
        control_freq=20,
    )


def to_uint8_rgb(img) -> np.ndarray:
    """Robosuite returns float[0,1] BGR-ish images flipped vertically. Normalize."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    # Robosuite renders top-down flipped; the dashboard expects upright RGB.
    arr = np.ascontiguousarray(arr[::-1])
    if arr.shape[:2] != (224, 224):
        arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    return arr


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
    args = ap.parse_args()

    print(f"Connecting to policy at ws://{args.host}:{args.port} ...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected. Server metadata:", policy.get_server_metadata())

    env = build_env(args.task)
    print(f"Env action_dim={env.action_dim}  (expecting 8: 7 qvel + 1 gripper)")
    obs = env.reset()

    t0 = time.time()
    n_infer = 0
    for step in range(args.steps):
        # Query the policy for a fresh action chunk every `horizon` steps.
        if step % args.horizon == 0:
            droid_obs = make_droid_obs(obs, args.prompt)
            result = policy.infer(droid_obs)
            chunk = np.asarray(result["actions"])  # (N, 8)
            n_infer += 1

        action = chunk[step % args.horizon]
        # Defensive: pad/truncate to env.action_dim in case action_dim != 8.
        if action.shape[0] != env.action_dim:
            a = np.zeros(env.action_dim, dtype=np.float32)
            a[: min(len(action), env.action_dim)] = action[: env.action_dim]
            action = a

        obs, _reward, _done, _info = env.step(action)
        env.render()

        ext = to_uint8_rgb(obs["agentview_image"])
        wrist = to_uint8_rgb(obs["robot0_eye_in_hand_image"])
        cv2.imshow("ext", cv2.cvtColor(ext, cv2.COLOR_RGB2BGR))
        cv2.imshow("wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    dt = time.time() - t0
    print(f"Done. {args.steps} sim steps, {n_infer} policy queries in {dt:.1f}s.")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
