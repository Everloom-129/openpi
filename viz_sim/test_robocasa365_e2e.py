"""End-to-end smoke test for pi05_robocasa365 on real robocasa kitchen tasks.

Wires the official `gym.make("robocasa/<TASK>", split=...)` wrapper from
`third_party/robocasa/robocasa/wrappers/gym_wrapper.py` so the env exposes:

  * dict obs with keys `state.{...}` (eef-relative + base + gripper),
    `video.{robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand}`,
    `annotation.human.task_description` (the templated kitchen language).
  * dict action space with `action.{end_effector_position,end_effector_rotation,
    gripper_close,base_motion,control_mode}` — the wrapper's `unmap_action` +
    `step` already do the painful `right_gripper`-after-`body_parts` slot
    remap that the raw `robosuite.make` path had to handle by hand.

Verifies the policy server returns `text_to_img_attn` (shape `(L, H, 512)`)
on a real-task obs by saving:
  * `results/test_robocasa365_e2e/{task}/attn_grid.png` — layer-7 head-mean
    16×16 grid for ext + wrist on a representative middle frame.
  * `results/test_robocasa365_e2e/{task}/rollout.webp` — animated overlay
    video (ext + wrist | layer-7 head-mean), one frame per env step.
  * `results/test_robocasa365_e2e/{task}/summary.json` — chunk shapes,
    success, max_reward, prompt, attention metadata.

Run sequence (assumes server already up on `--port`):

    bash viz_sim/run_pi0_policy_server.sh CONFIG=pi05_robocasa365 GPU=2 &
    # wait until log shows "websockets.server:server listening"
    /home/edward/miniconda3/envs/robocasa_sim/bin/python \
        viz_sim/test_robocasa365_e2e.py --task PickPlaceCounterToCabinet
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "results" / "test_robocasa365_e2e"

OPEN_LOOP_HORIZON = 8
TILE = 224
DISPLAY_LAYER = 7
HEADER = 28
FPS = 8


# --------------------------------------------------------------- adapters ----


def _state_16d(obs: dict) -> np.ndarray:
    """Pack gym wrapper's `state.*` into the 16-D upstream-order state the
    server's RobocasaInputs expects: [eef_pos_rel(3), eef_rot_rel(4 xyzw),
    base_pos(3), base_rot(4), gripper_qpos(2)].
    """
    return np.concatenate([
        np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_position"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.base_rotation"], dtype=np.float32).reshape(-1),
        np.asarray(obs["state.gripper_qpos"], dtype=np.float32).reshape(-1),
    ]).astype(np.float32)


def _ext(obs: dict) -> np.ndarray:
    return np.ascontiguousarray(obs["video.robot0_agentview_left"])


def _wrist(obs: dict) -> np.ndarray:
    return np.ascontiguousarray(obs["video.robot0_eye_in_hand"])


def model_to_gym_action(m12: np.ndarray) -> dict:
    """Layout-B 12-D model output → wrapper's 5-key dict action.

    Layout B (matches robocasa training): [eef_pos(3), eef_rot(3), gripper(1),
    base_motion(4)=base_x,base_y,base_yaw,torso_z, control_mode(1)].
    """
    m = np.asarray(m12, dtype=np.float32).reshape(-1)
    if m.size < 12:
        pad = np.zeros(12, dtype=np.float32); pad[:m.size] = m; m = pad
    return {
        "action.end_effector_position": np.clip(m[0:3], -1, 1),
        "action.end_effector_rotation": np.clip(m[3:6], -1, 1),
        "action.gripper_close":         np.clip(m[6:7], -1, 1),
        "action.base_motion":           np.clip(m[7:11], -1, 1),
        "action.control_mode":          np.clip(m[11:12], -1, 1),
    }


# --------------------------------------------------------------- viz utils ---


def overlay_attn(img_rgb: np.ndarray, attn256: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    grid = attn256.reshape(16, 16).astype(np.float32)
    up = cv2.resize(grid, img_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_rgb, alpha, 0)


def label_strip(text: str, width: int, height: int = HEADER,
                bg=(40, 40, 40), font_scale: float = 0.5) -> np.ndarray:
    s = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(s, text, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return s


def save_attn_grid_png(out_path: pathlib.Path, ext_img: np.ndarray, wrist_img: np.ndarray,
                       attn512: np.ndarray, prompt: str, step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_ext = attn512[:256].reshape(16, 16)
    a_wri = attn512[256:].reshape(16, 16)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.suptitle(f"pi05_robocasa365 layer={DISPLAY_LAYER} head=mean | step={step}\n"
                 f'prompt: "{prompt}"', fontsize=10)
    axes[0, 0].imshow(ext_img); axes[0, 0].set_title("ext frame"); axes[0, 0].axis("off")
    axes[0, 1].imshow(a_ext, cmap="viridis"); axes[0, 1].set_title("ext attn (16×16)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(overlay_attn(ext_img, attn512[:256])); axes[0, 2].set_title("ext overlay"); axes[0, 2].axis("off")
    axes[1, 0].imshow(wrist_img); axes[1, 0].set_title("wrist frame"); axes[1, 0].axis("off")
    axes[1, 1].imshow(a_wri, cmap="viridis"); axes[1, 1].set_title("wrist attn (16×16)"); axes[1, 1].axis("off")
    axes[1, 2].imshow(overlay_attn(wrist_img, attn512[256:])); axes[1, 2].set_title("wrist overlay"); axes[1, 2].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_rollout_webp(out_path: pathlib.Path, frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
                      task: str, prompt: str) -> None:
    if not frames:
        return
    W = TILE * 2
    top1 = label_strip(f"{task}  |  succ-test  |  layer={DISPLAY_LAYER} head=mean", W,
                       bg=(15, 15, 15), font_scale=0.5)
    top2 = label_strip(f'prompt: "{prompt}"', W, bg=(30, 30, 30), font_scale=0.45)
    col_lab = np.hstack([
        label_strip("ext", TILE, bg=(50, 50, 50), font_scale=0.5),
        label_strip("wrist", TILE, bg=(50, 50, 50), font_scale=0.5),
    ])

    pages = []
    for step, ext_img, wri_img, attn512 in frames:
        ext_o = overlay_attn(ext_img, attn512[:256])
        wri_o = overlay_attn(wri_img, attn512[256:])
        body = np.hstack([ext_o, wri_o])
        footer = label_strip(f"sim_step={step}", W, bg=(20, 20, 20))
        full = np.vstack([top1, top2, col_lab, body, footer])
        pages.append(Image.fromarray(full))
    duration = int(round(1000 / FPS))
    pages[0].save(out_path, format="WEBP", save_all=True, append_images=pages[1:],
                  duration=duration, loop=0, method=6, quality=82)


# --------------------------------------------------------------- main --------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToCabinet",
                    help="robocasa kitchen env (no 'robocasa/' prefix needed)")
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- env (gym wrapper, real kitchen task) -------------------------------
    import gymnasium as gym
    import robocasa  # noqa: F401  registers robocasa/<TASK> ids

    env_id = f"robocasa/{args.task}"
    print(f"[e2e] gym.make({env_id}, split={args.split}, seed={args.seed}, 224×224)")
    env = gym.make(env_id, split=args.split, seed=args.seed,
                   camera_widths=224, camera_heights=224)
    obs, _ = env.reset(seed=args.seed)
    prompt = obs.get("annotation.human.task_description", "complete the task")
    print(f"[e2e] prompt: {prompt!r}")

    # --- policy ---------------------------------------------------------------
    sys.path.insert(0, str(REPO / "packages" / "openpi-client" / "src"))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy = WebsocketClientPolicy(host=args.host, port=args.port)

    # --- rollout --------------------------------------------------------------
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    pred_chunks: list[np.ndarray] = []
    attn_meta = None
    chunk = None
    last_attn = None
    success = False
    max_reward = 0.0

    for step in range(args.max_steps):
        if step % OPEN_LOOP_HORIZON == 0:
            policy_obs = {
                "observation/image":         _ext(obs),
                "observation/wrist_image":   _wrist(obs),
                "observation/state":         _state_16d(obs),
                "prompt":                    prompt,
            }
            t0 = time.time()
            result = policy.infer(policy_obs)
            dt = time.time() - t0
            chunk = np.asarray(result["actions"], dtype=np.float32)
            pred_chunks.append(chunk.copy())
            attn = result.get("text_to_img_attn")
            if attn is not None:
                last_attn = np.asarray(attn, dtype=np.float32)  # (L, H, 512)
                attn_meta = result.get("text_to_img_meta")
            print(f"[e2e] step {step:03d}: infer {dt*1000:.0f}ms  "
                  f"chunk={chunk.shape}  attn={None if last_attn is None else last_attn.shape}")

        a = chunk[step % OPEN_LOOP_HORIZON]
        action_dict = model_to_gym_action(a)
        obs, reward, terminated, truncated, info = env.step(action_dict)
        max_reward = max(max_reward, float(reward))
        if info.get("success") or reward > 0.5:
            success = True

        if last_attn is not None:
            attn_l7_mean = last_attn[DISPLAY_LAYER].mean(axis=0)  # (512,)
            frames.append((step, _ext(obs).copy(), _wrist(obs).copy(),
                           attn_l7_mean.astype(np.float32)))

        if terminated or truncated:
            break

    env.close()

    # --- save artifacts -------------------------------------------------------
    if not frames:
        print("[e2e] FAIL: no frames captured (server may not be returning attn)")
        return 2

    mid = len(frames) // 2
    step_m, ext_m, wri_m, attn_m = frames[mid]
    save_attn_grid_png(out_dir / "attn_grid.png", ext_m, wri_m, attn_m, prompt, step_m)
    save_rollout_webp(out_dir / "rollout.webp", frames, args.task, prompt)

    summary = {
        "task": args.task, "split": args.split, "seed": int(args.seed),
        "prompt": prompt, "max_steps_run": len(frames),
        "success": bool(success), "max_reward": float(max_reward),
        "n_chunks": len(pred_chunks),
        "chunk_shape": list(pred_chunks[0].shape) if pred_chunks else None,
        "attn_meta": attn_meta,
        "attn_grid_png": str(out_dir / "attn_grid.png"),
        "rollout_webp": str(out_dir / "rollout.webp"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"[e2e] OK — wrote {out_dir}")
    print(f"[e2e]   attn_grid.png  ({attn_m.shape if attn_m is not None else 'n/a'})")
    print(f"[e2e]   rollout.webp   ({len(frames)} frames @ {FPS} fps)")
    print(f"[e2e]   summary.json   succ={success} max_r={max_reward:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
