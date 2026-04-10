·"""Example: capture suffix attention from the first denoising step on duck frame 40.

Usage:
    uv run python viz/example_suffix_attn.py
    uv run python viz/example_suffix_attn.py --capture-steps 3   # average first 3 NFE steps
    uv run python viz/example_suffix_attn.py --frame 0 --checkpoint checkpoints/viz/pi0_droid_pytorch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import h5py

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "viz")]:
    if p not in sys.path:
        sys.path.insert(0, p)

DUCK_DIR        = ROOT / "data" / "example" / "duck"
DEFAULT_CKPT    = str(ROOT / "checkpoints" / "viz" / "pi05_droid_pytorch")
OPEN_LOOP_HORIZON = 8


# ── Load one duck frame ───────────────────────────────────────────────────────

def load_duck_frame(frame_idx: int) -> dict:
    """Load a single frame from the duck episode (frames/ layout, no instruction.txt)."""
    frames_dir = DUCK_DIR / "frames"

    ext_img  = np.array(Image.open(frames_dir / "varied_camera_2" / f"{frame_idx:05d}.jpg").convert("RGB"))
    hand_img = np.array(Image.open(frames_dir / "hand_camera"     / f"{frame_idx:05d}.jpg").convert("RGB"))

    with h5py.File(DUCK_DIR / "trajectory.h5", "r") as f:
        joint_pos   = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gripper_pos = f["observation/robot_state/gripper_position"][frame_idx : frame_idx + 1].astype(np.float64)

        # GT actions
        traj_len = f["action/joint_velocity"].shape[0]
        end = min(frame_idx + OPEN_LOOP_HORIZON, traj_len)
        n   = end - frame_idx
        jv  = f["action/joint_velocity"][frame_idx:end].astype(np.float32)   # (n, 7)
        gp  = f["action/gripper_position"][frame_idx:end].astype(np.float32) # (n,)
        gt  = np.concatenate([jv, gp[:, None]], axis=1)                       # (n, 8)
        if n < OPEN_LOOP_HORIZON:
            pad = np.full((OPEN_LOOP_HORIZON - n, 8), np.nan, dtype=np.float32)
            gt  = np.concatenate([gt, pad], axis=0)

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left":      hand_img,
        "observation/joint_position":        joint_pos,
        "observation/gripper_position":      gripper_pos,
        "prompt":                            "pick up the duck",
        "gt_action":                         gt,
    }


# ── Load policy ───────────────────────────────────────────────────────────────

def load_policy(checkpoint: str, device: str = "cuda:0"):
    from openpi.training import config as _cfg
    from openpi.policies import policy_config as _pc

    raw = Path(checkpoint).name
    candidate = raw.removesuffix("_pytorch")
    # Try progressively shorter names
    from openpi.training import config as cfg_mod
    while candidate:
        try:
            cfg_mod.get_config(candidate)
            break
        except (ValueError, KeyError):
            pass
        idx = candidate.rfind("_")
        if idx == -1:
            candidate = "pi05_droid"
            break
        candidate = candidate[:idx]

    config = _cfg.get_config(candidate)
    print(f"[policy] config={candidate}  checkpoint={checkpoint}  device={device}")
    return _policy_config.create_trained_policy(config, checkpoint, pytorch_device=device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame",         type=int, default=40,            help="Frame index in duck episode")
    parser.add_argument("--checkpoint",    default=DEFAULT_CKPT,            help="Policy checkpoint dir")
    parser.add_argument("--capture-steps", type=int, default=3,             help="NFE steps to capture (1=first only, 999=all)")
    parser.add_argument("--device",        default="cuda:0")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[data] duck frame {args.frame}")
    example = load_duck_frame(args.frame)
    print(f"  exterior : {example['observation/exterior_image_1_left'].shape}")
    print(f"  wrist    : {example['observation/wrist_image_left'].shape}")
    print(f"  prompt   : '{example['prompt']}'")
    print(f"  gt_action: {example['gt_action'].shape}  (first step: {example['gt_action'][0]})")

    # ── Load policy ──────────────────────────────────────────────────────────
    from openpi.training import config as _cfg
    from openpi.policies import policy_config as _pc

    raw = Path(args.checkpoint).name
    candidate = raw.removesuffix("_pytorch")
    while candidate:
        try:
            _cfg.get_config(candidate)
            break
        except (ValueError, KeyError):
            pass
        idx = candidate.rfind("_")
        if idx == -1:
            candidate = "pi05_droid"
            break
        candidate = candidate[:idx]

    config = _cfg.get_config(candidate)
    print(f"\n[policy] config={candidate}  checkpoint={args.checkpoint}")
    policy = _pc.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)

    # ── Inference with attention capture ────────────────────────────────────
    from openpi.models_pytorch import gemma_pytorch as _gpt

    print(f"\n[attn] capture_steps={args.capture_steps}  (capturing first {args.capture_steps} NFE step(s))")

    _gpt.enable_attn_buffer()
    _gpt.enable_suffix_attn_buffer(capture_steps=args.capture_steps)
    try:
        result  = policy.infer(example)
        prefix_buf = _gpt.get_attn_buffer()
        suffix_buf = _gpt.get_suffix_attn_buffer()
        nfe_seen   = _gpt.get_suffix_attn_buffer_step_count()
    finally:
        _gpt.clear_attn_buffer()
        _gpt.clear_suffix_attn_buffer()

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n[result]")
    print(f"  pred_action shape : {result['actions'].shape}")
    print(f"  pred_action[0]    : {result['actions'][0]}")
    print(f"  NFE steps seen    : {nfe_seen}  (captured first {args.capture_steps})")

    if prefix_buf:
        layer0 = prefix_buf[0]           # (1, n_heads, seq, seq)
        print(f"\n[prefix] layers={len(prefix_buf)}  shape per layer: {layer0.shape}")
        # action-token → image attention slice (wrist = cols 256:512)
        # prefix forward: rows are text/image tokens, cols are full sequence

    if suffix_buf:
        layer0 = suffix_buf[0]           # (1, n_heads, 8, k)
        print(f"[suffix] layers={len(suffix_buf)}  shape per layer: {layer0.shape}")
        # cols 0:256   → exterior image patches
        # cols 256:512 → wrist image patches
        a2img_wrist = layer0[0, :, :, 256:512]  # (n_heads, 8_steps, 256_patches)
        print(f"  action→wrist attn  mean={a2img_wrist.mean():.4f}  max={a2img_wrist.max():.4f}")

        # Which wrist patch does each action step attend to most?
        layer_mid = suffix_buf[len(suffix_buf) // 2][0]  # mid-layer, (n_heads, 8, k)
        wrist = layer_mid.mean(axis=0)[:, 256:512]       # avg over heads → (8, 256)
        top_patch = wrist.argmax(axis=1)                  # (8,)  patch index 0-255
        top_row   = top_patch // 16
        top_col   = top_patch % 16
        print(f"\n  Mid-layer top wrist patch per action step (16×16 grid):")
        for step, (r, c) in enumerate(zip(top_row, top_col)):
            print(f"    step {step}: patch ({r:2d},{c:2d})  attn={wrist[step, top_patch[step]]:.4f}")


if __name__ == "__main__":
    main()
