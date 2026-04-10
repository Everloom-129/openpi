"""Visualize how action→image and action→text attention changes over denoising steps.

Runs one inference on duck frame 40 (or any episode/frame), captures suffix
attention at every NFE step, then produces a multi-panel figure:

  Panel A  — Attention mass breakdown (ext_img / wrist_img / text / action tokens)
             per denoising step, averaged over heads and action steps.
             One subplot per selected layer.

  Panel B  — Attention mass over denoising steps as a stacked-area chart
             (single representative layer), showing how the budget shifts.

  Panel C  — Wrist-image heatmap (16×16 patches, averaged over heads) at
             first / mid / last denoising step for the representative layer.

Usage:
    uv run python viz/plot_denoising_attn.py
    uv run python viz/plot_denoising_attn.py --frame 0 --checkpoint checkpoints/viz/pi0_droid_pytorch
    uv run python viz/plot_denoising_attn.py --layers 0 8 17 --out attn_denoising.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "viz")]:
    if p not in sys.path:
        sys.path.insert(0, p)

DUCK_DIR          = ROOT / "data" / "example" / "duck"
DEFAULT_CKPT      = str(ROOT / "checkpoints" / "viz" / "pi05_droid_pytorch")
OPEN_LOOP_HORIZON = 8

# Token layout (same for π₀ and π₀.₅)
EXT_START   = 0
EXT_END     = 256
WRIST_START = 256
WRIST_END   = 512
PAD_END     = 768   # zero-padding region, skip in analysis
TEXT_START  = 768


# ── Data loading ──────────────────────────────────────────────────────────────

def load_duck_frame(frame_idx: int) -> dict:
    frames_dir = DUCK_DIR / "frames"
    ext_img  = np.array(Image.open(frames_dir / "varied_camera_2" / f"{frame_idx:05d}.jpg").convert("RGB"))
    hand_img = np.array(Image.open(frames_dir / "hand_camera"     / f"{frame_idx:05d}.jpg").convert("RGB"))

    with h5py.File(DUCK_DIR / "trajectory.h5", "r") as f:
        joint_pos   = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gripper_pos = f["observation/robot_state/gripper_position"][frame_idx : frame_idx + 1].astype(np.float64)
        traj_len    = f["action/joint_velocity"].shape[0]
        end         = min(frame_idx + OPEN_LOOP_HORIZON, traj_len)
        n           = end - frame_idx
        jv          = f["action/joint_velocity"][frame_idx:end].astype(np.float32)
        gp          = f["action/gripper_position"][frame_idx:end].astype(np.float32)
        gt          = np.concatenate([jv, gp[:, None]], axis=1)
        if n < OPEN_LOOP_HORIZON:
            gt = np.concatenate([gt, np.full((OPEN_LOOP_HORIZON - n, 8), np.nan, dtype=np.float32)])

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left":      hand_img,
        "observation/joint_position":        joint_pos,
        "observation/gripper_position":      gripper_pos,
        "prompt":                            "pick up the duck",
        "gt_action":                         gt,
    }


# ── Policy loader ─────────────────────────────────────────────────────────────

def load_policy(checkpoint: str, device: str):
    from openpi.training import config as _cfg
    from openpi.policies import policy_config as _pc

    raw = Path(checkpoint).name
    candidate = raw.removesuffix("_pytorch")
    while candidate:
        try:
            _cfg.get_config(candidate)
            break
        except (ValueError, KeyError):
            idx = candidate.rfind("_")
            candidate = candidate[:idx] if idx != -1 else "pi05_droid"
            if idx == -1:
                break

    config = _cfg.get_config(candidate)
    print(f"[policy] config={candidate}  checkpoint={checkpoint}  device={device}")
    return _pc.create_trained_policy(config, checkpoint, pytorch_device=device)


# ── Attention extraction ──────────────────────────────────────────────────────

def extract_group_masses(
    steps_buf: list[dict[int, np.ndarray]],
    layer_idx: int,
    n_text: int,
) -> dict[str, np.ndarray]:
    """For one layer, compute per-step attention mass for each token group.

    Returns dict of arrays shaped (n_steps,), averaged over heads and action steps.
    Groups: ext_img, wrist_img, text, action_self.
    """
    n_steps = len(steps_buf)
    groups = {"ext_img": [], "wrist_img": [], "text": [], "action_self": []}

    for step_dict in steps_buf:
        attn = step_dict[layer_idx][0]          # (n_heads, 8, seq_len)
        seq_len = attn.shape[-1]
        text_end   = TEXT_START + n_text
        action_start = text_end

        # Mean over heads and action steps → (seq_len,)
        mean_attn = attn.mean(axis=(0, 1))

        groups["ext_img"].append(   mean_attn[EXT_START:EXT_END].sum())
        groups["wrist_img"].append( mean_attn[WRIST_START:WRIST_END].sum())
        groups["text"].append(      mean_attn[TEXT_START:text_end].sum())
        groups["action_self"].append(mean_attn[action_start:].sum())

    return {k: np.array(v) for k, v in groups.items()}


def extract_wrist_heatmap(
    step_dict: dict[int, np.ndarray],
    layer_idx: int,
) -> np.ndarray:
    """Return 16×16 wrist-patch attention heatmap, averaged over heads and action steps."""
    attn = step_dict[layer_idx][0]          # (n_heads, 8, seq_len)
    wrist = attn[:, :, WRIST_START:WRIST_END]  # (n_heads, 8, 256)
    mean  = wrist.mean(axis=(0, 1))             # (256,)
    return mean.reshape(16, 16)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "ext_img":    "#4e79a7",
    "wrist_img":  "#f28e2b",
    "text":       "#59a14f",
    "action_self":"#e15759",
}
LABELS = {
    "ext_img":    "Ext camera (0:256)",
    "wrist_img":  "Wrist camera (256:512)",
    "text":       "Text tokens",
    "action_self":"Action tokens (self)",
}


def plot(
    steps_buf: list[dict[int, np.ndarray]],
    images: dict,
    n_text: int,
    layers: list[int],
    rep_layer: int,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_steps = len(steps_buf)
    xs = np.arange(n_steps)
    groups_order = ["ext_img", "wrist_img", "text", "action_self"]

    n_layers = len(layers)
    # Layout: top row = per-layer line plots, middle = stacked area, bottom = heatmaps
    fig = plt.figure(figsize=(max(14, 4 * n_layers), 16))
    gs  = gridspec.GridSpec(3, max(n_layers, 3), figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: per-layer line plots ────────────────────────────────────────
    for col, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, col])
        masses = extract_group_masses(steps_buf, layer, n_text)
        for g in groups_order:
            ax.plot(xs, masses[g], color=COLORS[g], label=LABELS[g], linewidth=2)
        ax.set_title(f"Layer {layer}", fontsize=11)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Attention mass")
        ax.set_xlim(0, n_steps - 1)
        ax.set_ylim(0)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

    # ── Panel B: stacked area (representative layer) ─────────────────────────
    ax_stack = fig.add_subplot(gs[1, :])
    masses = extract_group_masses(steps_buf, rep_layer, n_text)
    stack = np.stack([masses[g] for g in groups_order], axis=0)   # (4, n_steps)
    # Normalize to sum=1 per step
    total = stack.sum(axis=0, keepdims=True)
    stack_norm = stack / np.where(total > 0, total, 1)
    ax_stack.stackplot(
        xs,
        stack_norm,
        labels=[LABELS[g] for g in groups_order],
        colors=[COLORS[g] for g in groups_order],
        alpha=0.85,
    )
    ax_stack.set_title(f"Normalized attention budget over denoising steps — layer {rep_layer}", fontsize=12)
    ax_stack.set_xlabel("Denoising step  (0 = most noisy, last = clean action)")
    ax_stack.set_ylabel("Fraction of attention")
    ax_stack.set_xlim(0, n_steps - 1)
    ax_stack.set_ylim(0, 1)
    ax_stack.legend(loc="upper left", fontsize=9)
    ax_stack.grid(True, alpha=0.2, axis="y")

    # ── Panel C: wrist heatmaps at first / mid / last step ──────────────────
    step_indices = [0, n_steps // 2, n_steps - 1]
    step_labels  = ["First step (most noisy)", f"Mid step ({n_steps // 2})", "Last step (clean)"]

    # Wrist image for context
    wrist_np = images.get("wrist")
    vmax = max(
        extract_wrist_heatmap(steps_buf[s], rep_layer).max()
        for s in step_indices
    )

    for col, (s_idx, s_label) in enumerate(zip(step_indices, step_labels)):
        ax = fig.add_subplot(gs[2, col])
        hmap = extract_wrist_heatmap(steps_buf[s_idx], rep_layer)
        im = ax.imshow(hmap, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Wrist attn — {s_label}\n(layer {rep_layer}, mean over heads+steps)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Optionally show wrist image alongside first heatmap if more columns exist
    if max(n_layers, 3) > 3 and wrist_np is not None:
        ax_img = fig.add_subplot(gs[2, 3])
        ax_img.imshow(wrist_np)
        ax_img.set_title("Wrist camera image", fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

    fig.suptitle(
        f"Action→token attention over {n_steps} denoising steps  "
        f"(duck frame, prompt: 'pick up the duck')",
        fontsize=13, y=0.98,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot] saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame",      type=int, default=40)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--layers",     type=int, nargs="+", default=[0, 8, 17],
                        help="Layers to show in Panel A")
    parser.add_argument("--rep-layer",  type=int, default=8,
                        help="Representative layer for stacked area + heatmaps")
    parser.add_argument("--out",        default="attn_denoising.png")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[data] duck frame {args.frame}")
    example = load_duck_frame(args.frame)

    # ── Load policy ──────────────────────────────────────────────────────────
    policy = load_policy(args.checkpoint, args.device)

    # ── Infer with per-step capture ──────────────────────────────────────────
    from openpi.models_pytorch import gemma_pytorch as _gpt

    print("[attn] capturing all NFE steps …")
    _gpt.enable_suffix_attn_steps_buffer()
    try:
        result = policy.infer(example)
        steps_buf = _gpt.get_suffix_attn_steps_buffer()
    finally:
        _gpt.clear_suffix_attn_steps_buffer()

    n_steps = len(steps_buf)
    print(f"[attn] {n_steps} NFE steps captured")

    if n_steps == 0:
        print("[error] no steps captured — suffix forward may not have run")
        return

    # Detect sequence length and n_text from first step, first layer
    first_layer = min(steps_buf[0].keys())
    seq_len = steps_buf[0][first_layer].shape[-1]        # k columns
    n_text  = seq_len - TEXT_START - OPEN_LOOP_HORIZON   # approximate; action tokens at end
    # More precisely: n_text = seq_len - TEXT_START - n_action_tokens
    # n_action_tokens ≤ 8, but seq_len could be TEXT_START + n_text + 8
    # Use a safe lower bound: anything after TEXT_START+1 and before the last 8 cols
    n_text = max(1, seq_len - TEXT_START - OPEN_LOOP_HORIZON)

    print(f"[info] seq_len={seq_len}  n_text≈{n_text}  layers captured={sorted(steps_buf[0].keys())}")

    # Print summary stats
    print("\n[stats] Attention mass at first vs last denoising step (layer {rep}):".format(rep=args.rep_layer))
    first_m = extract_group_masses(steps_buf, args.rep_layer, n_text)
    for g in ["ext_img", "wrist_img", "text", "action_self"]:
        print(f"  {g:15s}  step0={first_m[g][0]:.4f}  step-1={first_m[g][-1]:.4f}  "
              f"delta={first_m[g][-1]-first_m[g][0]:+.4f}")

    # ── Resize wrist for display ──────────────────────────────────────────────
    wrist_img = np.array(
        Image.fromarray(example["observation/wrist_image_left"]).resize((224, 224))
    )
    images = {"wrist": wrist_img}

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot(
        steps_buf  = steps_buf,
        images     = images,
        n_text     = n_text,
        layers     = args.layers,
        rep_layer  = args.rep_layer,
        out_path   = args.out,
    )


if __name__ == "__main__":
    main()
