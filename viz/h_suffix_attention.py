"""H7.2 — Per-Action-Step Suffix Attention Visualization.

During denoising (Path C / joint forward), the 8 action tokens (suffix) attend
to image patches.  This script loads the joint attention maps saved by
`gemma_pytorch.py` and produces:

  1. Per-action-step heatmaps (8 rows × 2 cameras).
  2. A side-by-side comparison of suffix vs prefix attention for the same layer.
  3. Attention entropy per action step, compared against prefix text tokens.

Prerequisites:
  - Run inference at least once so `results/layers_joint/` is populated.
  - `results/layers_prefix/` must also exist for the prefix comparison.

Usage (standalone):
    python viz/h_suffix_attention.py
"""
from __future__ import annotations

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import jax.numpy as jnp
from openpi.shared import image_tools


# ── Token layout constants ────────────────────────────────────────────────────
NUM_IMAGE_TOKENS = 256          # 16×16 patches per camera
# Pi05/DROID uses 3 image slots: exterior, wrist, and a zero-padded dummy camera
# (right_wrist_0_rgb = np.zeros_like, image_mask=False).  All three occupy 256
# tokens each in the attention sequence, so text tokens start at 768, not 512.
NUM_CAMERAS_REAL = 2            # exterior + wrist (real pixel content)
NUM_CAMERAS_TOTAL = 3           # includes the masked zero-padding slot
TOTAL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_REAL   # 512  — real cameras (for vis)
ALL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_TOTAL    # 768  — full image region in sequence
TEXT_START_IDX = ALL_IMAGE_TOKENS                          # 768  — where text tokens begin
NUM_ACTION_TOKENS = 8           # suffix length


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_joint_attn(layer_idx: int, joint_dir: str = "results/layers_joint") -> np.ndarray | None:
    """Load joint attention map → [1, heads, seq_len, seq_len]."""
    path = os.path.join(joint_dir, f"attn_map_layer_{layer_idx}.npy")
    if not os.path.exists(path):
        print(f"[h_suffix_attention] Joint attention file not found: {path}")
        return None
    return np.load(path).astype(np.float32)


def load_prefix_attn(layer_idx: int, prefix_dir: str = "results/layers_prefix") -> np.ndarray | None:
    """Load prefix attention map → [heads, seq_len, seq_len]."""
    path = os.path.join(prefix_dir, f"attn_map_layer_{layer_idx}.npy")
    if not os.path.exists(path):
        return None
    attn = np.load(path)
    if attn.ndim == 4:
        attn = attn[0]
    return attn.astype(np.float32)


def load_prefix_len(joint_dir: str = "results/layers_joint") -> int | None:
    """Load the prefix_len metadata saved during the joint forward pass."""
    path = os.path.join(joint_dir, "prefix_len.npy")
    if not os.path.exists(path):
        print(f"[h_suffix_attention] prefix_len.npy not found in {joint_dir}.")
        return None
    return int(np.load(path)[0])


def load_joint_logits(layer_idx: int, joint_dir: str = "results/layers_joint") -> np.ndarray | None:
    """Load pre-softmax attention logits → [1, heads, seq_len, seq_len]."""
    path = os.path.join(joint_dir, f"attn_logits_layer_{layer_idx}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)


# ── Heatmap overlay ───────────────────────────────────────────────────────────

def overlay_heatmap(img_np: np.ndarray, heatmap_16x16: np.ndarray) -> np.ndarray:
    """Return blended RGB 224×224 overlay."""
    img_jax = jnp.array(img_np)
    img_224 = np.array(image_tools.resize_with_pad(img_jax, 224, 224)).astype(np.uint8)
    hmap = cv2.resize(heatmap_16x16.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    hmap_u8 = np.uint8(255 * hmap / (np.max(hmap) + 1e-8))
    hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_224, 0.6, hmap_color, 0.4, 0)


# ── Main suffix visualization ─────────────────────────────────────────────────

def visualize_suffix_attention(
    example: dict,
    layer_idx: int,
    joint_dir: str = "results/layers_joint",
    output_dir: str = "results",
    name: str = "default",
    head_agg: str = "max",   # "max" | "mean" | "best_var"
):
    """Visualize per-action-step suffix attention as an 8-row × 2-camera grid.

    Layout per row (one action step):
        [Ext raw]  [Ext suffix attn]  [Wrist raw]  [Wrist suffix attn]

    Args:
        example:   Dict with image arrays and "prompt" key.
        layer_idx: Transformer layer to visualize.
        joint_dir: Directory containing `attn_map_layer_{i}.npy` (joint forward).
        output_dir: Root output directory.
        name:      Sub-folder name.
        head_agg:  Head aggregation mode.
    """
    attn4d = load_joint_attn(layer_idx, joint_dir)
    if attn4d is None:
        return

    prefix_len = load_prefix_len(joint_dir)
    if prefix_len is None:
        return

    # attn4d: [1, heads, seq_full, seq_full]
    attn = attn4d[0]   # [heads, seq_full, seq_full]
    _, seq_full, _ = attn.shape

    # Suffix rows: action tokens attend to image patches
    # shape: [heads, NUM_ACTION_TOKENS, TOTAL_IMAGE_TOKENS]
    suffix_start = prefix_len
    suffix_end = prefix_len + NUM_ACTION_TOKENS
    if suffix_end > seq_full:
        print(
            f"[h_suffix_attention] suffix_end={suffix_end} > seq_full={seq_full}; "
            "joint attention not captured for this layer."
        )
        return

    suffix_attn = attn[:, suffix_start:suffix_end, :TOTAL_IMAGE_TOKENS]
    # [heads, 8, 512]

    # Head aggregation
    if head_agg == "max":
        step_attn = suffix_attn.max(axis=0)          # [8, 512]
    elif head_agg == "mean":
        step_attn = suffix_attn.mean(axis=0)         # [8, 512]
    elif head_agg == "best_var":
        # Choose the head with the highest spatial variance (most focused)
        variances = suffix_attn.var(axis=-1).mean(axis=-1)   # [heads]
        best_head = int(np.argmax(variances))
        step_attn = suffix_attn[best_head]           # [8, 512]
    else:
        raise ValueError(f"Unknown head_agg: {head_agg}")

    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]

    n_steps = NUM_ACTION_TOKENS
    fig, axes = plt.subplots(
        n_steps, 4,
        figsize=(16, 3.5 * n_steps),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )

    for step in range(n_steps):
        attn_step = step_attn[step]   # [512]
        ext_hmap = attn_step[:NUM_IMAGE_TOKENS].reshape(16, 16)
        wrist_hmap = attn_step[NUM_IMAGE_TOKENS:].reshape(16, 16)

        ext_overlay = overlay_heatmap(ext_img, ext_hmap)
        wrist_overlay = overlay_heatmap(wrist_img, wrist_hmap)

        # Entropy for this step (over all heads)
        step_ent = -np.sum(
            suffix_attn[:, step, :] * np.log(suffix_attn[:, step, :] + 1e-10), axis=-1
        ).mean()

        from openpi.shared import image_tools as _it
        ext_raw_224 = np.array(
            _it.resize_with_pad(jnp.array(ext_img), 224, 224)
        ).astype(np.uint8)
        wrist_raw_224 = np.array(
            _it.resize_with_pad(jnp.array(wrist_img), 224, 224)
        ).astype(np.uint8)

        axes[step, 0].imshow(ext_raw_224);    axes[step, 0].axis("off")
        axes[step, 1].imshow(ext_overlay);    axes[step, 1].axis("off")
        axes[step, 2].imshow(wrist_raw_224);  axes[step, 2].axis("off")
        axes[step, 3].imshow(wrist_overlay);  axes[step, 3].axis("off")

        step_label = f"Action step {step}  H={step_ent:.2f}"
        axes[step, 0].set_ylabel(step_label, fontsize=9, rotation=0, labelpad=90, va="center")

    axes[0, 0].set_title("Exterior (raw)", fontsize=9)
    axes[0, 1].set_title("Exterior (suffix attn)", fontsize=9)
    axes[0, 2].set_title("Wrist (raw)", fontsize=9)
    axes[0, 3].set_title("Wrist (suffix attn)", fontsize=9)

    plt.suptitle(
        f'H7.2 Suffix Attention  |  Layer {layer_idx}  |  head_agg={head_agg}\n'
        f'"{example["prompt"]}"',
        fontsize=12, y=1.01,
    )

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"h7_suffix_attn_L{layer_idx}_{head_agg}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_suffix_attention] Saved → {out_path}")


# ── Prefix vs Suffix comparison ───────────────────────────────────────────────

def compare_prefix_suffix(
    example: dict,
    layer_idx: int,
    joint_dir: str = "results/layers_joint",
    prefix_dir: str = "results/layers_prefix",
    output_dir: str = "results",
    name: str = "default",
):
    """Side-by-side: prefix text attention vs suffix action-token attention.

    Left column  = prefix (all text tokens pooled, max over heads)
    Right column = suffix (all action tokens pooled, max over heads)
    """
    attn4d = load_joint_attn(layer_idx, joint_dir)
    prefix_attn = load_prefix_attn(layer_idx, prefix_dir)
    prefix_len = load_prefix_len(joint_dir)

    if attn4d is None or prefix_len is None:
        return

    attn = attn4d[0]   # [heads, seq_full, seq_full]

    # Suffix: pool over 8 action steps, max over heads
    suffix_attn_img = attn[:, prefix_len:prefix_len + NUM_ACTION_TOKENS, :TOTAL_IMAGE_TOKENS]
    suffix_pooled = suffix_attn_img.max(axis=0).max(axis=0)  # [512]

    # Prefix: pool over text tokens, max over heads (from prefix-only pass)
    if prefix_attn is not None:
        _, plen, _ = prefix_attn.shape
        if plen > TEXT_START_IDX:
            text_attn_img = prefix_attn[:, TEXT_START_IDX:, :TOTAL_IMAGE_TOKENS]
            prefix_pooled = text_attn_img.max(axis=0).max(axis=0)  # [512]
        else:
            prefix_pooled = None
    else:
        prefix_pooled = None

    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]

    n_rows = 2 if prefix_pooled is not None else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.25})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    def _fill_row(row_idx, pooled, row_label):
        ext_hmap = pooled[:NUM_IMAGE_TOKENS].reshape(16, 16)
        wrist_hmap = pooled[NUM_IMAGE_TOKENS:].reshape(16, 16)
        ext_overlay = overlay_heatmap(ext_img, ext_hmap)
        wrist_overlay = overlay_heatmap(wrist_img, wrist_hmap)
        _it = image_tools
        ext_raw = np.array(_it.resize_with_pad(jnp.array(ext_img), 224, 224)).astype(np.uint8)
        wrist_raw = np.array(_it.resize_with_pad(jnp.array(wrist_img), 224, 224)).astype(np.uint8)
        axes[row_idx, 0].imshow(ext_raw);     axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(ext_overlay); axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(wrist_raw);   axes[row_idx, 2].axis("off")
        axes[row_idx, 3].imshow(wrist_overlay); axes[row_idx, 3].axis("off")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=10, rotation=0, labelpad=80, va="center")

    if prefix_pooled is not None:
        _fill_row(0, prefix_pooled, "Prefix\n(text tokens)")
        _fill_row(1, suffix_pooled, "Suffix\n(action tokens)")
    else:
        _fill_row(0, suffix_pooled, "Suffix\n(action tokens)")

    axes[0, 0].set_title("Exterior (raw)", fontsize=9)
    axes[0, 1].set_title("Exterior (attn)", fontsize=9)
    axes[0, 2].set_title("Wrist (raw)", fontsize=9)
    axes[0, 3].set_title("Wrist (attn)", fontsize=9)

    plt.suptitle(
        f'H7.2 Prefix vs Suffix Attention  |  Layer {layer_idx}\n"{example["prompt"]}"',
        fontsize=12, y=1.01,
    )
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"h7_prefix_vs_suffix_L{layer_idx}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_suffix_attention] Prefix-vs-Suffix saved → {out_path}")


# ── Entropy comparison: prefix text vs suffix action ─────────────────────────

def entropy_prefix_vs_suffix(
    example: dict,
    layers: list[int],
    joint_dir: str = "results/layers_joint",
    prefix_dir: str = "results/layers_prefix",
    output_dir: str = "results",
    name: str = "default",
):
    """Plot attention entropy for prefix text tokens vs suffix action tokens.

    H7.3 prediction: suffix action tokens should have lower entropy (more
    focused) in deep layers compared to prefix text tokens.
    """
    prefix_entropies: list[float] = []
    suffix_entropies: list[float] = []
    valid_layers: list[int] = []

    prefix_len_global = load_prefix_len(joint_dir)

    for layer_idx in layers:
        joint = load_joint_attn(layer_idx, joint_dir)
        prefix = load_prefix_attn(layer_idx, prefix_dir)

        if joint is None:
            continue

        attn = joint[0]   # [heads, seq_full, seq_full]
        valid_layers.append(layer_idx)

        # Suffix entropy: per-head entropy over image columns, mean over steps
        if prefix_len_global is not None:
            suf = attn[:, prefix_len_global:prefix_len_global + NUM_ACTION_TOKENS, :TOTAL_IMAGE_TOKENS]
            suf_ent = -np.sum(suf * np.log(suf + 1e-10), axis=-1).mean()
        else:
            suf_ent = float("nan")
        suffix_entropies.append(float(suf_ent))

        # Prefix entropy: text tokens from the prefix-only pass
        if prefix is not None:
            _, plen, _ = prefix.shape
            if plen > TEXT_START_IDX:
                txt = prefix[:, TEXT_START_IDX:, :TOTAL_IMAGE_TOKENS]
                pre_ent = -np.sum(txt * np.log(txt + 1e-10), axis=-1).mean()
            else:
                pre_ent = float("nan")
        else:
            pre_ent = float("nan")
        prefix_entropies.append(float(pre_ent))

    if not valid_layers:
        print("[h_suffix_attention] No joint attention files found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid_layers, prefix_entropies, marker="o", label="Prefix (text tokens)", color="steelblue")
    ax.plot(valid_layers, suffix_entropies, marker="s", label="Suffix (action tokens)", color="darkorange")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy (bits)")
    ax.set_title(
        f'H7.3 Entropy: Prefix vs Suffix  |  "{example["prompt"]}"'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h7_entropy_prefix_vs_suffix.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_suffix_attention] Entropy comparison saved → {out_path}")


# ── Pre-softmax logit analysis ────────────────────────────────────────────────

def visualize_logits_vs_softmax(
    example: dict,
    layer_idx: int,
    joint_dir: str = "results/layers_joint",
    output_dir: str = "results",
    name: str = "default",
):
    """Compare pre-softmax logits vs post-softmax weights for suffix tokens.

    H7.3: pre-softmax logits show stronger object-word binding.
    """
    logits4d = load_joint_logits(layer_idx, joint_dir)
    attn4d = load_joint_attn(layer_idx, joint_dir)
    prefix_len = load_prefix_len(joint_dir)

    if logits4d is None:
        print(f"[h_suffix_attention] No pre-softmax logits for layer {layer_idx}.")
        return
    if attn4d is None or prefix_len is None:
        return

    # logits4d: [1, heads, seq_full, seq_full]
    logits = logits4d[0]   # [heads, seq_full, seq_full]
    attn = attn4d[0]

    # Suffix rows → image columns
    suf_logits = logits[:, prefix_len:prefix_len + NUM_ACTION_TOKENS, :TOTAL_IMAGE_TOKENS]
    suf_attn = attn[:, prefix_len:prefix_len + NUM_ACTION_TOKENS, :TOTAL_IMAGE_TOKENS]

    # Pool: max over heads and steps
    logit_pooled = suf_logits.max(axis=0).max(axis=0)   # [512]
    attn_pooled = suf_attn.max(axis=0).max(axis=0)      # [512]

    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.25})

    def _fill_row(row_idx, pooled, row_label):
        ext_hmap = pooled[:NUM_IMAGE_TOKENS].reshape(16, 16)
        wrist_hmap = pooled[NUM_IMAGE_TOKENS:].reshape(16, 16)
        # Normalize for display (logits may be negative)
        ext_hmap = ext_hmap - ext_hmap.min()
        wrist_hmap = wrist_hmap - wrist_hmap.min()
        ext_overlay = overlay_heatmap(ext_img, ext_hmap)
        wrist_overlay = overlay_heatmap(wrist_img, wrist_hmap)
        _it = image_tools
        ext_raw = np.array(_it.resize_with_pad(jnp.array(ext_img), 224, 224)).astype(np.uint8)
        wrist_raw = np.array(_it.resize_with_pad(jnp.array(wrist_img), 224, 224)).astype(np.uint8)
        axes[row_idx, 0].imshow(ext_raw);     axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(ext_overlay); axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(wrist_raw);   axes[row_idx, 2].axis("off")
        axes[row_idx, 3].imshow(wrist_overlay); axes[row_idx, 3].axis("off")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=10, rotation=0, labelpad=90, va="center")

    _fill_row(0, logit_pooled, "Pre-softmax\nlogits")
    _fill_row(1, attn_pooled, "Post-softmax\nweights")

    axes[0, 0].set_title("Exterior (raw)", fontsize=9)
    axes[0, 1].set_title("Exterior (attn)", fontsize=9)
    axes[0, 2].set_title("Wrist (raw)", fontsize=9)
    axes[0, 3].set_title("Wrist (attn)", fontsize=9)

    plt.suptitle(
        f'H7.3 Pre-softmax vs Post-softmax  |  Layer {layer_idx}  |  Suffix action tokens\n'
        f'"{example["prompt"]}"',
        fontsize=12, y=1.01,
    )
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"h7_logits_vs_softmax_L{layer_idx}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_suffix_attention] Logits-vs-Softmax saved → {out_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from attn_map import load_duck_example

    example = load_duck_example(camera="left", index=0)

    LAYERS = [1, 4, 5, 7, 10]
    JOINT_DIR = "results/layers_joint"
    PREFIX_DIR = "results/layers_prefix"
    OUTPUT_DIR = "results"
    NAME = "duck_left_0"

    print("=== H7.2/H7.3 Suffix Attention Visualization ===")
    print(f"Prompt: {example['prompt']}")
    print()

    for layer in LAYERS:
        visualize_suffix_attention(
            example,
            layer_idx=layer,
            joint_dir=JOINT_DIR,
            output_dir=OUTPUT_DIR,
            name=NAME,
            head_agg="max",
        )
        compare_prefix_suffix(
            example,
            layer_idx=layer,
            joint_dir=JOINT_DIR,
            prefix_dir=PREFIX_DIR,
            output_dir=OUTPUT_DIR,
            name=NAME,
        )
        visualize_logits_vs_softmax(
            example,
            layer_idx=layer,
            joint_dir=JOINT_DIR,
            output_dir=OUTPUT_DIR,
            name=NAME,
        )

    entropy_prefix_vs_suffix(
        example,
        layers=LAYERS,
        joint_dir=JOINT_DIR,
        prefix_dir=PREFIX_DIR,
        output_dir=OUTPUT_DIR,
        name=NAME,
    )
