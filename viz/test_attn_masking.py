"""Attention masking ablation experiment.

Masks attention logits in specific layers during the prefix forward pass
and measures how much the predicted actions deviate from the unmasked baseline.

Usage:
    uv run python viz/test_attn_masking.py

    # Restrict to a single GPU:
    CUDA_VISIBLE_DEVICES=0 uv run python viz/test_attn_masking.py
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np

JAX_CHECKPOINT_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache/openpi/openpi-assets/checkpoints"
)
CHECKPOINT_NAME = "pi05_droid"
CHECKPOINT_DIR = os.path.join(JAX_CHECKPOINT_ROOT, CHECKPOINT_NAME)
FRAME_IDX = 40
SAVE_DIR = os.path.join(_PROJECT_ROOT, "attn_jax", "masking_ablation")

# Joint names for DROID (first 7 = joint velocities, 8th = gripper)
JOINT_NAMES = ["j0_vel", "j1_vel", "j2_vel", "j3_vel", "j4_vel", "j5_vel", "j6_vel", "gripper"]

# ── Experiment configs ───────────────────────────────────────────────────────
# Each entry: (label, layer_config_dict, percentile)
# mode 1 = mask top N% (suppress strongest attention)
# mode 2 = mask bottom N% (suppress weakest attention)
EXPERIMENTS = [
    # Single-layer ablations (mode 1: mask top 10%)
    ("L0_top10",   {0: 1},  10.0),
    ("L3_top10",   {3: 1},  10.0),
    ("L5_top10",   {5: 1},  10.0),
    ("L7_top10",   {7: 1},  10.0),
    ("L10_top10",  {10: 1}, 10.0),
    ("L14_top10",  {14: 1}, 10.0),
    ("L17_top10",  {17: 1}, 10.0),
    # Stronger masking
    ("L7_top20",   {7: 1},  20.0),
    ("L7_top50",   {7: 1},  50.0),
    # Mode 2: mask bottom (suppress weakest, keep strongest)
    ("L7_bot10",   {7: 2},  10.0),
    ("L7_bot50",   {7: 2},  50.0),
    # Mode 3: min filter — keep only top N%, mask everything below
    ("L7_min10",   {7: 3},  10.0),   # keep top 10%, mask bottom 90%
    ("L7_min20",   {7: 3},  20.0),   # keep top 20%, mask bottom 80%
    ("L7_min50",   {7: 3},  50.0),   # keep top 50%, mask bottom 50%
    # Min filter across layers
    ("L0_min10",   {0: 3},  10.0),
    ("L5_min10",   {5: 3},  10.0),
    ("L10_min10",  {10: 3}, 10.0),
    ("L14_min10",  {14: 3}, 10.0),
    ("L17_min10",  {17: 3}, 10.0),
    # Multi-layer
    ("L5_7_10_top10", {5: 1, 7: 1, 10: 1}, 10.0),
    ("L0_17_top10",   {0: 1, 17: 1},       10.0),
    # All layers
    ("all_top10",  {i: 1 for i in range(18)}, 10.0),
    ("all_min10",  {i: 3 for i in range(18)}, 10.0),
]

NUM_STEPS = 10  # denoising steps


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


# ── Step 1: Load example ────────────────────────────────────────────────────
step("Load duck example (frame 40)")
try:
    from attn_map import load_duck_example

    example = load_duck_example(camera="left", index=FRAME_IDX)
    example["prompt"] = "place the duck toy into the pink bowl"
    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]
    print(f"  prompt: {example['prompt']}")
    print(f"  exterior: {ext_img.shape}, wrist: {wrist_img.shape}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 2: Load model ──────────────────────────────────────────────────────
step("Load JAX model")
try:
    import pathlib

    from openpi.models import model as _model
    from openpi.shared import download
    from openpi.training import config as _cfg

    raw = os.path.basename(os.path.normpath(CHECKPOINT_DIR))
    config_name = "pi05_droid" if "pi05" in raw.lower() else "pi0_droid"
    is_pi05 = "pi05" in config_name

    config = _cfg.get_config(config_name)
    ckpt = pathlib.Path(str(download.maybe_download(CHECKPOINT_DIR)))
    params_path = ckpt / "params" if (ckpt / "params").is_dir() else ckpt
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()
    print(f"  config: {config_name}, pi05: {is_pi05}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 3: Build observation ────────────────────────────────────────────────
step("Build observation & tokenize")
try:
    import jax
    import jax.numpy as jnp

    images = {}
    image_masks = {}
    for img_key, model_key in [
        ("observation/exterior_image_1_left", "base_0_rgb"),
        ("observation/wrist_image_left", "left_wrist_0_rgb"),
    ]:
        img = example[img_key]
        if img.ndim == 3:
            img = img[None]
        images[model_key] = jnp.array(img, dtype=jnp.float32) / 255.0
        image_masks[model_key] = jnp.ones((img.shape[0],), dtype=jnp.bool_)

    dummy = jnp.zeros_like(next(iter(images.values())))
    images["right_wrist_0_rgb"] = dummy
    image_masks["right_wrist_0_rgb"] = jnp.zeros((dummy.shape[0],), dtype=jnp.bool_)

    state = jnp.array(example["observation/joint_position"], dtype=jnp.float32)
    if state.ndim == 1:
        state = state[None]

    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer()
    prompt = example["prompt"]
    max_token_len = getattr(model, "max_token_len", 200)
    token_ids, token_mask = tokenizer.tokenize(prompt, state=state[0] if is_pi05 else None)
    token_ids = np.array(token_ids[:max_token_len], dtype=np.int32)
    token_mask = np.array(token_mask[:max_token_len], dtype=bool)
    n_real_tokens = int(token_mask.sum())
    token_texts = [tokenizer._tokenizer.id_to_piece(int(i)) for i in token_ids[:n_real_tokens]]

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=jnp.array(token_ids[None]),
        tokenized_prompt_mask=jnp.array(token_mask[None], dtype=bool),
    )
    print(f"  tokens ({n_real_tokens}): {' '.join(token_texts[:15])}...")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 4: Generate fixed noise ─────────────────────────────────────────────
step("Generate fixed noise for fair comparison")
try:
    rng = jax.random.key(42)
    batch_size = observation.state.shape[0]
    fixed_noise = jax.random.normal(
        rng, (batch_size, model.action_horizon, model.action_dim)
    )
    print(f"  noise shape: {fixed_noise.shape}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 5: Run baseline (no masking) ────────────────────────────────────────
step("Run baseline inference (no masking)")
try:
    # Ensure no masking
    model.attn_logit_mask_layers = None
    model.attn_logit_mask_percentile = 10.0

    obs_preprocessed = _model.preprocess_observation(None, observation, train=False)
    baseline_actions = model.sample_actions(
        rng, observation, num_steps=NUM_STEPS, noise=fixed_noise,
    )
    baseline_np = np.array(baseline_actions)
    print(f"  baseline actions: {baseline_np.shape}")
    print(f"  first action step: {baseline_np[0, 0, :8]}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 6: Run masked experiments ───────────────────────────────────────────
step("Run masking experiments")
results = {}
try:
    for label, layer_config, pct in EXPERIMENTS:
        print(f"\n  --- {label} (layers={layer_config}, pct={pct}) ---")
        model.attn_logit_mask_layers = layer_config
        model.attn_logit_mask_percentile = pct

        masked_actions = model.sample_actions(
            rng, observation, num_steps=NUM_STEPS, noise=fixed_noise,
        )
        masked_np = np.array(masked_actions)

        # Compute deviation from baseline
        diff = masked_np - baseline_np
        l2_per_step = np.linalg.norm(diff[0], axis=-1)  # (action_horizon,)
        l2_total = np.linalg.norm(diff[0])
        l2_first8 = np.linalg.norm(diff[0, :8, :8])  # first 8 steps, first 8 dims (DROID)
        max_dev = np.abs(diff[0, :8, :8]).max()
        mean_dev = np.abs(diff[0, :8, :8]).mean()

        results[label] = {
            "actions": masked_np,
            "diff": diff,
            "l2_total": float(l2_total),
            "l2_first8": float(l2_first8),
            "max_dev": float(max_dev),
            "mean_dev": float(mean_dev),
            "l2_per_step": l2_per_step,
            "layer_config": layer_config,
            "pct": pct,
        }
        print(f"    L2 (total): {l2_total:.6f}")
        print(f"    L2 (first 8 steps, 8 dims): {l2_first8:.6f}")
        print(f"    max |deviation|: {max_dev:.6f}")
        print(f"    mean |deviation|: {mean_dev:.6f}")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# Reset masking
model.attn_logit_mask_layers = None


# ── Step 7: Visualize results ────────────────────────────────────────────────
step("Visualize & save results")
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    labels_sorted = sorted(results.keys(), key=lambda k: results[k]["l2_first8"], reverse=True)
    n_exp = len(labels_sorted)
    n_joints = min(8, baseline_np.shape[-1])

    # ─── Fig 1: Horizontal bar chart — overall L2 deviation ranking ──────
    fig, ax = plt.subplots(figsize=(10, max(4, n_exp * 0.45)))
    def _bar_color(l):
        if "min" in l:
            return "#2ecc71"  # green for min-filter
        if "top" in l:
            return "#e74c3c"  # red for mask-top
        return "#3498db"      # blue for mask-bottom
    colors_bar = [_bar_color(l) for l in labels_sorted]
    vals = [results[l]["l2_first8"] for l in labels_sorted]
    y_pos = np.arange(n_exp)
    ax.barh(y_pos, vals, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("L2 Deviation from Baseline (first 8 steps x 8 dims)")
    ax.set_title("Attention Masking: Action Deviation Ranking")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.4f}", va="center", fontsize=8, color="#555")
    # Add legend for colors
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor="#e74c3c"), Patch(facecolor="#3498db"), Patch(facecolor="#2ecc71")],
        ["mask top (suppress strongest)", "mask bottom (suppress weakest)", "min filter (keep only top N%)"],
        loc="lower right", fontsize=9,
    )
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig1_deviation_ranking.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ─── Fig 2: Heatmap — layer x joint deviation (top10 vs min10 side by side) ─
    top10_exps = sorted(
        [l for l in results if l.startswith("L") and "_top10" in l and l.count("_") == 1],
        key=lambda l: int(l.split("_")[0][1:]),
    )
    min10_exps = sorted(
        [l for l in results if l.startswith("L") and "_min10" in l and l.count("_") == 1],
        key=lambda l: int(l.split("_")[0][1:]),
    )

    def _make_heatmap(ax, exp_list, title, cmap):
        layer_ids = [int(l.split("_")[0][1:]) for l in exp_list]
        data = np.zeros((len(exp_list), n_joints))
        for i, l in enumerate(exp_list):
            for j in range(n_joints):
                data[i, j] = np.abs(results[l]["diff"][0, :8, j]).mean()
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(np.arange(n_joints))
        ax.set_xticklabels([JOINT_NAMES[j] for j in range(n_joints)], rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(exp_list)))
        ax.set_yticklabels([f"Layer {l}" for l in layer_ids], fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                color = "white" if v > data.max() * 0.6 else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=7, color=color)
        return im

    n_panels = 1 + (1 if min10_exps else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, max(3, len(top10_exps) * 0.5)))
    if n_panels == 1:
        axes = [axes]
    if top10_exps:
        im = _make_heatmap(axes[0], top10_exps, "Mode 1: Mask Top 10% (suppress strongest)", "YlOrRd")
        fig.colorbar(im, ax=axes[0], label="Mean |dev|", shrink=0.8)
    if min10_exps:
        im = _make_heatmap(axes[1], min10_exps, "Mode 3: Min Filter 10% (keep only top 10%)", "YlGn")
        fig.colorbar(im, ax=axes[1], label="Mean |dev|", shrink=0.8)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig2_layer_joint_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    single_layer_exps = top10_exps

    # ─── Fig 3: Percentile sweep on Layer 7 (top + min) ────────────────
    pct_top = sorted([l for l in results if l.startswith("L7_top")], key=lambda l: results[l]["pct"])
    pct_min = sorted([l for l in results if l.startswith("L7_min")], key=lambda l: results[l]["pct"])
    if len(pct_top) >= 2 or len(pct_min) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 3a: L2 vs percentile — both modes
        ax = axes[0]
        if pct_top:
            pcts = [results[l]["pct"] for l in pct_top]
            l2s = [results[l]["l2_first8"] for l in pct_top]
            ax.plot(pcts, l2s, "o-", color="#e74c3c", linewidth=2, markersize=8, label="mask top N%")
            for p, v in zip(pcts, l2s):
                ax.annotate(f"{v:.4f}", (p, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8, color="#e74c3c")
        if pct_min:
            pcts = [results[l]["pct"] for l in pct_min]
            l2s = [results[l]["l2_first8"] for l in pct_min]
            ax.plot(pcts, l2s, "s-", color="#2ecc71", linewidth=2, markersize=8, label="min filter (keep top N%)")
            for p, v in zip(pcts, l2s):
                ax.annotate(f"{v:.4f}", (p, v), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=8, color="#2ecc71")
        ax.set_xlabel("Percentile (%)")
        ax.set_ylabel("L2 Deviation")
        ax.set_title("Layer 7: Masking Strength vs Action Deviation")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3b: Per-joint grouped bars — all L7 percentile experiments
        ax = axes[1]
        all_pct_exps = pct_top + pct_min
        x = np.arange(n_joints)
        width = 0.8 / max(len(all_pct_exps), 1)
        reds = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(pct_top), 1)))
        greens = plt.cm.Greens(np.linspace(0.3, 0.9, max(len(pct_min), 1)))
        for i, l in enumerate(pct_top):
            joint_devs = [np.abs(results[l]["diff"][0, :8, j]).mean() for j in range(n_joints)]
            ax.bar(x + i * width - 0.4 + width / 2, joint_devs, width,
                   label=f"top {results[l]['pct']:.0f}%", color=reds[i], edgecolor="white")
        for i, l in enumerate(pct_min):
            idx = len(pct_top) + i
            joint_devs = [np.abs(results[l]["diff"][0, :8, j]).mean() for j in range(n_joints)]
            ax.bar(x + idx * width - 0.4 + width / 2, joint_devs, width,
                   label=f"min {results[l]['pct']:.0f}%", color=greens[i], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([JOINT_NAMES[j] for j in range(n_joints)], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean |deviation|")
        ax.set_title("Layer 7: Per-Joint Impact (Top-Mask vs Min-Filter)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        path = os.path.join(SAVE_DIR, "fig3_percentile_sweep.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ─── Fig 4: Three-mode comparison (top vs bottom vs min) ──────────────
    # Find percentiles where all three modes exist
    three_mode_pcts = []
    for l in results:
        if "L7_top" in l:
            pct = results[l]["pct"]
            bot_key = f"L7_bot{int(pct)}"
            min_key = f"L7_min{int(pct)}"
            if bot_key in results or min_key in results:
                three_mode_pcts.append((l, bot_key, min_key, pct))
    if three_mode_pcts:
        fig, axes = plt.subplots(1, len(three_mode_pcts), figsize=(6.5 * len(three_mode_pcts), 5))
        if len(three_mode_pcts) == 1:
            axes = [axes]
        for ax, (tk, bk, mk, pct) in zip(axes, three_mode_pcts):
            x = np.arange(n_joints)
            bar_w = 0.25
            top_devs = [np.abs(results[tk]["diff"][0, :8, j]).mean() for j in range(n_joints)] if tk in results else [0] * n_joints
            bot_devs = [np.abs(results[bk]["diff"][0, :8, j]).mean() for j in range(n_joints)] if bk in results else [0] * n_joints
            min_devs = [np.abs(results[mk]["diff"][0, :8, j]).mean() for j in range(n_joints)] if mk in results else [0] * n_joints
            if tk in results:
                ax.bar(x - bar_w, top_devs, bar_w, label=f"mask top {pct:.0f}%", color="#e74c3c", alpha=0.85)
            if bk in results:
                ax.bar(x, bot_devs, bar_w, label=f"mask bottom {pct:.0f}%", color="#3498db", alpha=0.85)
            if mk in results:
                ax.bar(x + bar_w, min_devs, bar_w, label=f"min filter {pct:.0f}%", color="#2ecc71", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([JOINT_NAMES[j] for j in range(n_joints)], rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Mean |deviation|")
            ax.set_title(f"Layer 7: All Three Modes ({pct:.0f}%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        path = os.path.join(SAVE_DIR, "fig4_three_modes.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ─── Fig 5: Action trajectory — baseline vs masked (all 8 joints) ────
    key_exps = ["L7_top10", "L7_top50", "L7_min10", "all_top10", "all_min10", "L7_bot50"]
    key_exps = [k for k in key_exps if k in results]
    traj_colors = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4", "#42d4f4"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.flatten()
    steps = np.arange(8)
    for j in range(n_joints):
        ax = axes[j]
        jname = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"dim_{j}"
        # Baseline
        ax.plot(steps, baseline_np[0, :8, j], "k-", linewidth=2.5, label="baseline", zorder=10)
        # Masked experiments
        for ei, k in enumerate(key_exps):
            ax.plot(steps, results[k]["actions"][0, :8, j], "--",
                    color=traj_colors[ei % len(traj_colors)], linewidth=1.5, label=k, alpha=0.85)
        ax.set_title(jname, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Action Step" if j >= 4 else "")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Action Trajectories: Baseline vs Masked Attention", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(SAVE_DIR, "fig5_action_trajectories.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ─── Fig 6: Deviation over action steps (L2 per step) ───────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    steps_full = np.arange(baseline_np.shape[1])
    for label in ["L7_top10", "L7_top50", "L7_min10", "L7_min50", "all_top10", "all_min10"]:
        if label not in results:
            continue
        ax.plot(steps_full, results[label]["l2_per_step"], "-o", markersize=3, label=label, alpha=0.8)
    ax.set_xlabel("Action Step (0-49)")
    ax.set_ylabel("L2 Deviation per Step")
    ax.set_title("How Masking Deviation Propagates Across Action Horizon")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "fig6_deviation_over_steps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # ─── Fig 7: Multi-layer vs single-layer comparison ───────────────────
    compare_keys = ["L7_top10", "L7_min10", "L5_7_10_top10", "L0_17_top10", "all_top10", "all_min10"]
    compare_keys = [k for k in compare_keys if k in results]
    if compare_keys:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(n_joints)
        width = 0.8 / len(compare_keys)
        cmap = plt.cm.Set2(np.linspace(0, 1, len(compare_keys)))
        for i, k in enumerate(compare_keys):
            joint_devs = [np.abs(results[k]["diff"][0, :8, j]).mean() for j in range(n_joints)]
            ax.bar(x + i * width - 0.4 + width / 2, joint_devs, width,
                   label=k, color=cmap[i], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([JOINT_NAMES[j] for j in range(n_joints)], rotation=45, ha="right")
        ax.set_ylabel("Mean |deviation|")
        ax.set_title("Single-Layer vs Multi-Layer Masking")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        path = os.path.join(SAVE_DIR, "fig7_single_vs_multi_layer.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ─── Print summary table ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  {'Experiment':<25s} {'L2(8x8)':>10s} {'Max|dev|':>10s} {'Mean|dev|':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for label in labels_sorted:
        r = results[label]
        print(f"  {label:<25s} {r['l2_first8']:>10.5f} {r['max_dev']:>10.5f} {r['mean_dev']:>10.5f}")
    print(f"{'='*80}")

except Exception:
    traceback.print_exc()
    sys.exit(1)


print(f"\n{'='*60}")
print(f"  ALL DONE — results saved to {SAVE_DIR}")
print(f"{'='*60}")
