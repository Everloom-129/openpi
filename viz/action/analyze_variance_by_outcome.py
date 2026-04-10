"""Test whether failure episodes have significantly higher variance in action
self-attention at layer 17, step 9 than success episodes.

Variance is defined as within-episode std across frames — i.e., how much the
action self-attention fluctuates from frame to frame inside a single episode.

Steps:
  1. Load frame-level data (NOT episode-averaged) at layer 17, step 9.
  2. Compute per-episode std of action_self.
  3. Split by outcome (episode path starts with "success/" or "failure/").
  4. Mann-Whitney U test + plot.

Usage:
    uv run python viz/analyze_variance_by_outcome.py \
        --root /mnt/sda/edward/projects/pi05_vis/faraz_action/left \
        --layer 17 --step 9 --out outcome_variance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "viz")]:
    if p not in sys.path:
        sys.path.insert(0, p)

GROUPS = ["ext_img", "wrist_img", "text", "action_self"]
GROUP_LABEL = {
    "ext_img":     "Ext camera",
    "wrist_img":   "Wrist camera",
    "text":        "Text tokens",
    "action_self": "Action (self)",
}
NUM_LAYERS = 18


# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_h5(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.h5") if "_" not in p.stem)


def episode_id(h5_path: Path, root: Path) -> str:
    try:
        parts = h5_path.relative_to(root).parts
        return "/".join(parts[:-2])
    except ValueError:
        return str(h5_path.parent.parent)


def load_scalar(h5_path: Path, layer: int, step: int) -> float | None:
    """Load action_self attention mass for one frame at given layer and step."""
    try:
        with h5py.File(h5_path, "r") as f:
            key = f"suffix_denoising/layer_{layer}/group_masses"
            if key not in f:
                return None
            gm = f[key][()]   # (n_steps, 4)
            if step >= gm.shape[0]:
                return None
            return float(gm[step, 3])   # index 3 = action_self
    except Exception as e:
        print(f"  [warn] {h5_path.name}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/sda/edward/projects/pi05_vis/faraz_action/left")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--step",  type=int, default=9)
    parser.add_argument("--out",   default="outcome_variance")
    args = parser.parse_args()

    root   = Path(args.root)
    layer  = args.layer
    step   = args.step
    out    = Path(args.out)

    print(f"[scan] root={root}")
    h5_files = discover_h5(root)
    print(f"[scan] {len(h5_files)} H5 files")

    # ── Load frame-level scalars ───────────────────────────────────────────────
    records: list[dict] = []
    n_miss = 0
    for i, h5 in enumerate(h5_files):
        if i % 200 == 0:
            print(f"  {i}/{len(h5_files)}", flush=True)
        val = load_scalar(h5, layer, step)
        if val is None:
            n_miss += 1
            continue
        ep  = episode_id(h5, root)
        records.append({"episode": ep, "action_self": val})

    if not records:
        print("[error] No data found.")
        return

    frame_df = pd.DataFrame(records)
    frame_df["outcome"] = frame_df["episode"].str.split("/").str[0]
    print(f"[data]  {len(frame_df)} frames, {frame_df['episode'].nunique()} episodes, {n_miss} missing")
    print(f"        outcomes: {frame_df['outcome'].value_counts().to_dict()}")

    # ── Within-episode std ─────────────────────────────────────────────────────
    ep_stats = (
        frame_df.groupby(["episode", "outcome"])["action_self"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    # Episodes with only 1 frame get std=NaN — exclude them from std analysis
    ep_stats_2plus = ep_stats[ep_stats["n"] >= 2].copy()

    suc = ep_stats_2plus[ep_stats_2plus["outcome"] == "success"]["std"].dropna()
    fai = ep_stats_2plus[ep_stats_2plus["outcome"] == "failure"]["std"].dropna()

    print(f"\n[within-episode std of action_self | layer={layer}, step={step}]")
    print(f"  success  n={len(suc):3d}  mean={suc.mean():.4f}  median={suc.median():.4f}  "
          f"max={suc.max():.4f}")
    print(f"  failure  n={len(fai):3d}  mean={fai.mean():.4f}  median={fai.median():.4f}  "
          f"max={fai.max():.4f}")

    # ── Mann-Whitney U test ───────────────────────────────────────────────────
    from scipy import stats as sp_stats
    if len(suc) >= 3 and len(fai) >= 3:
        u_stat, p_val = sp_stats.mannwhitneyu(fai, suc, alternative="greater")
        print(f"\n[Mann-Whitney U] H₁: failure_std > success_std")
        print(f"  U={u_stat:.0f}  p={p_val:.4f}  {'*significant* (p<0.05)' if p_val < 0.05 else 'not significant'}")
    else:
        print("[warn] Too few episodes for Mann-Whitney test")
        p_val = float("nan")

    # ── Also compare episode means ────────────────────────────────────────────
    suc_m = ep_stats[ep_stats["outcome"] == "success"]["mean"].dropna()
    fai_m = ep_stats[ep_stats["outcome"] == "failure"]["mean"].dropna()
    print(f"\n[episode mean of action_self | layer={layer}, step={step}]")
    print(f"  success  n={len(suc_m):3d}  mean={suc_m.mean():.4f}  median={suc_m.median():.4f}")
    print(f"  failure  n={len(fai_m):3d}  mean={fai_m.mean():.4f}  median={fai_m.median():.4f}")
    if len(suc_m) >= 3 and len(fai_m) >= 3:
        u2, p2 = sp_stats.mannwhitneyu(fai_m, suc_m, alternative="greater")
        print(f"  Mann-Whitney (mean): U={u2:.0f}  p={p2:.4f}  "
              f"{'*significant*' if p2 < 0.05 else 'not significant'}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    ep_stats.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n[csv]  saved → {out.with_suffix('.csv')}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: strip + box for within-episode std
    ax = axes[0]
    for outcome, color in [("success", "#4e79a7"), ("failure", "#e15759")]:
        vals = ep_stats_2plus[ep_stats_2plus["outcome"] == outcome]["std"].dropna()
        x_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        xpos = 0 if outcome == "success" else 1
        ax.scatter(xpos + x_jitter, vals, alpha=0.6, s=30, color=color, label=outcome)
        ax.boxplot(vals, positions=[xpos], widths=0.25, patch_artist=False,
                   medianprops=dict(color="black", linewidth=2))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["success", "failure"])
    ax.set_ylabel("Within-episode std of action self-attn")
    ax.set_title(f"Intra-episode variability\nLayer {layer}, step {step}")
    p_str = f"p={p_val:.3f}" if not np.isnan(p_val) else "n/a"
    ax.text(0.98, 0.98, f"MW p={p_val:.3f}\n(H₁: failure>success)", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            color=("red" if p_val < 0.05 else "gray"))
    ax.grid(True, alpha=0.25, axis="y")

    # Right: episode means
    ax = axes[1]
    for outcome, color in [("success", "#4e79a7"), ("failure", "#e15759")]:
        vals = ep_stats[ep_stats["outcome"] == outcome]["mean"].dropna()
        x_jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        xpos = 0 if outcome == "success" else 1
        ax.scatter(xpos + x_jitter, vals, alpha=0.6, s=30, color=color, label=outcome)
        ax.boxplot(vals, positions=[xpos], widths=0.25, patch_artist=False,
                   medianprops=dict(color="black", linewidth=2))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["success", "failure"])
    ax.set_ylabel("Episode mean action self-attn")
    ax.set_title(f"Episode mean\nLayer {layer}, step {step}")
    if len(suc_m) >= 3 and len(fai_m) >= 3:
        ax.text(0.98, 0.98, f"MW p={p2:.3f}\n(H₁: failure>success)", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                color=("red" if p2 < 0.05 else "gray"))
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        f"Action self-attention: success vs failure\n"
        f"faraz_action/left | layer={layer}, denoising step={step}",
        fontsize=11,
    )
    plt.tight_layout()
    png = out.with_suffix(".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {png}")
    plt.close(fig)

    print("\n[done]")


if __name__ == "__main__":
    main()
