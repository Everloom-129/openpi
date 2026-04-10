"""Export action→token attention at denoising steps 0, 5, 9 to Excel + curves.

Averaging strategy (episode-fair):
    1. All frames within an episode  →  average  (episode mean)
    2. All episodes                  →  average  (grand mean)

This gives equal weight to each episode regardless of frame count.

Outputs:
  <out>.xlsx
    "episode_mean"    — one row per episode × layer × step  (averaged over frames)
    "grand_mean"      — one row per layer × step  (averaged over episodes)
    "pivot_wide"      — wide: rows=(episode, layer), cols=step×group
    "layer_L{i}"      — per-layer slice for each of the 18 layers

  <out>.png
    4-panel line chart  (x=layer, 3 lines for step 0/5/9, y=attn mass)
    averaged over all episodes

  <out>_grid.png
    4×3 grid: rows=groups, cols=steps, bars over layers

Usage:
    bash viz/export_attn_grid.sh
    # or directly:
    uv run python viz/export_denoising_spreadsheet.py \\
        --root /mnt/sda/edward/projects/pi05_vis/faraz_action/left \\
        --steps 0 5 9 --out denoising_attn
"""
from __future__ import annotations

import argparse
import os
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
GROUP_COLOR = {
    "ext_img":     "#4e79a7",
    "wrist_img":   "#f28e2b",
    "text":        "#59a14f",
    "action_self": "#e15759",
}
NUM_LAYERS = 18


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_h5(root: Path) -> list[Path]:
    """Find all main-inference H5 files (no underscore in stem)."""
    return sorted(p for p in root.rglob("*.h5") if "_" not in p.stem)


def episode_id(h5_path: Path, root: Path) -> str:
    """Return episode identifier (outcome/date/episode), dropping the frame sub-dir."""
    try:
        parts = h5_path.relative_to(root).parts
        # structure: outcome / date / episode_id / frame / frame.h5
        return "/".join(parts[:-2])   # drop frame dir + filename
    except ValueError:
        return str(h5_path.parent.parent)


def frame_id(h5_path: Path, root: Path) -> str:
    """Return frame sub-directory name (e.g. '00152')."""
    return h5_path.parent.name


# ── Data loading ──────────────────────────────────────────────────────────────

def load_group_masses(h5_path: Path, target_steps: list[int]) -> np.ndarray | None:
    """Load group_masses for all layers at the target denoising steps.

    Returns float32(n_target_steps, NUM_LAYERS, 4) or None.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if "suffix_denoising" not in f:
                return None
            n_steps = int(f["suffix_denoising/n_steps"][()])
            out = np.full((len(target_steps), NUM_LAYERS, 4), np.nan, dtype=np.float32)
            for li in range(NUM_LAYERS):
                key = f"suffix_denoising/layer_{li}/group_masses"
                if key not in f:
                    continue
                gm = f[key][()]   # (n_steps, 4)
                for si, step in enumerate(target_steps):
                    if step < n_steps:
                        out[si, li] = gm[step]
            return out
    except Exception as e:
        print(f"  [warn] {h5_path.name}: {e}")
        return None


# ── Aggregation ───────────────────────────────────────────────────────────────

def collect_data(
    h5_files: list[Path],
    root: Path,
    target_steps: list[int],
) -> tuple[pd.DataFrame, int]:
    """Load all frames, group by episode, average within episode.

    Returns:
        episode_df  — one row per (episode, layer, step) with 4 group columns
        n_missing   — frames that had no /suffix_denoising data
    """
    # episode_id → list of arrays (n_steps, n_layers, 4)
    episode_arrays: dict[str, list[np.ndarray]] = {}
    n_missing = 0

    print(f"  loading {len(h5_files)} frames …", flush=True)
    for i, h5 in enumerate(h5_files):
        if i % 200 == 0:
            print(f"  {i}/{len(h5_files)}", flush=True)
        arr = load_group_masses(h5, target_steps)   # (n_steps, n_layers, 4) or None
        if arr is None:
            n_missing += 1
            continue
        ep = episode_id(h5, root)
        episode_arrays.setdefault(ep, []).append(arr)

    if not episode_arrays:
        return pd.DataFrame(), n_missing

    # Average within each episode
    records = []
    for ep, arrays in episode_arrays.items():
        ep_mean = np.mean(arrays, axis=0)   # (n_steps, n_layers, 4)
        n_frames = len(arrays)
        for si, step in enumerate(target_steps):
            for li in range(NUM_LAYERS):
                row = {
                    "episode":        ep,
                    "n_frames":       n_frames,
                    "layer":          li,
                    "denoising_step": step,
                }
                for gi, g in enumerate(GROUPS):
                    row[g] = float(ep_mean[si, li, gi])
                records.append(row)

    return pd.DataFrame(records), n_missing


# ── Excel output ──────────────────────────────────────────────────────────────

def write_excel(
    ep_df: pd.DataFrame,
    grand_df: pd.DataFrame,
    out_path: Path,
) -> None:
    rename = {g: GROUP_LABEL[g] for g in GROUPS}

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # episode_mean sheet
        ep_out = ep_df[["episode", "n_frames", "layer", "denoising_step"] + GROUPS].copy()
        ep_out = ep_out.sort_values(["episode", "layer", "denoising_step"])
        ep_out.rename(columns=rename, inplace=True)
        ep_out.to_excel(writer, sheet_name="episode_mean", index=False)

        # grand_mean sheet
        gm_out = grand_df[["layer", "denoising_step"] + GROUPS].copy()
        gm_out = gm_out.sort_values(["layer", "denoising_step"])
        gm_out.rename(columns=rename, inplace=True)
        gm_out.to_excel(writer, sheet_name="grand_mean", index=False)

        # pivot_wide: rows=(episode, layer), cols=step×group
        try:
            pivot = ep_df.pivot_table(
                index=["episode", "layer"],
                columns="denoising_step",
                values=GROUPS,
                aggfunc="mean",
            )
            pivot.columns = [f"step{s}_{g}" for g, s in pivot.columns]
            pivot.reset_index().to_excel(writer, sheet_name="pivot_wide", index=False)
        except Exception as e:
            print(f"  [warn] pivot sheet: {e}")

        # per-layer sheets
        for li in range(NUM_LAYERS):
            ldf = ep_df[ep_df["layer"] == li][
                ["episode", "n_frames", "denoising_step"] + GROUPS
            ].sort_values(["episode", "denoising_step"])
            ldf.rename(columns=rename, inplace=True)
            ldf.to_excel(writer, sheet_name=f"layer_{li}", index=False)

    print(f"[excel] saved → {out_path}")


# ── Plot: line curves ─────────────────────────────────────────────────────────

def plot_curves(grand_df: pd.DataFrame, target_steps: list[int], out_path: Path, n_eps: int) -> None:
    import matplotlib.pyplot as plt

    step_kw = [
        dict(linestyle="-",  linewidth=2.5, marker="o", markersize=5),
        dict(linestyle="--", linewidth=2.0, marker="s", markersize=4),
        dict(linestyle=":",  linewidth=2.0, marker="^", markersize=4),
    ]
    layers = sorted(grand_df["layer"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, g in zip(axes.flatten(), GROUPS):
        for si, step in enumerate(target_steps):
            sub = grand_df[grand_df["denoising_step"] == step].sort_values("layer")
            ax.plot(
                sub["layer"], sub[g],
                color=GROUP_COLOR[g],
                label=f"step {step}",
                **step_kw[si % len(step_kw)],
            )
        ax.set_title(GROUP_LABEL[g], fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean attention mass")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Action→token attention  |  denoising steps {target_steps}\n"
        f"Grand mean over {n_eps} episodes (episode-fair averaging)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {out_path}")
    plt.close(fig)


# ── Plot: step × layer × group grid ──────────────────────────────────────────

def plot_grid(grand_df: pd.DataFrame, target_steps: list[int], out_path: Path, n_eps: int) -> None:
    import matplotlib.pyplot as plt

    layers = sorted(grand_df["layer"].unique())
    n_steps = len(target_steps)
    n_groups = len(GROUPS)

    fig, axes = plt.subplots(
        n_groups, n_steps,
        figsize=(5 * n_steps, 3.5 * n_groups),
        sharey="row", sharex=True,
    )

    for ri, g in enumerate(GROUPS):
        for ci, step in enumerate(target_steps):
            ax = axes[ri][ci]
            sub = grand_df[grand_df["denoising_step"] == step].sort_values("layer")
            ax.bar(sub["layer"], sub[g], color=GROUP_COLOR[g], alpha=0.70, width=0.7)
            ax.plot(sub["layer"], sub[g],
                    color=GROUP_COLOR[g], linewidth=1.8,
                    marker="o", markersize=3.5)
            if ri == 0:
                ax.set_title(f"Denoising step {step}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{GROUP_LABEL[g]}\nAttn mass", fontsize=9)
            ax.set_xticks(layers[::2])
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(True, alpha=0.2, axis="y")

    for ax in axes[-1]:
        ax.set_xlabel("Layer", fontsize=9)

    fig.suptitle(
        f"Action attn: step × layer × group  |  {n_eps} episodes (episode-fair avg)",
        fontsize=12,
    )
    plt.tight_layout()
    grid_path = out_path.with_stem(out_path.stem + "_grid")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"[plot]  saved → {grid_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get(
            "RESULTS_ROOT",
            "/mnt/sda/edward/projects/pi05_vis/faraz_action/left",
        ),
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 5, 9])
    parser.add_argument("--out", default="denoising_attn")
    args = parser.parse_args()

    root = Path(args.root)
    target_steps = sorted(args.steps)
    out_base = Path(args.out)

    print(f"[scan]  root  = {root}")
    h5_files = discover_h5(root)
    print(f"[scan]  found {len(h5_files)} main-inference H5 files")

    if not h5_files:
        print("[error] No H5 files found. Check --root path.")
        return

    ep_df, n_missing = collect_data(h5_files, root, target_steps)

    if ep_df.empty:
        print("[error] No /suffix_denoising data found. Re-run pipeline.py first.")
        return

    if n_missing:
        print(f"[info]  {n_missing} frames skipped (no denoising data)")

    n_eps = ep_df["episode"].nunique()
    print(f"[data]  {n_eps} episodes  ({len(h5_files) - n_missing} frames with data)")

    # Grand mean: average episode means across episodes (episode-fair)
    grand_df = ep_df.groupby(["layer", "denoising_step"])[GROUPS].mean().reset_index()

    # Write outputs
    write_excel(ep_df, grand_df, out_base.with_suffix(".xlsx"))
    plot_curves(grand_df, target_steps, out_base.with_suffix(".png"), n_eps)
    plot_grid(grand_df, target_steps, out_base.with_suffix(".png"), n_eps)

    print("\n[done]")


if __name__ == "__main__":
    main()
