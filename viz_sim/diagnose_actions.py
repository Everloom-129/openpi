"""Per-dim diagnostic plot for a recorded action stream.

Given a (T, D) array of policy actions, write a single figure with one row
per action dim showing:

    [ time series ]   [ histogram ]
       with overlays:
         * ±1 clip lines (red dashed)
         * training-distribution mean ± std band (green) if norm_stats provided
         * percentile clip-fraction annotation

Used by `run_pi0_policy_sim.py` to dump a post-mortem after each rollout so
you can spot dims that saturate, sit constant, or drift away from the
training distribution. Lives in viz_sim/ (sim-side env) and depends only on
numpy + matplotlib + json so it runs in robocasa_sim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Dim labels per config. Order matches the model output (post-Unnormalize).
# robocasa layout B: [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
DIM_LABELS = {
    "pi05_robocasa365": [
        "eef_x", "eef_y", "eef_z",
        "eef_rx", "eef_ry", "eef_rz",
        "grip",
        "base_m0", "base_m1", "base_m2", "base_m3",
        "ctrl_mode",
    ],
    "pi05_droid": [
        "Δjoint_0", "Δjoint_1", "Δjoint_2", "Δjoint_3",
        "Δjoint_4", "Δjoint_5", "Δjoint_6", "grip",
    ],
    "pi0_droid": [
        "Δjoint_0", "Δjoint_1", "Δjoint_2", "Δjoint_3",
        "Δjoint_4", "Δjoint_5", "Δjoint_6", "grip",
    ],
    "pi05_libero": [
        "Δeef_x", "Δeef_y", "Δeef_z", "Δeef_rx", "Δeef_ry", "Δeef_rz", "grip",
    ],
}


def _load_train_action_stats(norm_stats_path: str | Path, n_dims: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull `actions.mean`/`actions.std` from a norm_stats.json. Returns None
    if path missing or schema doesn't match."""
    p = Path(norm_stats_path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())["norm_stats"]["actions"]
        m = np.asarray(d["mean"], dtype=np.float32)
        s = np.asarray(d["std"], dtype=np.float32)
    except (KeyError, json.JSONDecodeError):
        return None
    if m.shape[0] < n_dims or s.shape[0] < n_dims:
        return None
    return m[:n_dims], s[:n_dims]


def plot_action_timeseries(
    actions: np.ndarray,
    out_path: str | Path,
    *,
    config: str | None = None,
    norm_stats_path: str | Path | None = None,
    clip_range: tuple[float, float] = (-1.0, 1.0),
    title: str | None = None,
) -> None:
    """Save a per-dim time-series + histogram diagnostic.

    Args:
        actions: (T, D) recorded model output (post-unnormalization, pre-clip).
        out_path: png path.
        config: name like 'pi05_robocasa365' for dim labels.
        norm_stats_path: optional path to norm_stats.json with `actions.{mean,std}`.
        clip_range: where to draw the saturation lines.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"expected (T, D) array, got {actions.shape}")
    T, D = actions.shape
    labels = DIM_LABELS.get(config or "", [f"dim_{i}" for i in range(D)])
    if len(labels) < D:
        labels = labels + [f"dim_{i}" for i in range(len(labels), D)]
    train_stats = _load_train_action_stats(norm_stats_path, D) if norm_stats_path else None

    fig_h = max(2.2, 1.0 * D)
    fig, axes = plt.subplots(D, 2, figsize=(13, fig_h),
                             gridspec_kw={"width_ratios": [4, 1]}, squeeze=False)
    fig.suptitle(title or f"Action diagnostic ({config}, T={T} steps)", fontsize=11)

    lo, hi = clip_range
    t = np.arange(T)
    for i in range(D):
        ax_t, ax_h = axes[i]
        col = actions[:, i]
        # Time series
        ax_t.plot(t, col, lw=0.8, color="#2266aa")
        ax_t.axhline(lo, color="red", ls="--", lw=0.6, alpha=0.6)
        ax_t.axhline(hi, color="red", ls="--", lw=0.6, alpha=0.6)
        ax_t.axhline(0, color="grey", ls=":", lw=0.4)
        # Training distribution band
        if train_stats is not None:
            m, s = float(train_stats[0][i]), float(train_stats[1][i])
            ax_t.axhspan(m - s, m + s, color="#22aa44", alpha=0.12)
            ax_t.axhline(m, color="#22aa44", ls="-", lw=0.6, alpha=0.6)
        ax_t.set_ylabel(labels[i], fontsize=8)
        ax_t.tick_params(labelsize=7)
        # Annotate clip fraction + per-dim summary
        clip_frac = float(((col <= lo) | (col >= hi)).mean())
        const_score = float(col.std()) < 1e-6
        annot = f"μ={col.mean():+.2f} σ={col.std():.2f} clip={clip_frac*100:.0f}%"
        if const_score:
            annot += " [CONST]"
        ax_t.text(0.99, 0.95, annot, transform=ax_t.transAxes,
                  ha="right", va="top", fontsize=7,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        if i < D - 1:
            ax_t.set_xticklabels([])
        else:
            ax_t.set_xlabel("step", fontsize=8)
        # Auto-zoom y if values fit comfortably inside [-1.5, 1.5]; else show outliers
        ymax = max(1.1, float(np.abs(col).max()) * 1.05)
        ax_t.set_ylim(-ymax, ymax)

        # Histogram
        ax_h.hist(col, bins=40, orientation="horizontal",
                  color="#2266aa", alpha=0.8, edgecolor="none")
        ax_h.axhline(lo, color="red", ls="--", lw=0.6, alpha=0.6)
        ax_h.axhline(hi, color="red", ls="--", lw=0.6, alpha=0.6)
        if train_stats is not None:
            m, s = float(train_stats[0][i]), float(train_stats[1][i])
            ax_h.axhspan(m - s, m + s, color="#22aa44", alpha=0.12)
        ax_h.set_ylim(-ymax, ymax)
        ax_h.tick_params(labelsize=7, left=False, labelleft=False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[diagnose_actions] wrote {out_path}  (T={T}, D={D})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to .npz with key 'actions' shaped (T, D)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--norm_stats", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arr = np.load(args.npz)["actions"]
    out = args.out or args.npz.replace(".npz", ".png")
    plot_action_timeseries(arr, out, config=args.config, norm_stats_path=args.norm_stats)
