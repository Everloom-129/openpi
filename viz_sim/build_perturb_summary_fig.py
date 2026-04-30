"""Per-task × per-condition perturbation summary figure (A + B).

Reads `/mnt/sda/edward/projects/robocasa_365_perturb/{task}/{cond}/ep_*.npz`
and writes a single PNG with four heatmap panels:

    [ A succ rate (5×7)   ]   [ A r_max avg (5×7) ]
    [ B Δeef_z (5×6)      ]   [ B Δgrip% (5×6)    ]

A panels include the baseline column; B panels are computed as
`(perturb − baseline)` so the baseline column is omitted. r_max is the
mean of `max_reward` across episodes (gives a smoother view than binary
success when N is small).

Output: `results/perturb_summary.png`.
"""
from __future__ import annotations
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PERT_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_perturb")
OUT = REPO / "results" / "perturb_summary.png"

TASKS = ["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"]
CONDS = ["baseline",
         "ext_zero_max", "ext_strengthen_max", "ext_zero_min",
         "wrist_zero_max", "wrist_strengthen_max", "wrist_zero_min"]
PERT_CONDS = CONDS[1:]


def _load_cell(task: str, cond: str) -> dict | None:
    """Aggregate summary statistics across episodes in one (task, cond) cell."""
    d = PERT_DIR / task / cond
    eps = sorted(d.glob("ep_*.npz"))
    if not eps:
        return None
    succ_list, rmax_list, ez_means, gpos_means = [], [], [], []
    for p in eps:
        z = np.load(p, allow_pickle=True)
        succ_list.append(int(bool(z["success"])))
        rmax_list.append(float(z["max_reward"]))
        a = z["action_taken"]
        if a.shape[1] >= 11:
            ez_means.append(float(a[:, 2].mean()))
            gpos_means.append(float((a[:, 10] > 0).mean()))
    return {
        "n": len(eps),
        "succ_rate": float(np.mean(succ_list)),
        "r_max_avg": float(np.mean(rmax_list)),
        "eef_z_mean": float(np.mean(ez_means)) if ez_means else 0.0,
        "grip_pos_rate": float(np.mean(gpos_means)) if gpos_means else 0.0,
    }


def _heatmap(ax, M, row_labels, col_labels, *, title, cmap, vmin=None, vmax=None,
             fmt="{:.2f}", center_zero=False):
    if center_zero:
        m = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
        if vmin is None: vmin = -m
        if vmax is None: vmax = m
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=8)
            else:
                # auto text color based on cell intensity
                norm = (v - (vmin if vmin is not None else np.nanmin(M))) / max(
                    (vmax if vmax is not None else np.nanmax(M))
                    - (vmin if vmin is not None else np.nanmin(M)),
                    1e-9,
                )
                color = "white" if cmap == "viridis" and norm < 0.55 else "black"
                if cmap == "RdBu_r":
                    color = "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def main():
    n_t, n_c = len(TASKS), len(CONDS)
    n_p = len(PERT_CONDS)

    # Allocate matrices, NaN where data missing.
    A_succ  = np.full((n_t, n_c), np.nan)
    A_rmax  = np.full((n_t, n_c), np.nan)
    base_z  = np.full(n_t, np.nan)
    base_g  = np.full(n_t, np.nan)
    cell_z  = np.full((n_t, n_c), np.nan)
    cell_g  = np.full((n_t, n_c), np.nan)

    for i, t in enumerate(TASKS):
        for j, c in enumerate(CONDS):
            r = _load_cell(t, c)
            if r is None:
                continue
            A_succ[i, j] = r["succ_rate"]
            A_rmax[i, j] = r["r_max_avg"]
            cell_z[i, j] = r["eef_z_mean"]
            cell_g[i, j] = r["grip_pos_rate"]
            if c == "baseline":
                base_z[i] = r["eef_z_mean"]
                base_g[i] = r["grip_pos_rate"]

    B_dz = cell_z[:, 1:] - base_z[:, None]
    B_dg = cell_g[:, 1:] - base_g[:, None]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    fig.suptitle("Perturbation eval summary (A: outcome, B: action-distribution shift vs baseline)",
                 fontsize=12)

    # A: succ rate
    _heatmap(axes[0, 0], A_succ, TASKS, CONDS,
             title="A1  Success rate (succ / N)", cmap="viridis",
             vmin=0, vmax=max(0.05, np.nanmax(A_succ) if not np.all(np.isnan(A_succ)) else 0.05),
             fmt="{:.2f}")
    # A: r_max avg
    _heatmap(axes[0, 1], A_rmax, TASKS, CONDS,
             title="A2  Avg max_reward (per-cell mean over episodes)", cmap="viridis",
             vmin=0, vmax=max(0.05, np.nanmax(A_rmax) if not np.all(np.isnan(A_rmax)) else 0.05),
             fmt="{:.2f}")
    # B: Δeef_z
    _heatmap(axes[1, 0], B_dz, TASKS, PERT_CONDS,
             title="B1  Δ eef_z (perturb − baseline; >0 = less downward)",
             cmap="RdBu_r", center_zero=True, fmt="{:+.3f}")
    # B: Δgrip
    _heatmap(axes[1, 1], B_dg, TASKS, PERT_CONDS,
             title="B2  Δ gripper close-rate (perturb − baseline)",
             cmap="RdBu_r", center_zero=True, fmt="{:+.2f}")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"[perturb-summary] wrote {OUT}")


if __name__ == "__main__":
    main()
