"""Export DeLock CPG result figures and an animated webp.

Takes a saved denoising trajectory `.npz` (the kind `viz_sim/run_pi0_policy_sim.py`
already writes via `diagnose_actions.py`, augmented with one column per
guidance scale `w`) and produces:

  * `<out>/cpg_lines.png`     — per-action-dim line plot of the final action
                                chunk, one line per `w` (and optionally a
                                ground-truth or τ⁺-only baseline).
  * `<out>/cpg_denoising.webp` — animated 2-row figure: top row shows the
                                action-chunk evolution along denoising
                                steps; bottom row shows ‖a^t_w − a^t_pos‖
                                vs t for each w (the contrastive distance
                                from vanilla τ⁺ sampling).
  * `<out>/cpg_summary.json`   — μ/σ of action chunk per w, plus argmax-w
                                per dim (which guidance setting moves
                                that dim the most relative to vanilla).

The npz schema is dict-like:
  {
     "trajectory": ndarray of shape (n_w, n_steps, action_horizon, action_dim),
     "w_values":   ndarray of shape (n_w,),
     "prompt_pos": str,
     "prompt_neg": str,
     "task":       str (optional),
     "ckpt":       str (optional),
     "gt_action":  ndarray of shape (action_horizon, action_dim) (optional),
  }

This module has no openpi imports — it is pure numpy + matplotlib + PIL.
That keeps the export feature usable from any post-hoc analysis context
(notebooks, CI artifact builders, etc.) without dragging in the model.

CLI:
  uv run python baseline/delock/export_cpg_results.py path/to/cpg_run.npz \
       --out-dir baseline/delock/results/run_42

For a smoke run that doesn't need real data, see
`baseline/delock/tests/test_export.py::test_export_smoke_synthetic`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return str(value.item()) if value.ndim == 0 else str(value.tolist())
    return str(value)


def _load(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as f:
        out = {k: f[k] for k in f.files}
    return out


def _final_chunks_per_w(traj: np.ndarray) -> np.ndarray:
    """traj: (n_w, n_steps, H, D) → (n_w, H, D), the action chunk after the
    last Euler step (t ≈ 0)."""
    return traj[:, -1, :, :]


def render_lines_png(out_path: Path, data: dict) -> Path:
    """Per-dim line plot of the final action chunk, one curve per w."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = np.asarray(data["trajectory"])
    w_values = np.asarray(data["w_values"]).reshape(-1)
    final = _final_chunks_per_w(traj)            # (n_w, H, D)
    n_w, H, D = final.shape

    fig, axes = plt.subplots(D, 1, figsize=(8, 1.6 * D), sharex=True)
    if D == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis", n_w)
    for d in range(D):
        ax = axes[d]
        for i, w in enumerate(w_values):
            ax.plot(np.arange(H), final[i, :, d], color=cmap(i), lw=1.6,
                    label=f"w={float(w):g}")
        if "gt_action" in data and data["gt_action"] is not None:
            gt = np.asarray(data["gt_action"])
            if gt.shape[-1] > d:
                ax.plot(np.arange(min(H, gt.shape[0])), gt[:H, d],
                        ":k", lw=1.0, label="GT")
        ax.set_ylabel(f"a[{d}]", fontsize=8)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.set_title(
                f"DeLock CPG: action chunk vs. guidance scale w\n"
                f"τ+={_safe_str(data.get('prompt_pos'))!r}   "
                f"τ-={_safe_str(data.get('prompt_neg'))!r}",
                fontsize=9,
            )
            ax.legend(loc="upper right", fontsize=7, ncol=min(n_w, 4))
    axes[-1].set_xlabel("action step (within chunk)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render_denoising_webp(out_path: Path, data: dict, fps: int = 6) -> Path:
    """Animated 2-row figure showing the per-step denoising trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    traj = np.asarray(data["trajectory"])         # (n_w, T, H, D)
    w_values = np.asarray(data["w_values"]).reshape(-1)
    n_w, T, H, D = traj.shape

    # Reference for the bottom row: pick the trajectory with w closest to 1.0
    # as "vanilla τ+ sampling" (since v_cpg|_{w=1} ≡ v_pos by construction).
    w_ref_idx = int(np.argmin(np.abs(w_values - 1.0)))

    cmap = plt.get_cmap("viridis", n_w)
    frames = []
    # Pick a single action dim to display in row 1 — first dim that varies most.
    flat = traj.reshape(n_w, T * H, D)
    var_per_dim = flat.var(axis=(0, 1))
    d_show = int(np.argmax(var_per_dim))

    for t in range(T):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.5, 5.0))

        # Row 0: action[d_show, :] over the chunk at the current denoising step,
        # one line per w.
        for i, w in enumerate(w_values):
            ax0.plot(np.arange(H), traj[i, t, :, d_show], color=cmap(i), lw=1.4,
                     label=f"w={float(w):g}")
        ax0.set_xlabel("chunk step")
        ax0.set_ylabel(f"a[{d_show}]")
        ax0.set_title(f"denoising step {t + 1}/{T}", fontsize=9)
        ax0.set_ylim(traj[..., d_show].min() - 0.1, traj[..., d_show].max() + 0.1)
        ax0.legend(loc="upper right", fontsize=7, ncol=min(n_w, 4))

        # Row 1: ‖a_w(t) - a_{w=1}(t)‖ vs t, one line per w. Builds up over
        # the animation so the user can see when each w starts diverging
        # from vanilla τ+.
        for i, w in enumerate(w_values):
            diff = traj[i, : t + 1] - traj[w_ref_idx, : t + 1]
            norms = np.linalg.norm(diff.reshape(t + 1, -1), axis=1)
            ax1.plot(np.arange(t + 1), norms, color=cmap(i), lw=1.4)
        ax1.set_xlabel("denoising step")
        ax1.set_ylabel(r"$\|a_w(t) - a_{w=1}(t)\|$")
        ax1.set_xlim(0, T - 1)
        ymax = max(0.01, np.linalg.norm(
            (traj - traj[w_ref_idx:w_ref_idx + 1]).reshape(n_w, -1), axis=1
        ).max())
        ax1.set_ylim(0, ymax * 1.05)
        ax1.set_title("contrastive distance from vanilla (w=1) sampling", fontsize=9)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    if not frames:
        raise RuntimeError("No frames produced — trajectory had zero steps?")
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, format="WEBP", quality=85,
    )
    return out_path


def render_summary_json(out_path: Path, data: dict) -> Path:
    traj = np.asarray(data["trajectory"])
    w_values = np.asarray(data["w_values"]).reshape(-1).tolist()
    final = _final_chunks_per_w(traj)             # (n_w, H, D)

    summary = {
        "n_w": int(final.shape[0]),
        "w_values": [float(w) for w in w_values],
        "action_horizon": int(final.shape[1]),
        "action_dim": int(final.shape[2]),
        "prompt_pos": _safe_str(data.get("prompt_pos")),
        "prompt_neg": _safe_str(data.get("prompt_neg")),
        "task": _safe_str(data.get("task")),
        "ckpt": _safe_str(data.get("ckpt")),
        "per_w_action_mean": final.mean(axis=(1, 2)).tolist(),
        "per_w_action_std": final.std(axis=(1, 2)).tolist(),
    }

    # argmax-w per dim: which w produces the largest L1 distance from the
    # w=1 baseline at the final chunk? Useful for spotting which dims are
    # actually steerable.
    if "1" in [f"{w}" for w in summary["w_values"]] or 1.0 in w_values:
        ref_idx = int(np.argmin(np.abs(np.asarray(w_values) - 1.0)))
        ref = final[ref_idx]
        per_dim_diffs = np.abs(final - ref).mean(axis=1)  # (n_w, D)
        argmax_w = per_dim_diffs.argmax(axis=0)
        summary["argmax_w_per_dim"] = [
            {"dim": int(d), "w": float(w_values[int(argmax_w[d])])}
            for d in range(per_dim_diffs.shape[1])
        ]

    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def export_all(npz_path: Path, out_dir: Path, *, fps: int = 6) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load(npz_path)
    if "trajectory" not in data or "w_values" not in data:
        raise KeyError(
            "Expected keys 'trajectory' and 'w_values' in the npz; "
            f"got: {list(data.keys())}"
        )
    return {
        "lines_png": render_lines_png(out_dir / "cpg_lines.png", data),
        "denoising_webp": render_denoising_webp(out_dir / "cpg_denoising.webp", data, fps=fps),
        "summary_json": render_summary_json(out_dir / "cpg_summary.json", data),
    }


def _cli():
    ap = argparse.ArgumentParser(description="Export DeLock CPG result figures.")
    ap.add_argument("npz", type=Path, help="Path to the saved CPG run npz.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: alongside the input as <stem>_export/).")
    ap.add_argument("--fps", type=int, default=6, help="Animation FPS (default 6).")
    args = ap.parse_args()
    out = args.out_dir or args.npz.with_suffix("").with_name(args.npz.stem + "_export")
    paths = export_all(args.npz, out, fps=args.fps)
    print("Exported:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
