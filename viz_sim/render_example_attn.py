"""Render example attention rollouts: 3 ckpts x 5 tasks = 15 PNGs.

For each (model, task), load ep_000.npz, sample 6 evenly-spaced snapshots,
render ext + wrist frames with attention overlay (layer 7, head-mean) and
save to results/example_attn_{model}_{task}.png.
"""
from __future__ import annotations
import pathlib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
ATTN_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365")
RESULTS = REPO / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODELS = ["pi05_droid", "pi05_libero", "pi05_robocasa365"]
TASKS = ["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"]
DISPLAY_LAYER = 7
N_SAMPLES = 6


def overlay_attn(img: np.ndarray, attn256: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    grid = attn256.reshape(16, 16)
    up = cv2.resize(grid, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, color, alpha, 0)


def render(model: str, task: str) -> bool:
    ep_path = ATTN_DIR / model / task / "ep_000.npz"
    if not ep_path.exists():
        print(f"[skip] {model}/{task}: no ep_000.npz")
        return False
    d = np.load(ep_path, allow_pickle=True)
    attn = d["attn_stacks"]      # (T, 18, 8, 512)
    ext = d["ext_frames"]        # (T, 224, 224, 3)
    wrist = d["wrist_frames"]    # (T, 224, 224, 3)
    steps = d["frame_steps"]
    if attn.size == 0 or ext.size == 0:
        print(f"[skip] {model}/{task}: empty attn or frames")
        return False
    T = min(attn.shape[0], ext.shape[0], wrist.shape[0])
    if T == 0:
        print(f"[skip] {model}/{task}: no snapshots")
        return False

    idxs = np.linspace(0, T - 1, N_SAMPLES).round().astype(int)
    # head-mean of layer DISPLAY_LAYER → (T, 512)
    t2i = attn[:, DISPLAY_LAYER].mean(axis=1)

    fig, axes = plt.subplots(2, N_SAMPLES, figsize=(2.4 * N_SAMPLES, 5.0), squeeze=False)
    success = bool(d["success"])
    fig.suptitle(
        f"{model} / {task}  ep_000  success={success}  reward={float(d['final_reward']):.2f}  "
        f"(layer {DISPLAY_LAYER} head-mean)",
        fontsize=11,
    )
    for c, i in enumerate(idxs):
        a = t2i[i]
        axes[0][c].imshow(overlay_attn(ext[i], a[:256]))
        axes[0][c].set_title(f"step {int(steps[i])}", fontsize=9)
        axes[0][c].axis("off")
        axes[1][c].imshow(overlay_attn(wrist[i], a[256:]))
        axes[1][c].axis("off")
    axes[0][0].set_ylabel("ext", fontsize=10)
    axes[1][0].set_ylabel("wrist", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / f"example_attn_{model}_{task}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")
    return True


def main():
    n_ok = 0
    for m in MODELS:
        for t in TASKS:
            if render(m, t):
                n_ok += 1
    print(f"[done] {n_ok}/{len(MODELS) * len(TASKS)} rendered")


if __name__ == "__main__":
    main()
