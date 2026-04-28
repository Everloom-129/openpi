"""Combine per-model/per-task eval npz dumps into one big PNG + HTML summary.

Reads from /mnt/sda/edward/projects/robocasa_365/{model}/{task}/ep_*.npz
Writes:
    results/combined_summary.json   — flat stats per (model, task)
    results/combined_{task}.png     — per-task grid (rows=models, cols=stats/attn/frames)
    results/combined.html           — index linking all per-task PNGs + a global table

Resumable: re-run anytime; uses whatever episodes exist on disk.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
ATTN_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365")
RESULTS = REPO / "results"

MODELS = ["pi05_droid", "pi05_libero", "pi05_robocasa365", "gr00t_n15_robocasa365"]
DISPLAY_LAYER = 7  # mid-network; head-mean for the displayed map


def discover_tasks() -> list[str]:
    tasks = set()
    for m in MODELS:
        d = ATTN_DIR / m
        if d.exists():
            for sub in d.iterdir():
                if sub.is_dir() and any(sub.glob("ep_*.npz")):
                    tasks.add(sub.name)
    return sorted(tasks)


def load_model_task(model: str, task: str) -> dict | None:
    d = ATTN_DIR / model / task
    eps = sorted(d.glob("ep_*.npz"))
    if not eps:
        return None
    rewards, steps, successes = [], [], []
    sum_attn_ext = np.zeros(256, dtype=np.float64)
    sum_attn_wrist = np.zeros(256, dtype=np.float64)
    n_attn = 0
    sample_ext, sample_wrist, sample_attn = None, None, None
    for ep_path in eps:
        try:
            data = np.load(ep_path, allow_pickle=True)
        except Exception:
            continue
        rewards.append(float(data["final_reward"]))
        steps.append(int(data["steps"]))
        successes.append(bool(data["success"]))
        attn = data["attn_stacks"]
        if attn.ndim == 4 and attn.size:
            t2i = attn[:, DISPLAY_LAYER].mean(axis=1)
            sum_attn_ext += t2i[:, :256].sum(axis=0)
            sum_attn_wrist += t2i[:, 256:].sum(axis=0)
            n_attn += t2i.shape[0]
            if sample_ext is None and data["ext_frames"].size:
                sample_ext = data["ext_frames"][len(data["ext_frames"]) // 2]
                sample_wrist = data["wrist_frames"][len(data["wrist_frames"]) // 2]
                sample_attn = t2i[len(t2i) // 2]
    return {
        "model": model, "task": task,
        "n_episodes": len(eps),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "n_successes": int(np.sum(successes)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_attn_ext": (sum_attn_ext / max(n_attn, 1)).astype(np.float32),
        "mean_attn_wrist": (sum_attn_wrist / max(n_attn, 1)).astype(np.float32),
        "sample_ext": sample_ext, "sample_wrist": sample_wrist,
        "sample_attn": sample_attn.astype(np.float32) if sample_attn is not None else None,
    }


def overlay_attn(img, attn256, alpha=0.45):
    import cv2
    grid = attn256.reshape(16, 16)
    up = cv2.resize(grid, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, color, alpha, 0)


def render_task_png(task: str, rows: list[dict], out_path: pathlib.Path):
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3 * len(rows)), squeeze=False)
    fig.suptitle(f"task = {task}", fontsize=14)
    for r, m in enumerate(rows):
        ax = axes[r][0]; ax.axis("off")
        ax.text(0.0, 0.5,
                f"{m['model']}\n\n"
                f"episodes: {m['n_episodes']}\n"
                f"success: {m['n_successes']}/{m['n_episodes']} ({m['success_rate']*100:.0f}%)\n"
                f"mean reward: {m['mean_reward']:.3f}\n"
                f"mean steps: {m['mean_steps']:.1f}",
                fontsize=10, family="monospace", verticalalignment="center")
        for c, key in enumerate(("mean_attn_ext", "mean_attn_wrist"), start=1):
            ax = axes[r][c]
            ax.imshow(m[key].reshape(16, 16), cmap="jet")
            ax.set_title(f"mean {key.split('_')[-1]} attn (L{DISPLAY_LAYER})", fontsize=9)
            ax.axis("off")
        if m["sample_ext"] is not None:
            axes[r][3].imshow(overlay_attn(m["sample_ext"], m["sample_attn"][:256]))
            axes[r][3].set_title("ext + attn (mid-ep)", fontsize=9); axes[r][3].axis("off")
            axes[r][4].imshow(overlay_attn(m["sample_wrist"], m["sample_attn"][256:]))
            axes[r][4].set_title("wrist + attn", fontsize=9); axes[r][4].axis("off")
        else:
            for c in (3, 4):
                axes[r][c].axis("off")
                axes[r][c].text(0.5, 0.5, "no data", ha="center", va="center")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_html(by_task: dict[str, list[dict]], out_path: pathlib.Path):
    table_rows = []
    for task, rows in by_task.items():
        for r in rows:
            table_rows.append(
                f"<tr><td>{task}</td><td><b>{r['model']}</b></td>"
                f"<td>{r['n_episodes']}</td>"
                f"<td>{r['n_successes']} ({r['success_rate']*100:.1f}%)</td>"
                f"<td>{r['mean_reward']:.3f}</td>"
                f"<td>{r['mean_steps']:.1f}</td></tr>")
    sections = []
    for task in by_task:
        sections.append(
            f'<h2>{task}</h2><img src="combined_{task}.png">')
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>robocasa_365 multi-task eval</title>
<style>body{{font-family:system-ui;margin:24px;max-width:1400px}}
table{{border-collapse:collapse;margin-bottom:16px}}td,th{{border:1px solid #ccc;padding:6px 12px}}
th{{background:#eee}}img{{max-width:100%;border:1px solid #ddd;margin-bottom:24px}}</style></head>
<body><h1>Per-(model, task) eval summary</h1>
<table><tr><th>task</th><th>model</th><th>episodes</th><th>success</th><th>mean reward</th><th>mean steps</th></tr>
{''.join(table_rows)}
</table>
{''.join(sections)}
</body></html>"""
    out_path.write_text(html)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks()
    if not tasks:
        print("[viz] no tasks with data")
        return
    by_task: dict[str, list[dict]] = {}
    flat: list[dict] = []
    for task in tasks:
        rows = []
        for m in MODELS:
            d = load_model_task(m, task)
            if d is None:
                continue
            rows.append(d)
            flat.append({k: v for k, v in d.items()
                         if k not in ("mean_attn_ext", "mean_attn_wrist",
                                      "sample_ext", "sample_wrist", "sample_attn")})
            print(f"[viz] {m}/{task}: {d['n_episodes']} eps, success={d['success_rate']*100:.0f}%")
        by_task[task] = rows
        render_task_png(task, rows, RESULTS / f"combined_{task}.png")
    (RESULTS / "combined_summary.json").write_text(json.dumps(flat, indent=2))
    render_html(by_task, RESULTS / "combined.html")
    print("[viz] wrote", RESULTS / "combined.html")


if __name__ == "__main__":
    main()
