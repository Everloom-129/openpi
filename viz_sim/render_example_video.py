"""Render side-by-side rollout animations: 5 tasks → 5 animated WebP files.

Each frame layout (per task, per timestep):
    | pi05_droid ext  | pi05_libero ext  | pi05_robocasa365 ext   |
    | pi05_droid wrist| pi05_libero wrist| pi05_robocasa365 wrist |

Attention (layer 7, head-mean) is overlaid on each ext/wrist tile.
Output: results/video/{task}.webp  (animated, VSCode-previewable)

Usage:
    uv run python viz_sim/render_example_video.py            # render all 5 tasks
    uv run python viz_sim/render_example_video.py --task Lift  # single task

Also exposed as `render_for_task(task)` so eval_runner.py can call it
automatically after writing `_summary.json`.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import cv2
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
ATTN_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365")
OUT_DIR = REPO / "results" / "video"

MODELS = ["pi05_droid", "pi05_libero", "pi05_robocasa365"]
TASKS = ["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"]
DISPLAY_LAYER = 7
TILE = 224
HEADER = 28
FPS = 8


def overlay_attn_rgb(img_rgb: np.ndarray, attn256: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    grid = attn256.reshape(16, 16).astype(np.float32)
    up = cv2.resize(grid, img_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_rgb, alpha, 0)


def load(model: str, task: str) -> dict | None:
    p = ATTN_DIR / model / task / "ep_000.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if d["attn_stacks"].size == 0 or d["ext_frames"].size == 0:
        return None
    T = min(d["attn_stacks"].shape[0], d["ext_frames"].shape[0], d["wrist_frames"].shape[0])
    return {
        "ext": d["ext_frames"][:T],
        "wrist": d["wrist_frames"][:T],
        "attn": d["attn_stacks"][:T, DISPLAY_LAYER].mean(axis=1),
        "steps": d["frame_steps"][:T],
        "success": bool(d["success"]),
        "reward": float(d["final_reward"]),
        "prompt": str(d["prompt"]),
    }


def label_strip(text: str, width: int, height: int = HEADER, bg=(40, 40, 40),
                font_scale: float = 0.5) -> np.ndarray:
    strip = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(strip, text, (6, height - 9),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def col_header(model: str, d: dict | None, width: int) -> np.ndarray:
    """Two-line column header: model name (line 1) + outcome (line 2)."""
    line1 = label_strip(model, width, height=HEADER, bg=(50, 50, 50))
    if d is None:
        line2 = label_strip("(no data)", width, height=HEADER, bg=(40, 40, 40))
    else:
        line2 = label_strip(
            f"succ={int(d['success'])}  r={d['reward']:.2f}",
            width, height=HEADER, bg=(40, 40, 40),
        )
    return np.vstack([line1, line2])


def render_for_task(task: str) -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [(m, load(m, task)) for m in MODELS]
    n_frames_list = [len(d["ext"]) for _, d in cols if d is not None]
    if not n_frames_list:
        print(f"[video] {task}: no data, skip")
        return False
    T = min(n_frames_list)
    W = TILE * len(MODELS)

    # Per-column 2-line header (model name + outcome).
    col_headers = np.hstack([col_header(m, d, TILE) for m, d in cols])
    # Global top strip: task + prompt (full width, slightly larger font).
    prompt_text = next((d["prompt"] for _, d in cols if d is not None), "")
    top_strip = label_strip(
        f'task={task}  prompt="{prompt_text}"',
        W, height=HEADER, bg=(15, 15, 15), font_scale=0.55,
    )

    frames: list[Image.Image] = []
    for t in range(T):
        ext_row, wrist_row = [], []
        for _, d in cols:
            if d is None:
                blank = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                ext_row.append(blank); wrist_row.append(blank)
                continue
            ext_row.append(overlay_attn_rgb(d["ext"][t], d["attn"][t, :256]))
            wrist_row.append(overlay_attn_rgb(d["wrist"][t], d["attn"][t, 256:]))
        ext_row = np.hstack(ext_row)
        wrist_row = np.hstack(wrist_row)
        step_idx = next((d["steps"][t] for _, d in cols if d is not None), 0)
        footer = label_strip(
            f"step={int(step_idx)}  layer={DISPLAY_LAYER} head-mean",
            W, height=HEADER, bg=(20, 20, 20),
        )
        full = np.vstack([top_strip, col_headers, ext_row, wrist_row, footer])
        frames.append(Image.fromarray(full))

    out = OUT_DIR / f"{task}.webp"
    duration_ms = int(round(1000 / FPS))
    frames[0].save(
        out,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        method=6,
        quality=85,
    )
    print(f"[video] wrote {out}  ({T} frames @ {FPS} fps)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    help="Render only this task; default renders all 5.")
    args = ap.parse_args()
    targets = [args.task] if args.task else TASKS
    n = sum(1 for t in targets if render_for_task(t))
    print(f"[video] {n}/{len(targets)} written to {OUT_DIR}")


if __name__ == "__main__":
    main()
