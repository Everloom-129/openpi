"""Render the 5-task perturbation comparison animations.

Per task: 7-column animated WebP showing ep_000 of each condition with
attention overlay. Cols: baseline + 6 perturbations. Rows: ext + wrist.

Output: results/video_perturb/{task}.webp
"""
from __future__ import annotations
import pathlib
import numpy as np
import cv2
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
PERT_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_perturb")
OUT_DIR = REPO / "results" / "video_perturb"

TASKS = ["Lift", "Stack", "Door", "PickPlaceCan", "NutAssemblySquare"]
CONDITIONS = [
    "baseline",
    "ext_zero_max", "ext_strengthen_max", "ext_zero_min",
    "wrist_zero_max", "wrist_strengthen_max", "wrist_zero_min",
]
DISPLAY_LAYER = 7
TILE = 224
HEADER = 28
FPS = 8
EP = 0


def overlay_attn_rgb(img_rgb, attn256, alpha=0.45):
    grid = attn256.reshape(16, 16).astype(np.float32)
    up = cv2.resize(grid, img_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_rgb, alpha, 0)


def label_strip(text, width, height=HEADER, bg=(40, 40, 40), font_scale=0.5):
    s = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(s, text, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return s


def col_header(cond, d, width):
    line1 = label_strip(cond, width, bg=(50, 50, 50), font_scale=0.45)
    if d is None:
        line2 = label_strip("(no data)", width)
    else:
        line2 = label_strip(f"succ={int(d['success'])}  rmax={d['max_reward']:.2f}", width)
    return np.vstack([line1, line2])


def load(task, condition):
    p = PERT_DIR / task / condition / f"ep_{EP:03d}.npz"
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
        "max_reward": float(d["max_reward"]),
        "prompt": str(d["prompt"]),
    }


def render_task(task):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [(c, load(task, c)) for c in CONDITIONS]
    n_list = [len(d["ext"]) for _, d in cols if d is not None]
    if not n_list:
        print(f"[perturb-vid] {task}: no data, skip")
        return False
    T = min(n_list)
    W = TILE * len(CONDITIONS)

    col_headers = np.hstack([col_header(c, d, TILE) for c, d in cols])
    prompt = next((d["prompt"] for _, d in cols if d is not None), "")
    top = label_strip(f'task={task}  prompt="{prompt}"  layer={DISPLAY_LAYER}',
                      W, bg=(15, 15, 15), font_scale=0.55)

    frames = []
    for t in range(T):
        ext_row, wrist_row = [], []
        for _, d in cols:
            if d is None:
                blank = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                ext_row.append(blank); wrist_row.append(blank); continue
            ext_row.append(overlay_attn_rgb(d["ext"][t], d["attn"][t, :256]))
            wrist_row.append(overlay_attn_rgb(d["wrist"][t], d["attn"][t, 256:]))
        ext_row = np.hstack(ext_row); wrist_row = np.hstack(wrist_row)
        step_idx = next((d["steps"][t] for _, d in cols if d is not None), 0)
        footer = label_strip(f"step={int(step_idx)}", W, bg=(20, 20, 20))
        full = np.vstack([top, col_headers, ext_row, wrist_row, footer])
        frames.append(Image.fromarray(full))

    out = OUT_DIR / f"{task}.webp"
    duration = int(round(1000 / FPS))
    frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, method=6, quality=82)
    print(f"[perturb-vid] wrote {out}  ({T} frames @ {FPS} fps)")
    return True


def main():
    n = sum(1 for t in TASKS if render_task(t))
    print(f"[perturb-vid] {n}/{len(TASKS)} written to {OUT_DIR}")


if __name__ == "__main__":
    main()
