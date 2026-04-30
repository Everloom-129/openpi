"""Render an attention-overlay video for every SUCCESSFUL perturbation episode.

Walks `/mnt/sda/edward/projects/robocasa_365_perturb/{task}/{cond}/ep_*.npz`
and writes one animated WebP per `success=True` episode to
`results/video_perturb/successes/{task}__{cond}__ep{NNN}.webp`.

Each video shows two rows (ext / wrist) with layer-7 head-mean attention
overlay; the header strip displays task, condition, episode, seed, and
final max_reward.

Usage:
    .venv/bin/python viz_sim/render_perturb_success.py [--task Lift]
"""
from __future__ import annotations
import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
PERT_DIR = pathlib.Path("/mnt/sda/edward/projects/robocasa_365_perturb")
OUT_DIR = REPO / "results" / "video_perturb" / "successes"

DISPLAY_LAYER = 7
TILE = 224
HEADER = 28
FPS = 8


def overlay_attn(img_rgb: np.ndarray, attn256: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    grid = attn256.reshape(16, 16).astype(np.float32)
    up = cv2.resize(grid, img_rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    norm = (up - up.min()) / (up.max() - up.min() + 1e-8)
    color_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, color_rgb, alpha, 0)


def label_strip(text: str, width: int, height: int = HEADER,
                bg=(40, 40, 40), font_scale: float = 0.5) -> np.ndarray:
    s = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(s, text, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return s


def render_one(npz_path: pathlib.Path) -> pathlib.Path | None:
    d = np.load(npz_path, allow_pickle=True)
    if not bool(d["success"]):
        return None
    if d["attn_stacks"].size == 0 or d["ext_frames"].size == 0:
        return None

    T = min(d["attn_stacks"].shape[0], d["ext_frames"].shape[0], d["wrist_frames"].shape[0])
    ext = d["ext_frames"][:T]
    wrist = d["wrist_frames"][:T]
    attn = d["attn_stacks"][:T, DISPLAY_LAYER].mean(axis=1)  # (T, 512)
    steps = d["frame_steps"][:T]

    task, cond = str(d["task"]), str(d["condition"])
    seed = int(d["seed"]); ep = int(npz_path.stem.split("_")[-1])
    rmax = float(d["max_reward"])
    prompt = str(d["prompt"])

    W = TILE * 2  # ext + wrist side-by-side
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{task}__{cond}__ep{ep:03d}.webp"

    top1 = label_strip(f'{task} | {cond} | ep{ep:03d} (seed {seed}) | succ=True r_max={rmax:.2f}',
                       W, bg=(15, 15, 15), font_scale=0.5)
    top2 = label_strip(f'prompt: "{prompt}"  layer={DISPLAY_LAYER} head=mean',
                       W, bg=(30, 30, 30), font_scale=0.45)
    col_lab = np.hstack([
        label_strip("ext", TILE, bg=(50, 50, 50), font_scale=0.5),
        label_strip("wrist", TILE, bg=(50, 50, 50), font_scale=0.5),
    ])

    frames = []
    for t in range(T):
        ext_o = overlay_attn(ext[t], attn[t, :256])
        wrist_o = overlay_attn(wrist[t], attn[t, 256:])
        body = np.hstack([ext_o, wrist_o])
        footer = label_strip(f"sim_step={int(steps[t])}", W, bg=(20, 20, 20))
        full = np.vstack([top1, top2, col_lab, body, footer])
        frames.append(Image.fromarray(full))

    duration = int(round(1000 / FPS))
    frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, method=6, quality=82)
    print(f"[succ-vid] wrote {out.name}  ({T} frames)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="Restrict to one task name")
    args = ap.parse_args()

    if not PERT_DIR.exists():
        print(f"[succ-vid] {PERT_DIR} not found")
        return

    tasks = [args.task] if args.task else sorted(p.name for p in PERT_DIR.iterdir()
                                                  if p.is_dir() and not p.name.startswith("_"))
    n_total = n_succ = 0
    for t in tasks:
        for npz in sorted((PERT_DIR / t).rglob("ep_*.npz")):
            n_total += 1
            if render_one(npz) is not None:
                n_succ += 1
    print(f"[succ-vid] {n_succ}/{n_total} successful episodes rendered to {OUT_DIR}")


if __name__ == "__main__":
    main()
