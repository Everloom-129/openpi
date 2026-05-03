"""End-to-end smoke test for GR00T attention capture.

What it does
------------
1. Connects to a running attention-capture GR00T server (port 5555 by default).
2. Spins up a short headless robosuite sim (default: Lift + Panda).
3. Runs N sim steps, querying the policy every `horizon` steps.
4. On every chunk, snapshots (a) the parsed VLM attention payload, (b) one
   raw VLM layer, and (c) the DiT step-averaged attention.
5. Writes three artifacts to OUT_DIR:
       attn_video.mp4       — sim canvas + ext/wrist + attention overlay tiles
       attn_diagram.png     — multi-panel matplotlib figure (single chunk)
       summary.txt          — shape/check report

Run from repo root in the robocasa_sim conda env (it has the deps:
robosuite, scipy, cv2, PIL, msgpack, zmq, matplotlib):

    conda run -n robocasa_sim python viz_sim/test_gr00t_attn_e2e.py \
        --task Lift --robot Panda --steps 24 --horizon 8 \
        --out viz_sim/results/gr00t_attn_e2e

The server must already be running (`bash viz_sim/run_gr00t_server.sh` with
ATTN=1, default). The test fails fast (with a readable error) if the server
is not reachable or the attention payload is missing.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import deque
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make sure we can import the sim helpers next to this file.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse helpers from the live-sim script.
from gr00t_client import PolicyClient  # noqa: E402
from run_policy_sim_gr00t import (  # noqa: E402
    DEFAULT_VIDEO_DELTA,
    GR00T_RES_H,
    GR00T_RES_W,
    SIM_VIEW_SIZE,
    TILE_SIZE,
    _agg_heads,
    _attn_overlay_for_image,
    _ext_image_key,
    _normalize_robosuite_image,
    _resize_with_pad,
    build_env,
    compose_canvas,
    make_gr00t_obs,
    parse_attn_payload,
    render_attn_tiles,
)


# ── attention summarization ───────────────────────────────────────────────────


def _summarize_attn_payload(parsed: dict) -> dict:
    """Pick a representative VLM layer + reduced text→image vector for plotting."""
    vlm = parsed.get("vlm") or {}
    if not vlm:
        return {}
    layer_keys = sorted(vlm.keys())
    # Use a mid-network layer (≈ 60% deep) which usually has the cleanest
    # text-conditioned image grounding.
    mid_layer = layer_keys[int(len(layer_keys) * 0.6)]
    layer_attn = vlm[mid_layer]                    # (H, seq, seq)
    head_mean = layer_attn.mean(axis=0)            # (seq, seq)

    image_mask = parsed.get("image_mask")
    text_mask = parsed.get("text_mask")
    grid_thw = parsed.get("image_grid_thw")
    if image_mask is None or text_mask is None:
        return {"layer_idx": int(mid_layer), "head_mean": head_mean}

    img_idx = np.where(image_mask)[0]
    txt_idx = np.where(text_mask)[0]
    if img_idx.size == 0 or txt_idx.size == 0:
        return {"layer_idx": int(mid_layer), "head_mean": head_mean}

    per_img = head_mean[np.ix_(txt_idx, img_idx)].mean(axis=0)   # (n_img,)
    n_image = per_img.shape[0]
    if grid_thw is not None and grid_thw.shape[0] >= 2:
        counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).astype(int)
        if counts.sum() == n_image:
            ext_count = int(counts[0])
            ext_attn = per_img[:ext_count]
            wrist_attn = per_img[ext_count : ext_count + int(counts[1])]
            ext_grid = (int(grid_thw[0, 1]), int(grid_thw[0, 2]))
            wrist_grid = (int(grid_thw[1, 1]), int(grid_thw[1, 2]))
        else:
            ext_attn = per_img[: n_image // 2]
            wrist_attn = per_img[n_image // 2 :]
            ext_grid = wrist_grid = None
    else:
        ext_attn = per_img[: n_image // 2]
        wrist_attn = per_img[n_image // 2 :]
        ext_grid = wrist_grid = None

    return {
        "layer_idx": int(mid_layer),
        "n_layers": len(layer_keys),
        "head_mean": head_mean,
        "per_img": per_img,
        "ext_attn": ext_attn,
        "wrist_attn": wrist_attn,
        "ext_grid": ext_grid,
        "wrist_grid": wrist_grid,
        "n_text": int(txt_idx.size),
        "n_image": int(img_idx.size),
        "n_seq": int(image_mask.size),
    }


def _attn_to_grid(attn_1d: np.ndarray, grid_hw: tuple[int, int] | None) -> np.ndarray:
    n = int(attn_1d.shape[0])
    if grid_hw is not None and grid_hw[0] * grid_hw[1] == n:
        h_p, w_p = grid_hw
    else:
        side = int(np.ceil(np.sqrt(max(n, 1))))
        h_p, w_p = side, int(np.ceil(n / side))
    grid = np.full((h_p * w_p,), float(attn_1d.min()), dtype=np.float32)
    grid[:n] = attn_1d.astype(np.float32)
    return grid.reshape(h_p, w_p)


# ── diagram & video ───────────────────────────────────────────────────────────


def render_diagram(
    summary: dict,
    ext_img: np.ndarray,
    wrist_img: np.ndarray,
    dit_payload: dict | None,
    instruction: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)

    # Row 0: ext image, ext attn grid, wrist image, wrist attn grid
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ext_img)
    ax.set_title("ext (180×320, padded)")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    if "ext_attn" in summary:
        ax.imshow(_attn_to_grid(summary["ext_attn"], summary["ext_grid"]),
                  cmap="viridis")
        ax.set_title(f"ext text→img attn  (layer {summary['layer_idx']})")
    else:
        ax.text(0.5, 0.5, "no mask info", ha="center", va="center")
        ax.set_title("ext text→img attn")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(wrist_img)
    ax.set_title("wrist (180×320, padded)")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    if "wrist_attn" in summary:
        ax.imshow(_attn_to_grid(summary["wrist_attn"], summary["wrist_grid"]),
                  cmap="viridis")
        ax.set_title(f"wrist text→img attn  (layer {summary['layer_idx']})")
    else:
        ax.text(0.5, 0.5, "no mask info", ha="center", va="center")
        ax.set_title("wrist text→img attn")
    ax.axis("off")

    # Row 1: full head-mean attention matrix (seq × seq) — heavy but informative
    ax = fig.add_subplot(gs[1, :2])
    if "head_mean" in summary:
        im = ax.imshow(np.log1p(summary["head_mean"]), cmap="magma", aspect="auto")
        ax.set_title(
            f"VLM head-mean attention (log1p) — layer {summary['layer_idx']}/"
            f"{summary.get('n_layers', '?')}, seq={summary.get('n_seq', '?')}, "
            f"n_text={summary.get('n_text', '?')}, n_image={summary.get('n_image', '?')}"
        )
        ax.set_xlabel("key index")
        ax.set_ylabel("query index")
        plt.colorbar(im, ax=ax, fraction=0.04)
    else:
        ax.text(0.5, 0.5, "no VLM attn", ha="center", va="center")
        ax.axis("off")

    # Row 1 right half: DiT averaged attention
    ax = fig.add_subplot(gs[1, 2:])
    if dit_payload:
        dit_layer_keys = sorted(dit_payload.keys())
        # pick last DiT layer (deepest, typically most action-coupled)
        dit_layer = dit_payload[dit_layer_keys[-1]]    # (B, H, q, k)
        if dit_layer.ndim == 4:
            dit_layer = dit_layer[0]                    # (H, q, k)
        dit_mean = dit_layer.mean(axis=0)               # (q, k)
        im = ax.imshow(np.log1p(dit_mean), cmap="cividis", aspect="auto")
        ax.set_title(
            f"DiT attention (log1p) — layer {dit_layer_keys[-1]}/"
            f"{len(dit_layer_keys)}, q={dit_mean.shape[0]}, k={dit_mean.shape[1]}"
        )
        ax.set_xlabel("key index")
        ax.set_ylabel("query (action) index")
        plt.colorbar(im, ax=ax, fraction=0.04)
    else:
        ax.text(0.5, 0.5, "no DiT attn captured", ha="center", va="center")
        ax.set_title("DiT attention")
        ax.axis("off")

    # Row 2: text→image attention vector (1D bar) for ext + wrist side by side
    ax = fig.add_subplot(gs[2, :2])
    if "ext_attn" in summary:
        ax.bar(np.arange(summary["ext_attn"].size), summary["ext_attn"], width=1.0)
        ax.set_title("ext text→img attention per token (sorted by index)")
        ax.set_xlabel("image-token index (ext)")
        ax.set_ylabel("attention")
    else:
        ax.axis("off")

    ax = fig.add_subplot(gs[2, 2:])
    if "wrist_attn" in summary:
        ax.bar(np.arange(summary["wrist_attn"].size), summary["wrist_attn"],
               width=1.0, color="tab:orange")
        ax.set_title("wrist text→img attention per token")
        ax.set_xlabel("image-token index (wrist)")
        ax.set_ylabel("attention")
    else:
        ax.axis("off")

    fig.suptitle(
        f"GR00T-N1.7-DROID attention E2E — instruction: \"{instruction}\"",
        fontsize=12, y=0.995,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift")
    ap.add_argument("--robot", default="Panda", choices=["Panda", "PandaOmron"])
    ap.add_argument("--prompt", default="pick up the red cube")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gripper-thresh", type=float, default=0.5)
    ap.add_argument("--out", default="viz_sim/results/gr00t_attn_e2e",
                    help="output dir for video + diagram + summary")
    ap.add_argument("--video-fps", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_log: list[str] = []

    def log(msg: str):
        print(msg)
        summary_log.append(msg)

    log(f"[e2e] connecting to GR00T server tcp://{args.host}:{args.port} ...")
    client = PolicyClient(host=args.host, port=args.port, timeout_ms=60000)
    log(f"[e2e] ping: {client.ping()}")

    modality = client.get_modality_config()
    video_deltas = list(modality["video"]["delta_indices"]) or DEFAULT_VIDEO_DELTA
    hist_span = max(-min(video_deltas), 0) + 1
    log(f"[e2e] video.delta_indices={video_deltas}  hist_span={hist_span}")

    log(f"[e2e] building env: task={args.task} robot={args.robot}")
    env = build_env(args.task, args.robot)
    log(f"[e2e] env.action_dim={env.action_dim}")
    obs = env.reset()

    def _resize_pair(env_obs):
        ext_full = _normalize_robosuite_image(env_obs[_ext_image_key(env_obs)])
        wrist_full = _normalize_robosuite_image(env_obs["robot0_eye_in_hand_image"])
        return {
            "ext": _resize_with_pad(ext_full, GR00T_RES_H, GR00T_RES_W),
            "wrist": _resize_with_pad(wrist_full, GR00T_RES_H, GR00T_RES_W),
        }

    frame_buf: deque = deque(maxlen=hist_span)
    initial = _resize_pair(obs)
    for _ in range(hist_span):
        frame_buf.append(initial)

    # Set up the video writer once we know canvas dims.
    canvas_seed = compose_canvas(
        cv2.resize(_normalize_robosuite_image(obs["frontview_image"]),
                   (SIM_VIEW_SIZE, SIM_VIEW_SIZE), interpolation=cv2.INTER_AREA),
        _normalize_robosuite_image(obs[_ext_image_key(obs)]),
        _normalize_robosuite_image(obs["robot0_eye_in_hand_image"]),
        prompt=args.prompt,
    )
    H, W = canvas_seed.shape[:2]
    video_path = out_dir / "attn_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.video_fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open at {video_path}")
    log(f"[e2e] writing video {W}x{H} @ {args.video_fps} fps → {video_path}")

    chunk: np.ndarray | None = None
    last_parsed: dict = {}
    last_dit: dict = {}
    saved_diagram = False
    n_infer = 0
    chunk_metrics: list[dict] = []

    try:
        for step in range(args.steps):
            frame_buf.append(_resize_pair(obs))

            if chunk is None or step % args.horizon == 0:
                request_obs = make_gr00t_obs(frame_buf, video_deltas, obs, args.prompt)
                action_dict, info = client.get_action(request_obs)
                jp = np.asarray(action_dict["joint_position"], dtype=np.float32)
                grip = np.asarray(action_dict["gripper_position"], dtype=np.float32)
                if jp.ndim == 3:
                    jp = jp[0]; grip = grip[0]
                chunk = np.concatenate([jp, grip], axis=1)
                n_infer += 1

                last_parsed = parse_attn_payload(info)
                if last_parsed.get("vlm"):
                    log(f"[e2e] chunk {n_infer}: VLM layers={len(last_parsed['vlm'])}, "
                        f"seq={last_parsed['vlm'][next(iter(last_parsed['vlm']))].shape[-1]}")
                else:
                    log(f"[e2e] chunk {n_infer}: NO VLM attention in payload")

                # DiT payload: extract per-layer averaged dict from info["attn"]["dit"]
                dit_raw = (info.get("attn") or {}).get("dit") or {}
                last_dit = {}
                for k, v in dit_raw.items():
                    try:
                        arr = np.asarray(v, dtype=np.float32)
                    except Exception:
                        continue
                    last_dit[int(k)] = arr
                if last_dit:
                    sample_shape = last_dit[next(iter(last_dit))].shape
                    log(f"[e2e] chunk {n_infer}: DiT layers={len(last_dit)}, "
                        f"sample_shape={sample_shape}")
                else:
                    log(f"[e2e] chunk {n_infer}: NO DiT attention in payload")

                # Save diagram from the FIRST chunk that has a usable payload.
                if not saved_diagram and last_parsed.get("vlm"):
                    summary = _summarize_attn_payload(last_parsed)
                    cur = _resize_pair(obs)
                    render_diagram(
                        summary,
                        cur["ext"],
                        cur["wrist"],
                        last_dit,
                        args.prompt,
                        out_dir / "attn_diagram.png",
                    )
                    saved_diagram = True
                    log(f"[e2e] wrote diagram → {out_dir/'attn_diagram.png'}")

                # Light per-chunk metric: mean text→image attention magnitude
                if last_parsed.get("vlm") and last_parsed.get("image_mask") is not None:
                    s = _summarize_attn_payload(last_parsed)
                    chunk_metrics.append({
                        "chunk": n_infer,
                        "ext_max": float(s.get("ext_attn", np.array([0])).max()),
                        "wrist_max": float(s.get("wrist_attn", np.array([0])).max()),
                    })

            model_action = chunk[step % args.horizon].astype(np.float32, copy=True)
            target_qpos = model_action[:7]
            gripper_abs = float(model_action[7])
            gripper_cmd = 1.0 if gripper_abs > args.gripper_thresh else -1.0

            env_action = np.zeros(env.action_dim, dtype=np.float32)
            env_action[0:7] = target_qpos
            if args.robot == "PandaOmron":
                if env.action_dim > 7:  env_action[7] = 0.0
                if env.action_dim > 10: env_action[8:11] = 0.0
                grip_idx = 11 if env.action_dim > 11 else (env.action_dim - 1)
                env_action[grip_idx] = gripper_cmd
                if env.action_dim > 12: env_action[12] = -1.0
            else:
                env_action[7] = gripper_cmd
            env_action[7:] = np.clip(env_action[7:], -1.0, 1.0)

            obs, _r, _d, _i = env.step(env_action)

            sim_view = _normalize_robosuite_image(obs["frontview_image"])
            sim_view = cv2.resize(sim_view, (SIM_VIEW_SIZE, SIM_VIEW_SIZE),
                                  interpolation=cv2.INTER_AREA)
            ext = _normalize_robosuite_image(obs[_ext_image_key(obs)])
            wrist = _normalize_robosuite_image(obs["robot0_eye_in_hand_image"])

            if last_parsed.get("vlm"):
                ext_tile, wrist_tile = render_attn_tiles(
                    last_parsed, ext, wrist, layer_idx=0, head_mode="mean"
                )
            else:
                ext_tile = wrist_tile = None

            canvas = compose_canvas(
                sim_view, ext, wrist,
                ext_attn=ext_tile, wrist_attn=wrist_tile,
                prompt=f"{args.prompt}  [step {step+1}/{args.steps}]",
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        try:
            env.close()
        except Exception:
            pass

    log(f"[e2e] sim done: {args.steps} steps, {n_infer} policy queries.")
    log(f"[e2e] video → {video_path}  (size={video_path.stat().st_size} bytes)")
    if saved_diagram:
        log(f"[e2e] diagram → {out_dir/'attn_diagram.png'}")
    else:
        log("[e2e] WARNING: no diagram written — VLM attention never appeared")

    log("\n[e2e] CHECKLIST")
    vlm_ok = saved_diagram
    log(f"  [{'PASS' if vlm_ok else 'FAIL'}] VLM attention captured")
    dit_ok = bool(last_dit)
    log(f"  [{'PASS' if dit_ok else 'FAIL'}] DiT attention captured")
    log(f"  [{'PASS' if video_path.exists() else 'FAIL'}] Video file exists")
    log(f"  [{'PASS' if (out_dir/'attn_diagram.png').exists() else 'FAIL'}] Diagram PNG exists")

    (out_dir / "summary.txt").write_text("\n".join(summary_log) + "\n")
    log(f"[e2e] summary → {out_dir/'summary.txt'}")

    if not (vlm_ok and dit_ok):
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
