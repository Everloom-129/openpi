"""Batch attention pipeline for RoboCasa LeRobot datasets.

Runs Pi0/Pi0.5 inference on robocasa episodes and captures cross-attention
between text and images into HDF5 files, using the same format as the DROID
batch pipeline (viz/pipeline.py).

Usage:
    uv run python viz/robocasa_pipeline.py \\
        --lerobot-root third_party/robocasa/datasets/.../lerobot \\
        --output-dir /tmp/robocasa_attn \\
        --episodes 4
    uv run python viz/robocasa_pipeline.py \\
        --lerobot-root third_party/robocasa/datasets/.../lerobot \\
        --output-dir /tmp/robocasa_attn \\
        --episodes 4,5,6 \\
        --checkpoint ./checkpoints/viz/pi05_droid_pytorch \\
        --no-counterfactual
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attn_map import get_policy, select_best_gpu
from pipeline import OPEN_LOOP_HORIZON, infer_and_save, load_cf_config
from robocasa_loader import get_episode_info, load_episode_metadata, load_robocasa_example

DEFAULT_CHECKPOINT: str = "./checkpoints/viz/pi05_droid_pytorch"
DEFAULT_CF_CONFIG: str = str(Path(__file__).parent / "config" / "counterfactual.yaml")


def process_episode(
    policy,
    lerobot_root: Path,
    episode_index: int,
    output_dir: Path,
    cf_prompts: list[dict],
    ext_camera: str = "left",
    state_mode: str = "ee",
    force: bool = False,
) -> dict:
    """Process all keyframes of one robocasa episode.

    Returns stats dict: {total, ok, skipped, errors}.
    """
    ep_info = get_episode_info(lerobot_root, episode_index)
    n_frames = ep_info["length"]
    instruction = ep_info.get("tasks", [""])[0]

    ep_dir = output_dir / f"episode_{episode_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    marker = ep_dir / "robocasa.md"
    if marker.exists() and not force:
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0, "marker_skip": True}

    keyframes = list(range(0, n_frames, OPEN_LOOP_HORIZON))
    stats = {"total": len(keyframes), "ok": 0, "skipped": 0, "errors": 0}

    print(f"  {n_frames} frames, {len(keyframes)} keyframes")
    print(f"  instruction: {instruction!r}")

    for frame_idx in keyframes:
        frame_dir = ep_dir / f"{frame_idx:05d}"
        frame_dir.mkdir(exist_ok=True)

        try:
            example = load_robocasa_example(
                lerobot_root, episode_index, frame_idx,
                ext_camera=ext_camera, state_mode=state_mode,
            )

            # Main inference
            h5_main = frame_dir / f"{frame_idx:05d}.h5"
            if h5_main.exists():
                print(f"    {frame_idx:05d}.h5  (skip)")
                stats["skipped"] += 1
            else:
                infer_and_save(policy, example, h5_main, frame_idx)
                print(f"    {frame_idx:05d}.h5  done")

            # Counterfactual prompts
            for cf in cf_prompts:
                h5_cf = frame_dir / f"{frame_idx:05d}_{cf['key']}.h5"
                if h5_cf.exists():
                    continue
                cf_example = {**example, "prompt": cf["prompt"]}
                infer_and_save(policy, cf_example, h5_cf, frame_idx)
                print(f"    {frame_idx:05d}_{cf['key']}.h5  done  [{cf['method']}]")

            stats["ok"] += 1

        except Exception as e:
            import traceback
            print(f"    frame {frame_idx:05d}: error - {e}")
            traceback.print_exc()
            stats["errors"] += 1

    # Write completion marker
    marker.write_text(
        f"episode: {episode_index}\n"
        f"instruction: {instruction}\n"
        f"keyframes: {len(keyframes)}\n"
        f"ok: {stats['ok']}, skipped: {stats['skipped']}, errors: {stats['errors']}\n"
    )
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RoboCasa batch attention pipeline")
    parser.add_argument("--lerobot-root", type=Path, required=True,
                        help="Path to LeRobot dataset root (contains meta/, videos/, data/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for attention HDF5 files")
    parser.add_argument("--episodes", type=str, default="all",
                        help="Comma-separated episode indices, or 'all' (default: all)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=None,
                        help="Training config name (e.g. pi0_droid, pi05_droid). "
                             "If omitted, inferred from checkpoint dir name.")
    parser.add_argument("--ext-camera", default="left", choices=["left", "right"],
                        help="Exterior camera to use (default: left)")
    parser.add_argument("--state-mode", default="ee", choices=["ee", "zeros"],
                        help="State mapping: 'ee' (end-effector) or 'zeros' (default: ee)")
    parser.add_argument("--cf-config", default=DEFAULT_CF_CONFIG,
                        help="Path to counterfactual YAML config")
    parser.add_argument("--no-counterfactual", dest="counterfactual", action="store_false",
                        help="Skip counterfactual prompt inference")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess episodes even if robocasa.md marker exists")
    args = parser.parse_args(argv)

    lerobot_root = args.lerobot_root

    # Resolve episode list
    if args.episodes == "all":
        all_eps = load_episode_metadata(lerobot_root)
        episode_indices = [ep["episode_index"] for ep in all_eps]
    else:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",")]

    cf_prompts = load_cf_config(args.cf_config) if args.counterfactual else []
    if cf_prompts:
        print(f"Loaded {len(cf_prompts)} counterfactual prompts from {args.cf_config}")

    device = select_best_gpu()
    print(f"Loading policy from {args.checkpoint} on {device} ...")
    policy = get_policy(args.checkpoint, device=device, config_name=args.model)
    print("Policy loaded.\n")

    total = processed = skipped = 0
    for ep_idx in episode_indices:
        total += 1
        print(f"\n[{total}/{len(episode_indices)}] Episode {ep_idx}")
        t0 = time.perf_counter()

        stats = process_episode(
            policy=policy,
            lerobot_root=lerobot_root,
            episode_index=ep_idx,
            output_dir=args.output_dir,
            cf_prompts=cf_prompts,
            ext_camera=args.ext_camera,
            state_mode=args.state_mode,
            force=args.force,
        )

        if stats.get("marker_skip"):
            print(f"  [skip] already processed")
            skipped += 1
            continue

        elapsed = time.perf_counter() - t0
        print(f"  done {elapsed:.0f}s  ok={stats['ok']} skip={stats['skipped']} err={stats['errors']}")
        processed += 1

    print(f"\nFinished: {processed} processed, {skipped} skipped, {total} total")


if __name__ == "__main__":
    main()
