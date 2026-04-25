"""Perception pipeline for RoboCasa LeRobot datasets.

Runs Gemini object detection + SAM2 segmentation on robocasa episode frames
and writes perception.h5 files in the same schema as the DROID pipeline.

Usage:
    python viz/robocasa_perception.py \\
        --lerobot-root third_party/robocasa/datasets/.../lerobot \\
        --episode 4 \\
        --output-dir /tmp/robocasa_perception
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow imports from viz/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robocasa_loader import (
    extract_frame,
    get_dataset_info,
    get_episode_info,
    load_episode_metadata,
    _CAMERA_KEYS,
    _video_path,
)
from perception_pipeline import (
    detect_objects,
    save_perception_h5,
    segment_objects,
)

OPEN_LOOP_HORIZON = 8

SAM2_CHECKPOINT = "./checkpoints/viz/sam2.1_hiera_large.pt"
GEMINI_MODEL = "gemini-2.5-flash"


def process_episode(
    lerobot_root: Path,
    episode_index: int,
    output_dir: Path,
    *,
    cameras: list[str] = ("wrist", "left"),
    use_sam2: bool = True,
    sam2_checkpoint: str = SAM2_CHECKPOINT,
    device: str = "cuda",
    gemini_model: str = GEMINI_MODEL,
) -> dict:
    """Run perception on all keyframes of one robocasa episode.

    Args:
        lerobot_root: Path to the LeRobot dataset root.
        episode_index: Episode to process.
        output_dir: Where to write perception.h5 files.
        cameras: Which cameras to process (subset of "wrist", "left", "right").
        use_sam2: Whether to run SAM2 segmentation after detection.
        sam2_checkpoint: Path to SAM2 model checkpoint.
        device: Torch device for SAM2.
        gemini_model: Gemini model ID for detection.

    Returns:
        Stats dict with counts of processed/skipped/failed frames.
    """
    lerobot_root = Path(lerobot_root)
    ep_info = get_episode_info(lerobot_root, episode_index)
    ds_info = get_dataset_info(lerobot_root)
    fps = ds_info.get("fps", 20)
    n_frames = ep_info["length"]
    instruction = ep_info.get("tasks", [""])[0]

    # Resolve video dimensions
    wrist_feat = ds_info["features"].get("observation.images.robot0_eye_in_hand", {})
    vid_shape = wrist_feat.get("shape", [256, 256, 3])
    height, width = vid_shape[0], vid_shape[1]

    ep_output = output_dir / f"episode_{episode_index:06d}"
    keyframes = list(range(0, n_frames, OPEN_LOOP_HORIZON))
    stats = {"ok": 0, "skip": 0, "error": 0}

    print(f"[ep {episode_index}] {n_frames} frames, {len(keyframes)} keyframes, "
          f"instruction: {instruction!r}")

    for frame_idx in keyframes:
        h5_path = ep_output / f"{frame_idx:05d}" / "perception.h5"
        if h5_path.exists():
            stats["skip"] += 1
            continue

        try:
            camera_results = {}

            for cam_name in cameras:
                cam_key = _CAMERA_KEYS[cam_name]
                video = _video_path(lerobot_root, episode_index, cam_key)
                img_arr = extract_frame(video, frame_idx, width=width, height=height, fps=fps)
                pil_img = Image.fromarray(img_arr)

                print(f"  [frame {frame_idx:05d}/{cam_name}] detecting...", end=" ", flush=True)
                bboxes = detect_objects(pil_img, instruction, model_id=gemini_model)
                print(f"{len(bboxes)} objects", end="", flush=True)

                if use_sam2 and bboxes:
                    print(", segmenting...", end=" ", flush=True)
                    masks = segment_objects(pil_img, bboxes, checkpoint=sam2_checkpoint, device=device)
                    print(f"{masks.shape[0]} masks")
                else:
                    H, W = pil_img.height, pil_img.width
                    masks = np.zeros((0, H, W), dtype=np.uint8)
                    print()

                camera_results[cam_name] = {
                    "image": img_arr,
                    "bboxes": bboxes,
                    "masks": masks,
                }

            h5_path.parent.mkdir(parents=True, exist_ok=True)
            save_perception_h5(h5_path, camera_results)
            stats["ok"] += 1

        except Exception as e:
            print(f"  [frame {frame_idx:05d}] ERROR: {e}")
            stats["error"] += 1

    # Write completion marker
    marker = ep_output / "perception.md"
    marker.write_text(
        f"episode: {episode_index}\n"
        f"instruction: {instruction}\n"
        f"frames: {stats}\n"
    )
    print(f"[ep {episode_index}] done: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="RoboCasa perception pipeline")
    parser.add_argument("--lerobot-root", type=Path, required=True, help="Path to LeRobot dataset root")
    parser.add_argument("--episode", type=int, required=True, help="Episode index to process")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--cameras", nargs="+", default=["wrist", "left"],
                        choices=["wrist", "left", "right"],
                        help="Cameras to process (default: wrist left)")
    parser.add_argument("--no-sam2", action="store_true", help="Skip SAM2 segmentation")
    parser.add_argument("--sam2-checkpoint", default=SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    args = parser.parse_args()

    process_episode(
        lerobot_root=args.lerobot_root,
        episode_index=args.episode,
        output_dir=args.output_dir,
        cameras=args.cameras,
        use_sam2=not args.no_sam2,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        gemini_model=args.gemini_model,
    )


if __name__ == "__main__":
    main()
