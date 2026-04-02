"""Batch perception pipeline — detect objects with Gemini and segment with SAM2.

Writes perception.h5 files directly into each episode directory (co-located
with trajectory.h5), since this is pure labeling data rather than inference results.

Output layout:

    DATA_ROOT/
    └── {success,failure}/
        └── {date}/
            └── {episode}/
                ├── perception.md            ← completion marker
                └── perception/
                    └── {frame:05d}/
                        └── perception.h5
                            ├── /wrist/
                            │   ├── image      uint8(H, W, 3)
                            │   ├── bboxes/
                            │   │   ├── labels  variable-length str(N,)
                            │   │   └── box_2d  int16(N, 4)  [ymin, xmin, ymax, xmax] 0–1000
                            │   └── masks       uint8(N, H, W)  SAM2 segmentation masks
                            └── /right/      (future: external camera)

Usage:
    uv run python viz/perception_pipeline.py <DATA_ROOT>
    uv run python viz/perception_pipeline.py <DATA_ROOT> --no-sam2
    uv run python viz/perception_pipeline.py <DATA_ROOT> --force
"""
from __future__ import annotations

import argparse
import json
import time
from functools import cache
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from attn_map import get_keyframes

# Reuse data helpers from the attention pipeline
from pipeline import OPEN_LOOP_HORIZON, get_video_length, load_example

# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA: str = "right"
SAM2_CHECKPOINT: str = "./checkpoints/viz/sam2.1_hiera_large.pt"
SAM2_CONFIG: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
GEMINI_MODEL: str = "gemini-2.0-flash"
_DETECT_PROMPT_PATH = Path(__file__).parent / "perception" / "prompts" / "detect.txt"


# ── Gemini detection ───────────────────────────────────────────────────────────

@cache
def _load_detect_prompt() -> str:
    return _DETECT_PROMPT_PATH.read_text().strip()


@cache
def _gemini_client():
    from google import genai
    return genai.Client()


def detect_objects(
    image: Image.Image,
    instruction: str,
    model_id: str = GEMINI_MODEL,
) -> list[dict]:
    """Detect objects in an image using Gemini.

    Returns a list of dicts: {"label": str, "box_2d": [ymin, xmin, ymax, xmax]} (0–1000 scale).
    Returns [] on parse failure.
    """
    from google.genai import types

    prompt = _load_detect_prompt().format(task_instruction=instruction)
    response = _gemini_client().models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=None,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    text = response.text.strip()
    # Strip optional markdown code fencing
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"    [warn] Gemini returned non-JSON: {text[:200]}")
        return []

    return [b for b in data.get("bboxes", []) if len(b.get("box_2d", [])) == 4]


# ── SAM2 segmentation ──────────────────────────────────────────────────────────

@cache
def _sam2_predictor(checkpoint: str, device: str):
    """Load and cache the SAM2 image predictor (one instance per checkpoint+device)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"Loading SAM2 from {checkpoint} on {device} ...")
    model = build_sam2(SAM2_CONFIG, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    print("SAM2 loaded.")
    return predictor


def segment_objects(
    image: Image.Image,
    bboxes: list[dict],
    checkpoint: str = SAM2_CHECKPOINT,
    device: str = "cuda",
) -> np.ndarray:
    """Segment objects using SAM2, prompted with Gemini bounding boxes.

    Args:
        image: PIL Image to segment.
        bboxes: List of dicts with 'box_2d' key ([ymin, xmin, ymax, xmax], 0–1000 scale).
        checkpoint: Path to SAM2 checkpoint file.
        device: Torch device string.

    Returns:
        masks: uint8 array of shape (N, H, W), values 0 or 1.
               Returns shape (0, H, W) if bboxes is empty.
    """
    import torch

    H, W = image.height, image.width

    if not bboxes:
        return np.zeros((0, H, W), dtype=np.uint8)

    # Convert Gemini [ymin, xmin, ymax, xmax] (0–1000) → SAM2 [x0, y0, x1, y1] (pixels)
    boxes = np.array(
        [
            [
                (b["box_2d"][1] / 1000.0) * W,  # xmin → x0
                (b["box_2d"][0] / 1000.0) * H,  # ymin → y0
                (b["box_2d"][3] / 1000.0) * W,  # xmax → x1
                (b["box_2d"][2] / 1000.0) * H,  # ymax → y1
            ]
            for b in bboxes
        ],
        dtype=np.float32,
    )

    predictor = _sam2_predictor(checkpoint, device)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

    # masks shape: (N, 1, H, W) or (N, H, W) → normalize to (N, H, W) uint8
    if masks.ndim == 4:
        masks = masks[:, 0]
    return masks.astype(np.uint8)


# ── HDF5 writer ────────────────────────────────────────────────────────────────

def save_perception_h5(
    h5_path: Path,
    camera_results: dict[str, dict],
) -> None:
    """Write perception results for one frame to HDF5.

    Args:
        h5_path: Output file path.
        camera_results: Dict mapping camera name (e.g. "wrist", "right") to a dict with:
            - "image":  np.ndarray uint8(H, W, 3)
            - "bboxes": list[dict] with keys "label" and "box_2d"
            - "masks":  np.ndarray uint8(N, H, W)

    HDF5 schema per camera group:
        /{camera}/image       uint8(H, W, 3)           gzip-4
        /{camera}/bboxes/
            labels            variable-length str(N,)
            box_2d            int16(N, 4)
        /{camera}/masks       uint8(N, H, W)            gzip-4
    """
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as f:
        for camera, data in camera_results.items():
            grp = f.create_group(camera)
            image: np.ndarray = data["image"]
            bboxes: list[dict] = data["bboxes"]
            masks: np.ndarray = data["masks"]

            grp.create_dataset("image", data=image, compression="gzip", compression_opts=4)

            bbox_grp = grp.create_group("bboxes")
            if bboxes:
                labels = [b["label"] for b in bboxes]
                box_2d = np.array([b["box_2d"] for b in bboxes], dtype=np.int16)
                bbox_grp.create_dataset("labels", data=np.array(labels, dtype=object), dtype=str_dt)
                bbox_grp.create_dataset("box_2d", data=box_2d)
            else:
                bbox_grp.create_dataset("labels", data=np.array([], dtype=object), dtype=str_dt)
                bbox_grp.create_dataset("box_2d", data=np.zeros((0, 4), dtype=np.int16))

            H, W = image.shape[:2]
            if masks.shape[0] > 0:
                grp.create_dataset("masks", data=masks, compression="gzip", compression_opts=4)
            else:
                grp.create_dataset("masks", data=np.zeros((0, H, W), dtype=np.uint8))


# ── Per-frame processing ───────────────────────────────────────────────────────

def process_frame(
    data_dir: Path,
    frame_idx: int,
    episode_perception_dir: Path,
    *,
    camera: str,
    use_sam2: bool,
    sam2_checkpoint: str,
    device: str,
    gemini_model: str,
) -> str:
    """Detect and segment one keyframe.

    Returns: "ok", "skip", or "error".
    """
    h5_path = episode_perception_dir / f"{frame_idx:05d}" / "perception.h5"

    if h5_path.exists():
        return "skip"

    example = load_example(data_dir, frame_idx, camera=camera)
    instruction = example["prompt"]

    # ── Wrist camera ───────────────────────────────────────────────────────────
    wrist_img = Image.fromarray(example["observation/wrist_image_left"])
    wrist_bboxes = detect_objects(wrist_img, instruction, model_id=gemini_model)

    if use_sam2:
        wrist_masks = segment_objects(wrist_img, wrist_bboxes, checkpoint=sam2_checkpoint, device=device)
    else:
        H, W = wrist_img.height, wrist_img.width
        wrist_masks = np.zeros((0, H, W), dtype=np.uint8)

    camera_results = {
        "wrist": {
            "image": np.array(wrist_img),
            "bboxes": wrist_bboxes,
            "masks": wrist_masks,
        }
    }

    # ── Future: external camera ────────────────────────────────────────────────
    # ext_img = Image.fromarray(example["observation/exterior_image_1_left"])
    # ext_bboxes = detect_objects(ext_img, instruction, model_id=gemini_model)
    # ext_masks = segment_objects(ext_img, ext_bboxes, ...) if use_sam2 else ...
    # camera_results["right"] = {"image": np.array(ext_img), "bboxes": ext_bboxes, "masks": ext_masks}

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    save_perception_h5(h5_path, camera_results)
    return "ok"


# ── Episode processing ─────────────────────────────────────────────────────────

def process_episode(
    data_dir: Path,
    **frame_kwargs,
) -> dict:
    """Process all keyframes of one episode.

    Returns stats dict: {total, ok, skipped, errors}.
    """
    total_frames = get_video_length(data_dir)
    if total_frames == 0:
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0}

    keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)
    episode_perception_dir = data_dir / "perception"
    stats = {"total": len(keyframes), "ok": 0, "skipped": 0, "errors": 0}

    for frame_idx in keyframes:
        try:
            result = process_frame(
                data_dir=data_dir,
                frame_idx=frame_idx,
                episode_perception_dir=episode_perception_dir,
                **frame_kwargs,
            )
            if result == "skip":
                print(f"    {frame_idx:05d}/perception.h5  (skip)")
                stats["skipped"] += 1
            else:
                n_obj = 0  # reported inside process_frame via bboxes length
                print(f"    {frame_idx:05d}/perception.h5  ✓")
                stats["ok"] += 1

        except FileNotFoundError as e:
            print(f"    frame {frame_idx:05d}: file not found — {e}")
            stats["errors"] += 1
        except Exception as e:
            import traceback
            print(f"    frame {frame_idx:05d}: error — {e}")
            traceback.print_exc()
            stats["errors"] += 1

    return stats


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Perception labeling pipeline (Gemini + SAM2)")
    parser.add_argument("data_root", help="Root dir with success/ and failure/ subdirs")
    parser.add_argument(
        "--camera", default=CAMERA, choices=["right", "left"],
        help="Which external camera to use in load_example (wrist is always included)",
    )
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    parser.add_argument("--sam2-checkpoint", default=SAM2_CHECKPOINT)
    parser.add_argument(
        "--no-sam2", dest="use_sam2", action="store_false",
        help="Skip SAM2 segmentation and save bboxes only",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess episodes even if perception.md marker exists",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    DATA_ROOT = Path(args.data_root)

    if args.use_sam2:
        import torch
        device = args.device if torch.cuda.is_available() else "cpu"
        print(f"SAM2 enabled — checkpoint={args.sam2_checkpoint}, device={device}")
    else:
        device = "cpu"
        print("SAM2 disabled — saving bboxes only")

    frame_kwargs = dict(
        camera=args.camera,
        use_sam2=args.use_sam2,
        sam2_checkpoint=args.sam2_checkpoint,
        device=device,
        gemini_model=args.gemini_model,
    )

    total_episodes = processed = skipped = errors = 0

    for outcome in ("success", "failure"):
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            continue

        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for traj_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = traj_path.parent
                episode_id = data_dir.name
                total_episodes += 1

                marker = data_dir / "perception.md"
                if marker.exists() and not args.force:
                    print(f"[skip] {outcome}/{date_dir.name}/{episode_id}")
                    skipped += 1
                    continue

                print(f"\n[{total_episodes}] {outcome}/{date_dir.name}/{episode_id}")
                t0 = time.perf_counter()

                stats = process_episode(data_dir=data_dir, **frame_kwargs)

                elapsed = time.perf_counter() - t0
                print(
                    f"  done {elapsed:.0f}s  "
                    f"ok={stats['ok']} skip={stats['skipped']} err={stats['errors']}"
                )

                if stats["errors"] > 0 and stats["ok"] == 0:
                    print("  all frames failed, skipping marker")
                    errors += 1
                else:
                    total_frames = get_video_length(data_dir)
                    marker.write_text(
                        f"# Perception Complete\n\n"
                        f"Episode: {episode_id}\n"
                        f"Outcome: {outcome}\n"
                        f"Date: {date_dir.name}\n"
                        f"Total Frames: {total_frames}\n"
                        f"Keyframes: {get_keyframes(total_frames, OPEN_LOOP_HORIZON)}\n"
                        f"SAM2: {args.use_sam2}\n"
                        f"GeminiModel: {args.gemini_model}\n"
                    )
                    processed += 1

    print(f"\n{'=' * 50}")
    print(f"Total episodes:         {total_episodes}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (done):         {skipped}")
    print(f"Errors:                 {errors}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
