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

Issue:
1. need to add external cam
2. object label is too general and unique now (1k+ labels fro 11 episodes)
3. object label is inconsitent within an episode
4. overlap rule?


Usage:
    uv run python viz/perception_pipeline.py <DATA_ROOT>
    uv run python viz/perception_pipeline.py <DATA_ROOT> --no-sam2
    uv run python viz/perception_pipeline.py <DATA_ROOT> --force
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

from attn_map import get_keyframes

# Reuse data helpers from the attention pipeline
from pipeline import OPEN_LOOP_HORIZON, get_video_length, load_example

# ── Configuration ──────────────────────────────────────────────────────────────
CAMERA: str = "right"
SAM2_CHECKPOINT: str = "./checkpoints/viz/sam2.1_hiera_large.pt"
SAM2_CONFIG: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
GEMINI_MODEL: str = "gemini-2.5-flash"
_DETECT_PROMPT_PATH = Path(__file__).parent / "perception" / "prompts" / "detect.txt"


# ── Gemini detection ───────────────────────────────────────────────────────────

@cache
def _load_detect_prompt() -> str:
    return _DETECT_PROMPT_PATH.read_text().strip()


@cache
def _gemini_client():
    from google import genai
    return genai.Client()


_LABEL_ALIASES = ("label", "name", "object", "description", "class")
_RETRY_PROMPT_SUFFIX = "\n\nCRITICAL: Return ONLY the raw JSON object. No markdown, no code fences, no extra text. The response must start with {{ and end with }}."
MAX_DETECT_RETRIES = 3


def _parse_detect_response(text: str) -> list[dict] | None:
    """Parse and normalise a Gemini detection response.

    Returns a list of valid bbox dicts on success, or None if parsing failed.
    """
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    normalized = []
    for b in data.get("bboxes", []):
        if len(b.get("box_2d", [])) != 4:
            continue
        if "label" not in b:
            for alias in _LABEL_ALIASES[1:]:
                if alias in b:
                    b["label"] = b.pop(alias)
                    break
            else:
                print(f"    [warn] bbox missing label key (keys={list(b.keys())}), skipping")
                continue
        normalized.append(b)
    return normalized


def detect_objects(
    image: Image.Image,
    instruction: str,
    model_id: str = GEMINI_MODEL,
) -> list[dict]:
    """Detect objects in an image using Gemini, with up to MAX_DETECT_RETRIES retries.

    Returns a list of dicts: {"label": str, "box_2d": [ymin, xmin, ymax, xmax]} (0–1000 scale).
    Returns [] if all attempts fail.
    """
    from google.genai import types

    base_prompt = _load_detect_prompt().format(task_instruction=instruction)

    for attempt in range(1, MAX_DETECT_RETRIES + 1):
        prompt = base_prompt if attempt == 1 else base_prompt + _RETRY_PROMPT_SUFFIX
        try:
            response = _gemini_client().models.generate_content(
                model=model_id,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=None,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            result = _parse_detect_response(response.text)
        except Exception as e:
            print(f"    [warn] Gemini API error (attempt {attempt}/{MAX_DETECT_RETRIES}): {e}")
            result = None

        if result is not None:
            if attempt > 1:
                print(f"    [retry ok] succeeded on attempt {attempt}")
            return result

        print(f"    [warn] Gemini non-JSON (attempt {attempt}/{MAX_DETECT_RETRIES}): {response.text[:120]!r}")

    print(f"    [error] detect_objects failed after {MAX_DETECT_RETRIES} attempts, returning []")
    return []


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


# ── Label statistics ───────────────────────────────────────────────────────────

def write_episode_label_stats(
    episode_perception_dir: Path,
    episode_id: str,
    instruction: str,
    outcome: str,
) -> dict[str, int]:
    """Scan all perception.h5 files in an episode and write label_stats.yaml.

    Reads whatever Gemini detected (new or pre-existing frames) so the stats are
    always complete even when frames were skipped.

    Returns: {label: frames_detected} for dataset-level aggregation.
    """
    label_frame_counts: Counter = Counter()
    total_keyframes = 0

    for frame_dir in sorted(episode_perception_dir.iterdir()):
        h5_path = frame_dir / "perception.h5"
        if not h5_path.exists():
            continue
        total_keyframes += 1
        try:
            with h5py.File(h5_path, "r") as f:
                if "wrist/bboxes/labels" in f:
                    for raw in f["wrist/bboxes/labels"][:]:
                        label = raw.decode() if isinstance(raw, bytes) else str(raw)
                        label_frame_counts[label] += 1
        except Exception as e:
            print(f"    [label stats warn] {h5_path.name}: {e}")

    if not total_keyframes:
        return {}

    label_stats = {
        label: {
            "frames_detected": count,
            "detection_rate": round(count / total_keyframes, 4),
        }
        for label, count in sorted(label_frame_counts.items())
    }

    yaml_path = episode_perception_dir / "label_stats.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "episode": episode_id,
                "instruction": instruction,
                "outcome": outcome,
                "total_keyframes": total_keyframes,
                "label_stats": label_stats,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            f, default_flow_style=False, sort_keys=False, allow_unicode=True,
        )
    print(f"  Label stats → {yaml_path.relative_to(episode_perception_dir.parent.parent)}")
    return dict(label_frame_counts)


def write_dataset_label_stats(
    data_root: Path,
    all_episode_counts: list[dict[str, int]],
    total_episodes_processed: int,
) -> None:
    """Aggregate per-episode label counts into a dataset-level label_stats.yaml.

    For each label records: total detections, episodes_detected, episode_rate,
    and per-episode detection stats (mean / min / max / std).
    """
    # Collect per-label list of frame-detection counts (one entry per episode where seen)
    label_ep_counts: dict[str, list[int]] = defaultdict(list)
    for ep_counts in all_episode_counts:
        for label, count in ep_counts.items():
            label_ep_counts[label].append(count)

    label_stats = {}
    for label in sorted(label_ep_counts.keys()):
        arr = np.array(label_ep_counts[label], dtype=float)
        label_stats[label] = {
            "total_detections": int(arr.sum()),
            "episodes_detected": int(len(arr)),
            "episode_rate": round(float(len(arr) / total_episodes_processed), 4) if total_episodes_processed else 0.0,
            "detections_per_episode": {
                "mean": round(float(arr.mean()), 2),
                "min":  int(arr.min()),
                "max":  int(arr.max()),
                "std":  round(float(arr.std()), 2),
            },
        }

    yaml_path = data_root / "label_stats.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "total_episodes_processed": total_episodes_processed,
                "unique_labels": len(label_stats),
                "label_stats": label_stats,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            f, default_flow_style=False, sort_keys=False, allow_unicode=True,
        )
    print(f"Dataset label stats → {yaml_path}")


# ── Episode processing ─────────────────────────────────────────────────────────

def process_episode(
    data_dir: Path,
    outcome: str = "",
    **frame_kwargs,
) -> tuple[dict, dict[str, int]]:
    """Process all keyframes of one episode.

    Returns: (stats dict, label_frame_counts dict)
      - stats: {total, ok, skipped, errors}
      - label_frame_counts: {label: frames_detected} across this episode
    """
    total_frames = get_video_length(data_dir)
    if total_frames == 0:
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0}, {}

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

    # Write episode-level label stats (reads all h5 files, including pre-existing ones)
    episode_perception_dir.mkdir(exist_ok=True)
    instruction_path = data_dir / "instruction.txt"
    instruction = instruction_path.read_text().strip() if instruction_path.exists() else ""
    label_counts = write_episode_label_stats(
        episode_perception_dir, data_dir.name, instruction, outcome
    )

    return stats, label_counts


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
    all_episode_label_counts: list[dict[str, int]] = []  # for dataset-level stats

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
                    # Still collect label stats from existing perception.h5 files
                    ep_perc_dir = data_dir / "perception"
                    if ep_perc_dir.exists():
                        instruction_path = data_dir / "instruction.txt"
                        instruction = instruction_path.read_text().strip() if instruction_path.exists() else ""
                        label_counts = write_episode_label_stats(ep_perc_dir, episode_id, instruction, outcome)
                        if label_counts:
                            all_episode_label_counts.append(label_counts)
                    skipped += 1
                    continue

                print(f"\n[{total_episodes}] {outcome}/{date_dir.name}/{episode_id}")
                t0 = time.perf_counter()

                stats, label_counts = process_episode(
                    data_dir=data_dir, outcome=outcome, **frame_kwargs
                )
                if label_counts:
                    all_episode_label_counts.append(label_counts)

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

    # Dataset-level label stats (written even if some episodes were skipped)
    total_counted = skipped + processed
    if all_episode_label_counts:
        write_dataset_label_stats(DATA_ROOT, all_episode_label_counts, total_counted)

    print(f"\n{'=' * 50}")
    print(f"Total episodes:         {total_episodes}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (done):         {skipped}")
    print(f"Errors:                 {errors}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
