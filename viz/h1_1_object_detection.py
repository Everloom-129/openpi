"""
Hypothesis 1.1: Object Detection Correlation

Evaluates the correlation between VLM attention maps and object detection masks.
For each layer, computes IoU/overlap metrics between attention heatmaps and
ground truth object masks from DINO-X.
"""

import json
import os
from pathlib import Path

import cv2
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.shared import image_tools
from openpi.training import config as _config

from attn_map import select_best_gpu


def load_pineapple_example(camera: str = "left", index: int = 0):
    """Load example from pineapple test episode with object detection masks"""

    if camera == "left":
        camera = "varied_camera_1"
    elif camera == "right":
        camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    with open("data/visualization/aawr_pineapple/white_list.json") as f:
        white_list = json.load(f)

    if index not in white_list:
        raise ValueError(f"index {index} must be in the white list")

    data_dir = "data/visualization/aawr_pineapple/recordings/frames"

    # Load images
    ext_path = os.path.join(data_dir, camera, f"{index:05d}.jpg")
    hand_path = os.path.join(data_dir, "hand_camera", f"{index:05d}.jpg")

    if not os.path.exists(ext_path):
        raise FileNotFoundError(f"Exterior image not found: {ext_path}")
    if not os.path.exists(hand_path):
        raise FileNotFoundError(f"Hand image not found: {hand_path}")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    instruction_path = os.path.join(data_dir, "instruction.txt")
    if os.path.exists(instruction_path):
        with open(instruction_path) as f:
            instruction = f.read().strip()
    else:
        print(f"Warning: Instruction file not found at {instruction_path}, using default.")
        instruction = "find the pineapple toy and pick it up"

    print(f"Instruction: {instruction}")

    # Load joint and gripper positions from trajectory h5 file
    traj_path = os.path.join("data/visualization/aawr_pineapple/", "trajectory.h5")
    if os.path.exists(traj_path):
        with h5py.File(traj_path, "r") as f:
            joint_positions = np.array(f["observation/robot_state/joint_positions"][index])
            gripper_position = np.array(f["observation/robot_state/gripper_position"][index])
    else:
        print(f"Warning: Trajectory file not found at {traj_path}, using random data.")
        joint_positions = np.random.rand(7)
        gripper_position = np.random.rand()

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": joint_positions,
        "observation/gripper_position": gripper_position,
        "prompt": instruction,
    }


def load_object_masks(data_dir: str, index: int, camera: str = "left") -> dict[str, np.ndarray]:
    """Load object detection masks for a specific frame"""
    dinox_info_path = os.path.join(data_dir, "dinox_info.json")
    masks_dir = os.path.join(data_dir, "masks")

    with open(dinox_info_path) as f:
        dinox_info = json.load(f)

    index_str = str(index)
    if index_str not in dinox_info:
        return {}

    detections = dinox_info[index_str]
    masks = {}

    for i, det in enumerate(detections):
        if det["mask_file"] and det["score"] > 0:
            mask_path = os.path.join(masks_dir, det["mask_file"])
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                masks[f"object_{i}"] = {
                    "mask": mask,
                    "bbox": det["bbox"],
                    "category": det["category"],
                    "score": det["score"],
                }

    return masks


def resize_mask_to_224(mask_raw: np.ndarray) -> np.ndarray:
    """Resize raw image mask to 224x224 with padding (matching openpi's preprocessing)"""
    raw_h, raw_w = mask_raw.shape[:2]
    target_h, target_w = 224, 224

    # Calculate scaling to match resize_with_pad behavior
    scale = min(target_w / raw_w, target_h / raw_h)
    new_w = int(raw_w * scale)
    new_h = int(raw_h * scale)

    # Resize mask
    mask_resized = cv2.resize(mask_raw.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Add padding
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    mask_224 = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_224[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = mask_resized

    return mask_224 > 0


def compute_attention_on_object(attn_map_16x16: np.ndarray, object_mask_224: np.ndarray) -> dict[str, float]:
    """
    Compute overlap metrics between attention map and object mask.

    Args:
        attn_map_16x16: Attention map in 16x16 resolution
        object_mask_224: Binary object mask in 224x224 resolution

    Returns:
        Dictionary with metrics: overlap_ratio, attention_concentration, iou
    """
    # Resize attention to 224x224
    attn_224 = cv2.resize(attn_map_16x16, (224, 224), interpolation=cv2.INTER_CUBIC)
    attn_224 = np.maximum(attn_224, 0)  # Ensure non-negative

    # Normalize attention to [0, 1]
    attn_224_norm = attn_224 / attn_224.max() if attn_224.max() > 0 else attn_224

    # Compute metrics
    object_mask_bool = object_mask_224.astype(bool)

    # 1. Attention mass on object (sum of attention weights inside mask)
    attn_on_object = attn_224_norm[object_mask_bool].sum()
    total_attn = attn_224_norm.sum()
    overlap_ratio = attn_on_object / (total_attn + 1e-8)

    # 2. Mean attention inside vs outside object
    mean_attn_inside = attn_224_norm[object_mask_bool].mean() if object_mask_bool.sum() > 0 else 0
    mean_attn_outside = attn_224_norm[~object_mask_bool].mean() if (~object_mask_bool).sum() > 0 else 0
    attention_concentration = mean_attn_inside / (mean_attn_outside + 1e-8)

    # 3. IoU-like metric (thresholded attention vs mask)
    # Threshold attention at median or 0.3
    attn_thresh = attn_224_norm > 0.3
    intersection = (attn_thresh & object_mask_bool).sum()
    union = (attn_thresh | object_mask_bool).sum()
    iou = intersection / (union + 1e-8)

    return {
        "overlap_ratio": float(overlap_ratio),
        "attention_concentration": float(attention_concentration),
        "iou": float(iou),
        "mean_attn_inside": float(mean_attn_inside),
        "mean_attn_outside": float(mean_attn_outside),
    }


def visualize_attention_on_mask(
    attn_map: np.ndarray, object_masks: dict, raw_image: np.ndarray, output_path: str, layer_idx: int
):
    """Visualize attention overlaid with object masks"""
    # Resize attention to 224x224
    attn_224 = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Resize image to 224x224 with padding
    img_jax = jnp.array(raw_image)
    img_224 = np.array(image_tools.resize_with_pad(img_jax, 224, 224)).astype(np.uint8)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image (Wrist Camera)
    axes[0].imshow(img_224)
    axes[0].set_title("Wrist Camera (Original)")
    axes[0].axis("off")

    # Attention heatmap
    im = axes[1].imshow(attn_224, cmap="jet", interpolation="bilinear")
    axes[1].set_title(f"Wrist Attention (Layer {layer_idx})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay with mask
    overlay = img_224.copy()
    for obj_data in object_masks.values():
        mask_224 = resize_mask_to_224(obj_data["mask"])
        # Draw mask boundary in green
        contours, _ = cv2.findContours(mask_224.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Overlay attention heatmap
    attn_norm = np.uint8(255 * attn_224 / (np.max(attn_224) + 1e-8))
    attn_color = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
    attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.6, attn_color, 0.4, 0)

    axes[2].imshow(overlay)
    axes[2].set_title("Wrist Attention + Object Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_object_detection(
    policy,
    example,
    frame_idx: int,
    data_dir: str,
    episode_dir: Path,
    layers: list[int] | None = None,
    camera: str = "left",
):
    """
    Runs the object detection correlation analysis for specified layers on a single frame.
    Returns:
        Dictionary with results for each layer
    """
    if layers is None:
        layers = [1, 4, 5, 7, 10, 17]

    results = {}

    # Load object masks for this frame
    object_masks = load_object_masks(data_dir, frame_idx, camera=camera)

    if not object_masks:
        print(f"  Warning: No object masks found for frame {frame_idx}. Skipping.")
        return results

    print(f"  Found {len(object_masks)} objects in frame {frame_idx}")

    # Create frame-specific directory: episode_dir/object/{frame_idx:05d}/
    frame_dir = episode_dir / "object" / f"{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    attn_mode = "prefix"
    device_id = str(select_best_gpu())
    attn_base = Path(f"attn/{device_id}/layers_{attn_mode}")
    if not str(attn_base).endswith(f"layers_{attn_mode}"):
        attn_base = attn_base / f"layers_{attn_mode}"

    for layer_idx in layers:
        print(f"  Processing Layer {layer_idx}...")

        attn_path = attn_base / f"attn_map_layer_{layer_idx}.npy"

        if not attn_path.exists():
            print(f"    Warning: Attention map not found for Layer {layer_idx} at {attn_path}. Skipping.")
            continue

        attn_map = np.load(attn_path)  # [Batch, Heads, Seq, Seq]

        # Process attention: average across heads, then extract text->image attention
        if attn_map.ndim == 4:
            attn_map = attn_map[0]  # Remove batch dimension: [Heads, Seq, Seq]

        # Extract Image Attention (Text -> Image)
        # Tokens 0-255: Ext Image, 256-511: Wrist Image, 512+: Text
        # NOTE: Object masks are for WRIST camera, so we use tokens 256-511
        num_img = 256
        total_img = 512
        wrist_start = num_img  # 256
        wrist_end = total_img  # 512

        layer_results = {}

        # Average attention across heads for layer-level analysis
        attn_avg = attn_map.mean(axis=0) if attn_map.ndim == 3 else attn_map

        if attn_avg.shape[0] <= total_img:
            # Skip if this is not prefix attention
            continue

        # Extract text->WRIST attention (max over text tokens)
        text_attn_avg = attn_avg[total_img:, :total_img].max(axis=0)
        attn_wrist = text_attn_avg[wrist_start:wrist_end].reshape(16, 16)

        # Visualize - save as L{layer}.jpg in frame directory
        vis_path = frame_dir / f"L{layer_idx}.jpg"
        visualize_attention_on_mask(
            attn_wrist, object_masks, example["observation/wrist_image_left"], str(vis_path), layer_idx
        )

        # Compute aggregate metrics across all objects
        aggregate_metrics = {}
        for obj_name, obj_data in object_masks.items():
            mask_224 = resize_mask_to_224(obj_data["mask"])
            metrics = compute_attention_on_object(attn_wrist, mask_224)
            metrics["object_score"] = obj_data["score"]
            metrics["object_category"] = obj_data["category"]
            aggregate_metrics[obj_name] = metrics

        layer_results["aggregate_metrics"] = aggregate_metrics

        results[f"layer_{layer_idx}"] = layer_results

    return results


def create_layer_video(
    episode_dir: Path, layer: int, fps: int = 5, output_path: str | None = None, *, add_text: bool = True
):
    """
    Create video showing one layer's attention across all frames.

    Args:
        episode_dir: Base directory containing object/frame folders
        layer: Layer index to create video for
        fps: Frames per second for output video
        output_path: Custom output path, or None to auto-generate
        add_text: Whether to add frame number text overlay

    Returns:
        Path to created video file, or None if no frames found
    """
    object_dir = episode_dir / "object"
    if not object_dir.exists():
        print(f"Object directory not found: {object_dir}")
        return None

    # Collect all frame directories
    frame_dirs = sorted([d for d in object_dir.iterdir() if d.is_dir()])
    if not frame_dirs:
        print(f"No frame directories found in {object_dir}")
        return None

    # Collect frames for this layer
    frames = []
    frame_indices = []
    for frame_dir in frame_dirs:
        layer_img_path = frame_dir / f"L{layer}.jpg"
        if layer_img_path.exists():
            img = cv2.imread(str(layer_img_path))
            if img is not None:
                if add_text:
                    # Add frame number overlay
                    frame_idx = int(frame_dir.name)
                    cv2.putText(
                        img,
                        f"Frame: {frame_idx}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        f"Layer: {layer}",
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )
                frames.append(img)
                frame_indices.append(frame_dir.name)

    if not frames:
        print(f"No frames found for Layer {layer}")
        return None

    # Generate output path if not provided
    if output_path is None:
        video_dir = episode_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        output_path = video_dir / f"L{layer:02d}_object_attn.webm"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video with VP90 codec for webm
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Created video for Layer {layer}: {output_path} ({len(frames)} frames)")
    return str(output_path)


def create_all_layer_videos(episode_dir: Path, layers: list[int] | None = None, fps: int = 5, *, add_text: bool = True):
    """
    Create videos for all specified layers.

    Args:
        episode_dir: Base directory containing object/frame folders
        layers: List of layer indices, or None to auto-detect
        fps: Frames per second for output videos
        add_text: Whether to add frame number text overlay

    Returns:
        Dictionary mapping layer index to video path
    """
    # Auto-detect layers if not specified
    if layers is None:
        object_dir = episode_dir / "object"
        frame_dirs = sorted([d for d in object_dir.iterdir() if d.is_dir()])
        if not frame_dirs:
            print("No frames found, cannot auto-detect layers")
            return {}

        # Check first frame for available layers
        first_frame = frame_dirs[0]
        layer_files = sorted(first_frame.glob("L*.jpg"))
        layers = [int(f.stem[1:]) for f in layer_files]
        print(f"Auto-detected {len(layers)} layers: {layers}")

    # Create video for each layer
    video_paths = {}
    for layer in layers:
        video_path = create_layer_video(episode_dir, layer, fps=fps, add_text=add_text)
        if video_path:
            video_paths[layer] = video_path

    return video_paths


def generate_summary_plot(all_results: dict, output_path: str, layers: list[int]):
    """Generate summary plot showing attention correlation with objects across layers"""

    # Aggregate metrics across all frames and objects
    layer_metrics = {layer: [] for layer in layers}

    for frame_results in all_results.values():
        for layer_key, layer_data in frame_results.items():
            layer_idx = int(layer_key.split("_")[1])
            if layer_idx not in layers:
                continue

            if "aggregate_metrics" in layer_data:
                for metrics in layer_data["aggregate_metrics"].values():
                    layer_metrics[layer_idx].append(
                        {
                            "overlap_ratio": metrics["overlap_ratio"],
                            "attention_concentration": metrics["attention_concentration"],
                            "iou": metrics["iou"],
                        }
                    )

    # Compute mean and std for each layer
    layer_stats = {}
    for layer_idx in layers:
        if not layer_metrics[layer_idx]:
            continue

        overlap_ratios = [m["overlap_ratio"] for m in layer_metrics[layer_idx]]
        concentrations = [m["attention_concentration"] for m in layer_metrics[layer_idx]]
        ious = [m["iou"] for m in layer_metrics[layer_idx]]

        layer_stats[layer_idx] = {
            "overlap_ratio_mean": np.mean(overlap_ratios),
            "overlap_ratio_std": np.std(overlap_ratios),
            "concentration_mean": np.mean(concentrations),
            "concentration_std": np.std(concentrations),
            "iou_mean": np.mean(ious),
            "iou_std": np.std(ious),
            "n_samples": len(overlap_ratios),
        }

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sorted_layers = sorted(layer_stats.keys())

    # Plot 1: Overlap Ratio
    means = [layer_stats[layer]["overlap_ratio_mean"] for layer in sorted_layers]
    stds = [layer_stats[layer]["overlap_ratio_std"] for layer in sorted_layers]
    axes[0].bar(range(len(sorted_layers)), means, yerr=stds, capsize=5, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Layer", fontsize=12)
    axes[0].set_ylabel("Overlap Ratio", fontsize=12)
    axes[0].set_title("Attention Mass on Object", fontsize=14)
    axes[0].set_xticks(range(len(sorted_layers)))
    axes[0].set_xticklabels([f"L{layer}" for layer in sorted_layers])
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: Attention Concentration
    means = [layer_stats[layer]["concentration_mean"] for layer in sorted_layers]
    stds = [layer_stats[layer]["concentration_std"] for layer in sorted_layers]
    axes[1].bar(range(len(sorted_layers)), means, yerr=stds, capsize=5, alpha=0.7, color="coral")
    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_ylabel("Concentration Ratio", fontsize=12)
    axes[1].set_title("Attention Inside/Outside Object", fontsize=14)
    axes[1].set_xticks(range(len(sorted_layers)))
    axes[1].set_xticklabels([f"L{layer}" for layer in sorted_layers])
    axes[1].grid(axis="y", alpha=0.3)

    # Plot 3: IoU
    means = [layer_stats[layer]["iou_mean"] for layer in sorted_layers]
    stds = [layer_stats[layer]["iou_std"] for layer in sorted_layers]
    axes[2].bar(range(len(sorted_layers)), means, yerr=stds, capsize=5, alpha=0.7, color="mediumseagreen")
    axes[2].set_xlabel("Layer", fontsize=12)
    axes[2].set_ylabel("IoU", fontsize=12)
    axes[2].set_title("Attention-Mask IoU", fontsize=14)
    axes[2].set_xticks(range(len(sorted_layers)))
    axes[2].set_xticklabels([f"L{layer}" for layer in sorted_layers])
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved summary plot to {output_path}")

    return layer_stats


def main():
    # 1. Config & Policy
    print("Loading Policy...")
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 2. Test frames (from user request - using available frames in white list)
    test_frames = [73, 75, 76, 77, 78]
    # with open("data/visualization/aawr_pineapple/white_list.json") as f:
    #     test_frames = json.load(f)
    # camera = "left"
    camera = "right"
    layers_to_test = [1, 4, 5, 7, 10, 17]
    # layers_to_test = range(18)
    data_dir = Path("data/visualization/aawr_pineapple")

    # Episode-based directory structure
    episode_name = "aawr_pineapple"
    episode_dir = Path("results") / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for frame_idx in test_frames:
        print(f"\n{'=' * 60}")
        print(f"Processing Frame {frame_idx}")
        print(f"{'=' * 60}")

        try:
            # Load example
            example = load_pineapple_example(camera=camera, index=frame_idx)

            # Save original images (last frame only) in episode root
            if frame_idx == test_frames[-1]:
                cv2.imwrite(
                    str(episode_dir / f"frame_{frame_idx:05d}_{camera}_orig.jpg"),
                    cv2.cvtColor(example["observation/exterior_image_1_left"], cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(episode_dir / f"frame_{frame_idx:05d}_wristview_orig.jpg"),
                    cv2.cvtColor(example["observation/wrist_image_left"], cv2.COLOR_RGB2BGR),
                )

            print("Running Inference to generate attention maps...")
            _ = policy.infer(example)

            # Analyze object detection correlation
            print("Analyzing attention-object correlation...")
            frame_results = run_object_detection(
                policy, example, frame_idx, str(data_dir), episode_dir, layers=layers_to_test, camera=camera
            )

            all_results[frame_idx] = frame_results

            # Print summary for this frame
            if frame_results:
                print(f"\n  Frame {frame_idx} Summary:")
                for layer_key, layer_data in frame_results.items():
                    if "aggregate_metrics" in layer_data:
                        layer_idx = layer_key.split("_")[1]
                        metrics_list = list(layer_data["aggregate_metrics"].values())
                        if metrics_list:
                            avg_overlap = np.mean([m["overlap_ratio"] for m in metrics_list])
                            avg_concentration = np.mean([m["attention_concentration"] for m in metrics_list])
                            print(
                                f"- Layer {layer_idx}: Overlap={avg_overlap:.3f}, Concentration={avg_concentration:.3f}"
                            )

        except FileNotFoundError as e:
            print(f"  Skipping frame {frame_idx}: {e}")
        except Exception as e:
            print(f"  Error processing frame {frame_idx}: {e}")
            import traceback

            traceback.print_exc()

    # Generate summary visualization
    print(f"\n{'=' * 60}")
    print("Generating Summary Plot...")
    print(f"{'=' * 60}")

    if all_results:
        layer_stats = generate_summary_plot(all_results, str(episode_dir / "h1_1_obj_attn_summary.png"), layers_to_test)

        results_json_path = episode_dir / "h1_1_obj_attn_results.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        all_results_serializable = convert_to_native(all_results)
        layer_stats_serializable = convert_to_native(layer_stats)

        with open(results_json_path, "w") as f:
            json.dump(
                {"frame_results": all_results_serializable, "layer_statistics": layer_stats_serializable}, f, indent=2
            )

        print(f"Saved json results to {results_json_path}")

        # Generate videos for all layers
        print(f"\n{'=' * 60}")
        print("Generating Layer Videos...")
        print(f"{'=' * 60}")

        video_paths = create_all_layer_videos(episode_dir, layers=list(layers_to_test), fps=3, add_text=True)
        if video_paths:
            print(f"\nCreated {len(video_paths)} videos:")
            for layer, path in sorted(video_paths.items()):
                print(f"  Layer {layer}: {path}")

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Processed {len(all_results)} frames")
        print(f"Output directory: {episode_dir}")
        print(f"Frame structure: {episode_dir}/object/{{frame:05d}}/L{{layer}}.jpg")
        print(f"Video directory: {episode_dir}/videos/")
        print("\nLayer Statistics:")
        for layer_idx in sorted(layer_stats.keys()):
            stats = layer_stats[layer_idx]
            print(f"\nLayer {layer_idx} (n={stats['n_samples']} samples):")
            print(f"  Overlap Ratio:    {stats['overlap_ratio_mean']:.3f} ± {stats['overlap_ratio_std']:.3f}")
            print(f"  Concentration:    {stats['concentration_mean']:.3f} ± {stats['concentration_std']:.3f}")
            print(f"  IoU:              {stats['iou_mean']:.3f} ± {stats['iou_std']:.3f}")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
