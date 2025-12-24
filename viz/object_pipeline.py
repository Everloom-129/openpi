"""
Object Detection Pipeline for Batch Processing

Process multiple episodes to evaluate attention-object correlation.
Compatible with DROID format datasets with object detection masks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import attn_map
from h1_1_object_detection import (
    create_all_layer_videos,
    generate_summary_plot,
    run_object_detection,
)
from openpi.shared import image_tools
from pipeline import copy_instruction, get_video_length, load_toy_example, timer


# ============================================================================
# CONTROL FLAGS
# ============================================================================
ENABLE_OBJECT_DETECTION = True  # Run object detection correlation analysis
ENABLE_COUNTERFACTUAL = True  # Run counterfactual prompt analysis

# ============================================================================
# CONFIGURATION
# ============================================================================
OPEN_LOOP_HORIZON = 8
LAYERS = range(18)
FPS_VIDEO = 5
CAMERA = "right"

# Counterfactual prompt configuration
COUNTERFACTUAL_LAYERS = range(18)  # Key layers for counterfactual analysis
COUNTERFACTUAL_PROMPTS = {
    "baseline": "find the {object} and pick it up",
    "duck": "find the duck toy and pick it up",
    "banana": "find the banana and pick it up",
    "cat": "find the cat toy and pick it up",
    "bottle": "find the bottle and pick it up",
}
ANALYSIS_CAMERA = "wrist"  # Which camera's attention to analyze for counterfactual

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def has_object_masks(data_dir: Path, frame_idx: int) -> bool:
    """Check if object detection masks exist for a given frame"""
    dinox_info_path = data_dir / "dinox_info.json"
    if not dinox_info_path.exists():
        return False

    try:
        with open(dinox_info_path) as f:
            dinox_info = json.load(f)

        frame_key = str(frame_idx)
        if frame_key not in dinox_info:
            return False

        detections = dinox_info[frame_key]
        # Check if any detection has valid mask
        for det in detections:
            if det.get("mask_file") and det.get("score", 0) > 0:
                mask_path = data_dir / "masks" / det["mask_file"]
                if mask_path.exists():
                    return True
        return False
    except Exception as e:
        print(f"Error checking masks: {e}")
        return False


def get_white_list_frames(data_dir: Path) -> list[int]:
    """Load white list frames if available, otherwise return empty list"""
    white_list_path = data_dir / "white_list.json"
    if not white_list_path.exists():
        return []

    try:
        with open(white_list_path) as f:
            white_list = json.load(f)
        return sorted(white_list)
    except Exception as e:
        print(f"Error loading white list: {e}")
        return []


def extract_attention_map_from_policy(policy, example, layer: int, camera: str = "wrist"):
    """
    Extract attention map for a specific layer and camera after inference.

    Args:
        policy: Trained policy
        example: Input example (already processed by policy.infer)
        layer: Layer index
        camera: 'wrist' or 'exterior'

    Returns:
        Attention map (16x16) or None if not found
    """
    # Load attention map (should already exist from policy.infer call)
    attn_path = Path("results") / "layers_prefix" / f"attn_map_layer_{layer}.npy"
    if not attn_path.exists():
        return None

    attn_map = np.load(attn_path)  # [Batch, Heads, Seq, Seq]

    # Process attention
    if attn_map.ndim == 4:
        attn_map = attn_map[0]  # Remove batch: [Heads, Seq, Seq]

    # Average across heads
    attn_avg = attn_map.mean(axis=0) if attn_map.ndim == 3 else attn_map

    # Extract text->image attention
    num_img = 256
    total_img = 512

    if attn_avg.shape[0] <= total_img:
        return None

    # Max over text tokens
    text_attn = attn_avg[total_img:, :total_img].max(axis=0)

    # Select camera
    if camera == "wrist":
        attn_cam = text_attn[num_img:total_img].reshape(16, 16)
    else:  # exterior
        attn_cam = text_attn[:num_img].reshape(16, 16)

    return attn_cam


def visualize_counterfactual_comparison(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    camera: str = "wrist",
):
    """Visualize attention maps for different prompts side-by-side."""
    n_prompts = len(attention_maps)
    fig, axes = plt.subplots(2, n_prompts + 1, figsize=(5 * (n_prompts + 1), 10))

    # Resize reference image
    img_jax = jnp.array(reference_image)
    img_224 = np.array(image_tools.resize_with_pad(img_jax, 224, 224)).astype(np.uint8)

    # First column: Original image (both rows)
    axes[0, 0].imshow(img_224)
    axes[0, 0].set_title(f"{camera.capitalize()} Camera\n(Original)", fontsize=10)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(img_224)
    axes[1, 0].set_title("Reference", fontsize=10)
    axes[1, 0].axis("off")

    # Other columns: Attention maps
    for idx, (prompt_key, attn_map) in enumerate(attention_maps.items(), start=1):
        prompt_text = prompts[prompt_key]

        # Resize attention to 224x224
        attn_224 = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Row 1: Heatmap only
        im = axes[0, idx].imshow(attn_224, cmap="jet", interpolation="bilinear", vmin=0)
        axes[0, idx].set_title(f'"{prompt_text}"\n(Heatmap)', fontsize=9, wrap=True)
        axes[0, idx].axis("off")
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

        # Row 2: Overlay on image
        attn_norm = np.uint8(255 * attn_224 / (np.max(attn_224) + 1e-8))
        attn_color = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
        attn_color = cv2.cvtColor(attn_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_224, 0.6, attn_color, 0.4, 0)

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'"{prompt_text}"\n(Overlay)', fontsize=9, wrap=True)
        axes[1, idx].axis("off")

    plt.suptitle(
        f"Attention Shift with Different Prompts (Layer {layer}, {camera.capitalize()} Camera)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_counterfactual_difference(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    baseline_key: str,
    camera: str = "wrist",
):
    """Visualize difference maps: M_counterfactual - M_baseline."""
    if baseline_key not in attention_maps:
        return

    baseline_attn = attention_maps[baseline_key]
    counterfactual_keys = [k for k in attention_maps if k != baseline_key]

    n_counterfactuals = len(counterfactual_keys)
    fig, axes = plt.subplots(2, n_counterfactuals + 1, figsize=(5 * (n_counterfactuals + 1), 10))

    # Resize reference image
    img_jax = jnp.array(reference_image)
    img_224 = np.array(image_tools.resize_with_pad(img_jax, 224, 224)).astype(np.uint8)

    # First column: Baseline attention
    baseline_224 = cv2.resize(baseline_attn, (224, 224), interpolation=cv2.INTER_CUBIC)

    im = axes[0, 0].imshow(baseline_224, cmap="jet", interpolation="bilinear")
    axes[0, 0].set_title(f'Baseline:\n"{prompts[baseline_key]}"', fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    axes[1, 0].imshow(img_224)
    axes[1, 0].set_title("Reference Image", fontsize=10)
    axes[1, 0].axis("off")

    # Other columns: Difference maps
    for idx, cf_key in enumerate(counterfactual_keys, start=1):
        cf_attn = attention_maps[cf_key]
        cf_text = prompts[cf_key]

        # Compute difference
        diff_map = cf_attn - baseline_attn
        diff_224 = cv2.resize(diff_map, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Row 1: Difference map (diverging colormap)
        vmax = np.abs(diff_224).max()
        im = axes[0, idx].imshow(diff_224, cmap="RdBu_r", interpolation="bilinear", vmin=-vmax, vmax=vmax)
        axes[0, idx].set_title(f'Δ: "{cf_text}"\n- Baseline', fontsize=9)
        axes[0, idx].axis("off")
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

        # Row 2: Overlay difference on image
        diff_norm = (diff_224 - diff_224.min()) / (diff_224.max() - diff_224.min() + 1e-8)
        diff_norm = np.uint8(255 * diff_norm)
        diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_224, 0.6, diff_color, 0.4, 0)

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'Overlay: "{cf_text}"', fontsize=9)
        axes[1, idx].axis("off")

    plt.suptitle(
        f"Attention Difference Maps (Layer {layer}, {camera.capitalize()} Camera)\n"
        f"Red: More attention | Blue: Less attention",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_counterfactual_statistics(attention_maps: dict, prompts: dict, baseline_key: str):
    """Compute statistics for attention shift analysis."""
    if baseline_key not in attention_maps:
        return {}

    baseline_attn = attention_maps[baseline_key]
    stats = {}

    for key, attn in attention_maps.items():
        if key == baseline_key:
            continue

        # Compute difference
        diff = attn - baseline_attn

        # Metrics
        stats[key] = {
            "prompt": prompts[key],
            "mean_diff": float(np.mean(diff)),
            "abs_mean_diff": float(np.mean(np.abs(diff))),
            "max_increase": float(np.max(diff)),
            "max_decrease": float(np.min(diff)),
            "l2_distance": float(np.linalg.norm(diff)),
            "correlation": float(np.corrcoef(baseline_attn.flatten(), attn.flatten())[0, 1]),
        }

    return stats


def run_counterfactual_analysis(
    policy,
    example_base: dict,
    frame_idx: int,
    episode_dir: Path,
    layers: list[int],
    prompts: dict,
    baseline_key: str,
    camera: str = "wrist",
    object_name: str = "object",
):
    """
    Run counterfactual prompt analysis on a single frame.

    Args:
        policy: Trained policy
        example_base: Base example (images, state)
        frame_idx: Frame index
        episode_dir: Episode output directory
        layers: List of layer indices to analyze
        prompts: Dictionary of prompt templates
        baseline_key: Key for baseline prompt
        camera: Which camera's attention to analyze
        object_name: Object name to substitute in baseline prompt

    Returns:
        Dictionary with results for each layer
    """
    # Create counterfactual output directory
    cf_dir = episode_dir / "counterfactual" / f"{frame_idx:05d}"
    cf_dir.mkdir(parents=True, exist_ok=True)

    # Get reference image
    if camera == "wrist":
        reference_image = example_base["observation/wrist_image_left"]
    else:
        reference_image = example_base["observation/exterior_image_1_left"]

    # Save reference image
    cv2.imwrite(
        str(cf_dir / f"frame_{frame_idx:05d}_{camera}_reference.jpg"),
        cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR),
    )

    # Prepare actual prompts (substitute object name in baseline)
    actual_prompts = {}
    for key, template in prompts.items():
        if key == baseline_key:
            actual_prompts[key] = template.format(object=object_name)
        else:
            actual_prompts[key] = template

    all_layer_results = {}

    for layer in layers:
        print(f"    Counterfactual Layer {layer}...")

        attention_maps = {}

        # Extract attention for each prompt
        for prompt_key, prompt_text in actual_prompts.items():
            # Create example with this prompt
            example = example_base.copy()
            example["prompt"] = prompt_text

            # Run inference
            _ = policy.infer(example)

            # Extract attention
            attn_map = extract_attention_map_from_policy(policy, example, layer, camera=camera)

            if attn_map is not None:
                attention_maps[prompt_key] = attn_map

        if not attention_maps:
            continue

        # Visualize comparison
        comparison_path = cf_dir / f"L{layer:02d}_prompt_comparison.png"
        visualize_counterfactual_comparison(
            attention_maps, actual_prompts, reference_image, str(comparison_path), layer, camera
        )

        # Visualize difference maps
        if baseline_key in attention_maps and len(attention_maps) > 1:
            diff_path = cf_dir / f"L{layer:02d}_difference_maps.png"
            visualize_counterfactual_difference(
                attention_maps, actual_prompts, reference_image, str(diff_path), layer, baseline_key, camera
            )

            # Compute statistics
            stats = compute_counterfactual_statistics(attention_maps, actual_prompts, baseline_key)
            all_layer_results[layer] = stats

    return all_layer_results


def create_counterfactual_videos(
    episode_dir: Path,
    layers: list[int],
    video_types: list[str] = None,
    fps: int = 5,
    add_text: bool = True,
):
    """
    Create videos for counterfactual analysis results.

    Args:
        episode_dir: Base directory containing counterfactual/frame folders
        layers: List of layer indices to create videos for
        video_types: List of video types to create ['comparison', 'difference'], or None for all
        fps: Frames per second for output videos
        add_text: Whether to add frame number text overlay

    Returns:
        Dictionary mapping (layer, video_type) to video path
    """
    if video_types is None:
        video_types = ["comparison", "difference"]

    cf_dir = episode_dir / "counterfactual"
    if not cf_dir.exists():
        print(f"Counterfactual directory not found: {cf_dir}")
        return {}

    # Collect all frame directories
    frame_dirs = sorted([d for d in cf_dir.iterdir() if d.is_dir()])
    if not frame_dirs:
        print(f"No frame directories found in {cf_dir}")
        return {}

    video_dir = episode_dir / "videos_counterfactual"
    video_dir.mkdir(exist_ok=True)

    video_paths = {}

    for layer in layers:
        for video_type in video_types:
            frames = []
            frame_indices = []

            # Determine filename pattern
            if video_type == "comparison":
                pattern = f"L{layer:02d}_prompt_comparison.png"
            elif video_type == "difference":
                pattern = f"L{layer:02d}_difference_maps.png"
            else:
                continue

            # Collect frames
            for frame_dir in frame_dirs:
                img_path = frame_dir / pattern
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        if add_text:
                            # Add frame number overlay
                            frame_idx = int(frame_dir.name)
                            cv2.putText(
                                img,
                                f"Frame: {frame_idx}",
                                (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255, 255, 255),
                                4,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                img,
                                f"Layer: {layer}",
                                (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255, 255, 255),
                                4,
                                cv2.LINE_AA,
                            )
                        frames.append(img)
                        frame_indices.append(frame_dir.name)

            if not frames:
                print(f"  No frames found for Layer {layer} ({video_type})")
                continue

            # Create video
            output_path = video_dir / f"L{layer:02d}_cf_{video_type}.webm"
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"VP90")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in frames:
                out.write(frame)

            out.release()
            video_paths[(layer, video_type)] = str(output_path)
            print(f"  ✓ Created {video_type} video for Layer {layer}: {output_path.name} ({len(frames)} frames)")

    return video_paths


def create_all_counterfactual_videos(
    episode_dir: Path, layers: list[int] | None = None, fps: int = 5, add_text: bool = True
):
    """
    Create all counterfactual videos for an episode.

    Args:
        episode_dir: Base directory containing counterfactual/frame folders
        layers: List of layer indices, or None to auto-detect
        fps: Frames per second for output videos
        add_text: Whether to add frame number text overlay

    Returns:
        Dictionary mapping (layer, video_type) to video path
    """
    # Auto-detect layers if not specified
    if layers is None:
        cf_dir = episode_dir / "counterfactual"
        if not cf_dir.exists():
            print("Counterfactual directory not found, cannot auto-detect layers")
            return {}

        frame_dirs = sorted([d for d in cf_dir.iterdir() if d.is_dir()])
        if not frame_dirs:
            print("No frames found, cannot auto-detect layers")
            return {}

        # Check first frame for available layers
        first_frame = frame_dirs[0]
        comparison_files = sorted(first_frame.glob("L*_prompt_comparison.png"))
        if comparison_files:
            layers = [int(f.stem.split("_")[0][1:]) for f in comparison_files]
            print(f"  Auto-detected {len(layers)} layers: {layers}")
        else:
            print("  No counterfactual visualizations found")
            return {}

    # Create videos
    video_paths = create_counterfactual_videos(episode_dir, layers, fps=fps, add_text=add_text)

    return video_paths


def aggregate_counterfactual_analysis(
    results_root: Path, layers: list[int], prompts: dict, output_dir: Path | None = None
):
    """
    Aggregate counterfactual results from multiple episodes.

    Args:
        results_root: Root directory containing all processed episodes
        layers: List of layer indices to analyze
        prompts: Dictionary of prompts used
        output_dir: Output directory for aggregated results (defaults to results_root)

    Returns:
        Dictionary with aggregated statistics
    """
    if output_dir is None:
        output_dir = results_root

    output_dir = Path(output_dir)
    cf_aggregate_dir = output_dir / "counterfactual_aggregate"
    cf_aggregate_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("AGGREGATING COUNTERFACTUAL RESULTS")
    print(f"{'=' * 60}\n")
    print(f"Results root: {results_root}")

    # Collect all counterfactual results
    all_cf_data = []
    episode_count = {"success": 0, "failure": 0}

    for outcome in ["success", "failure"]:
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue

        # Find all counterfactual JSON files
        json_files = list(outcome_dir.rglob("h2_1_counterfactual_results.json"))
        print(f"Found {len(json_files)} episodes with counterfactual data in {outcome}")

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)

                episode_path = json_path.parent
                episode_id = episode_path.name

                all_cf_data.append(
                    {
                        "outcome": outcome,
                        "episode_id": episode_id,
                        "episode_path": str(episode_path),
                        "frame_results": data.get("frame_results", {}),
                        "frame_count": len(data.get("frame_results", {})),
                    }
                )
                episode_count[outcome] += 1

            except Exception as e:
                print(f"  Error loading {json_path}: {e}")

    if not all_cf_data:
        print("No counterfactual data found!")
        return None

    total_episodes = sum(episode_count.values())
    print(f"\nLoaded {total_episodes} episodes:")
    print(f"  Success: {episode_count['success']}")
    print(f"  Failure: {episode_count['failure']}")

    # Get prompt keys (excluding baseline)
    prompt_keys = [k for k in prompts.keys() if k != "baseline"]

    # Aggregate statistics by layer and prompt
    layer_aggregates = {}
    for layer in layers:
        layer_aggregates[layer] = {}
        for prompt_key in prompt_keys:
            layer_aggregates[layer][prompt_key] = {
                "success": {"l2_distance": [], "correlation": [], "abs_mean_diff": []},
                "failure": {"l2_distance": [], "correlation": [], "abs_mean_diff": []},
            }

    # Collect data
    for episode in all_cf_data:
        outcome = episode["outcome"]
        frame_results = episode["frame_results"]

        for frame_idx, frame_data in frame_results.items():
            for layer_str, layer_stats in frame_data.items():
                layer = int(layer_str)
                if layer not in layers:
                    continue

                for prompt_key, stats in layer_stats.items():
                    if prompt_key not in prompt_keys:
                        continue

                    target = layer_aggregates[layer][prompt_key][outcome]
                    target["l2_distance"].append(stats.get("l2_distance", 0))
                    target["correlation"].append(stats.get("correlation", 0))
                    target["abs_mean_diff"].append(stats.get("abs_mean_diff", 0))

    # Generate visualizations
    print("\nGenerating counterfactual aggregate visualizations...")

    # 1. L2 Distance comparison across prompts and layers
    fig, axes = plt.subplots(len(prompt_keys), 1, figsize=(14, 4 * len(prompt_keys)))
    if len(prompt_keys) == 1:
        axes = [axes]

    colors = {"success": "steelblue", "failure": "coral"}

    for idx, prompt_key in enumerate(prompt_keys):
        ax = axes[idx]
        x_pos = np.arange(len(layers))
        width = 0.35

        success_means = []
        success_stds = []
        failure_means = []
        failure_stds = []

        for layer in layers:
            success_vals = layer_aggregates[layer][prompt_key]["success"]["l2_distance"]
            failure_vals = layer_aggregates[layer][prompt_key]["failure"]["l2_distance"]

            success_means.append(np.mean(success_vals) if success_vals else 0)
            success_stds.append(np.std(success_vals) if success_vals else 0)
            failure_means.append(np.mean(failure_vals) if failure_vals else 0)
            failure_stds.append(np.std(failure_vals) if failure_vals else 0)

        ax.bar(
            x_pos - width / 2,
            success_means,
            width,
            yerr=success_stds,
            label="Success",
            color=colors["success"],
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x_pos + width / 2,
            failure_means,
            width,
            yerr=failure_stds,
            label="Failure",
            color=colors["failure"],
            alpha=0.8,
            capsize=5,
        )

        ax.set_ylabel("L2 Distance", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_title(f'Prompt: "{prompts[prompt_key]}"', fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Add threshold line
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (1.0)")

    plt.suptitle(
        f"Counterfactual L2 Distance Across Layers\n({total_episodes} episodes)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    l2_path = cf_aggregate_dir / "cf_l2_distance_comparison.png"
    plt.savefig(l2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ L2 Distance plot: {l2_path}")

    # 2. Correlation comparison
    fig, axes = plt.subplots(len(prompt_keys), 1, figsize=(14, 4 * len(prompt_keys)))
    if len(prompt_keys) == 1:
        axes = [axes]

    for idx, prompt_key in enumerate(prompt_keys):
        ax = axes[idx]
        x_pos = np.arange(len(layers))
        width = 0.35

        success_means = []
        success_stds = []
        failure_means = []
        failure_stds = []

        for layer in layers:
            success_vals = layer_aggregates[layer][prompt_key]["success"]["correlation"]
            failure_vals = layer_aggregates[layer][prompt_key]["failure"]["correlation"]

            success_means.append(np.mean(success_vals) if success_vals else 0)
            success_stds.append(np.std(success_vals) if success_vals else 0)
            failure_means.append(np.mean(failure_vals) if failure_vals else 0)
            failure_stds.append(np.std(failure_vals) if failure_vals else 0)

        ax.bar(
            x_pos - width / 2,
            success_means,
            width,
            yerr=success_stds,
            label="Success",
            color=colors["success"],
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x_pos + width / 2,
            failure_means,
            width,
            yerr=failure_stds,
            label="Failure",
            color=colors["failure"],
            alpha=0.8,
            capsize=5,
        )

        ax.set_ylabel("Correlation", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_title(f'Prompt: "{prompts[prompt_key]}"', fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add threshold line
        ax.axhline(y=0.7, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Threshold (0.7)")

    plt.suptitle(
        f"Counterfactual Correlation Across Layers\n({total_episodes} episodes)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    corr_path = cf_aggregate_dir / "cf_correlation_comparison.png"
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Correlation plot: {corr_path}")

    # 3. Heatmap: Layer x Prompt (L2 Distance)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for outcome_idx, outcome in enumerate(["success", "failure"]):
        ax = axes[outcome_idx]

        # Create matrix: rows=prompts, cols=layers
        matrix = np.zeros((len(prompt_keys), len(layers)))

        for prompt_idx, prompt_key in enumerate(prompt_keys):
            for layer_idx, layer in enumerate(layers):
                vals = layer_aggregates[layer][prompt_key][outcome]["l2_distance"]
                matrix[prompt_idx, layer_idx] = np.mean(vals) if vals else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.set_yticks(range(len(prompt_keys)))
        ax.set_yticklabels([prompts[k] for k in prompt_keys], fontsize=9)
        ax.set_title(f"{outcome.capitalize()} Episodes", fontsize=12, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Counterfactual Prompt", fontsize=11)

        # Add text annotations
        for i in range(len(prompt_keys)):
            for j in range(len(layers)):
                text_color = "white" if matrix[i, j] > 1.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("L2 Distance Heatmap: Prompt Control Strength", fontsize=14, fontweight="bold")
    plt.tight_layout()

    heatmap_path = cf_aggregate_dir / "cf_l2_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap: {heatmap_path}")

    # 4. Generate summary report
    report_lines = [
        "# Counterfactual Prompt Analysis - Multi-Episode Report\n\n",
        f"**Total Episodes**: {total_episodes}\n",
        f"- Success: {episode_count['success']}\n",
        f"- Failure: {episode_count['failure']}\n\n",
        f"**Layers Analyzed**: {list(layers)}\n",
        f"**Prompts Tested**: {list(prompt_keys)}\n\n",
        "## Hypothesis 2.1: Text Prompt Controls Visual Attention\n\n",
        "### Interpretation Criteria\n\n",
        "**Hypothesis SUPPORTED** (Prompt controls attention):\n",
        "- L2 Distance > 1.0\n",
        "- Correlation < 0.7\n\n",
        "**Hypothesis NOT SUPPORTED** (Prompt ignored):\n",
        "- L2 Distance < 0.5\n",
        "- Correlation > 0.9\n\n",
        "## Results Summary\n\n",
    ]

    for outcome in ["success", "failure"]:
        report_lines.append(f"### {outcome.capitalize()} Episodes\n\n")

        for layer in layers:
            report_lines.append(f"#### Layer {layer}\n\n")
            report_lines.append("| Prompt | L2 Distance | Correlation | |Δ| | Samples |\n")
            report_lines.append("|--------|-------------|-------------|------|----------|\n")

            for prompt_key in prompt_keys:
                l2_vals = layer_aggregates[layer][prompt_key][outcome]["l2_distance"]
                corr_vals = layer_aggregates[layer][prompt_key][outcome]["correlation"]
                diff_vals = layer_aggregates[layer][prompt_key][outcome]["abs_mean_diff"]

                if l2_vals:
                    l2_mean = np.mean(l2_vals)
                    corr_mean = np.mean(corr_vals)
                    diff_mean = np.mean(diff_vals)
                    n_samples = len(l2_vals)

                    # Determine if hypothesis is supported
                    supported = "✅" if (l2_mean > 1.0 and corr_mean < 0.7) else "❌"

                    report_lines.append(
                        f"| {supported} {prompt_key} | {l2_mean:.3f} | {corr_mean:.3f} | {diff_mean:.4f} | {n_samples} |\n"
                    )
                else:
                    report_lines.append(f"| {prompt_key} | - | - | - | 0 |\n")

            report_lines.append("\n")

    # Overall findings
    report_lines.append("## Overall Findings\n\n")

    # Calculate average metrics across all prompts and layers
    all_l2_success = []
    all_l2_failure = []
    all_corr_success = []
    all_corr_failure = []

    for layer in layers:
        for prompt_key in prompt_keys:
            all_l2_success.extend(layer_aggregates[layer][prompt_key]["success"]["l2_distance"])
            all_l2_failure.extend(layer_aggregates[layer][prompt_key]["failure"]["l2_distance"])
            all_corr_success.extend(layer_aggregates[layer][prompt_key]["success"]["correlation"])
            all_corr_failure.extend(layer_aggregates[layer][prompt_key]["failure"]["correlation"])

    report_lines.append("### Success Episodes\n")
    if all_l2_success:
        report_lines.append(f"- Average L2 Distance: {np.mean(all_l2_success):.3f} ± {np.std(all_l2_success):.3f}\n")
        report_lines.append(
            f"- Average Correlation: {np.mean(all_corr_success):.3f} ± {np.std(all_corr_success):.3f}\n"
        )
        hypothesis_supported = np.mean(all_l2_success) > 1.0 and np.mean(all_corr_success) < 0.7
        report_lines.append(
            f"- **Hypothesis 2.1**: {'✅ SUPPORTED' if hypothesis_supported else '❌ NOT SUPPORTED'}\n\n"
        )
    else:
        report_lines.append("- No data\n\n")

    report_lines.append("### Failure Episodes\n")
    if all_l2_failure:
        report_lines.append(f"- Average L2 Distance: {np.mean(all_l2_failure):.3f} ± {np.std(all_l2_failure):.3f}\n")
        report_lines.append(
            f"- Average Correlation: {np.mean(all_corr_failure):.3f} ± {np.std(all_corr_failure):.3f}\n"
        )
        hypothesis_supported = np.mean(all_l2_failure) > 1.0 and np.mean(all_corr_failure) < 0.7
        report_lines.append(
            f"- **Hypothesis 2.1**: {'✅ SUPPORTED' if hypothesis_supported else '❌ NOT SUPPORTED'}\n\n"
        )
    else:
        report_lines.append("- No data\n\n")

    # Best and worst prompts
    report_lines.append("### Prompt Effectiveness Ranking\n\n")
    report_lines.append("Ranked by L2 Distance (higher = stronger prompt control):\n\n")

    prompt_scores = {}
    for prompt_key in prompt_keys:
        all_l2 = []
        for layer in layers:
            all_l2.extend(layer_aggregates[layer][prompt_key]["success"]["l2_distance"])
            all_l2.extend(layer_aggregates[layer][prompt_key]["failure"]["l2_distance"])
        if all_l2:
            prompt_scores[prompt_key] = np.mean(all_l2)

    for rank, (prompt_key, score) in enumerate(sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True), 1):
        report_lines.append(f'{rank}. **{prompt_key}**: {score:.3f} - "{prompts[prompt_key]}"\n')

    report_path = cf_aggregate_dir / "cf_multi_episode_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"  ✓ Report: {report_path}")

    # Save aggregated JSON
    aggregated_json = {
        "total_episodes": total_episodes,
        "episode_count": episode_count,
        "layers": list(layers),
        "prompts": prompts,
        "prompt_keys": prompt_keys,
        "layer_aggregates": {
            str(layer): {
                prompt_key: {
                    outcome: {
                        metric: {
                            "values": layer_aggregates[layer][prompt_key][outcome][metric],
                            "mean": float(np.mean(layer_aggregates[layer][prompt_key][outcome][metric]))
                            if layer_aggregates[layer][prompt_key][outcome][metric]
                            else 0,
                            "std": float(np.std(layer_aggregates[layer][prompt_key][outcome][metric]))
                            if layer_aggregates[layer][prompt_key][outcome][metric]
                            else 0,
                            "count": len(layer_aggregates[layer][prompt_key][outcome][metric]),
                        }
                        for metric in ["l2_distance", "correlation", "abs_mean_diff"]
                    }
                    for outcome in ["success", "failure"]
                }
                for prompt_key in prompt_keys
            }
            for layer in layers
        },
    }

    json_path = cf_aggregate_dir / "cf_multi_episode_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(aggregated_json, f, indent=2)
    print(f"  ✓ JSON data: {json_path}")

    print(f"\n{'=' * 60}")
    print("COUNTERFACTUAL AGGREGATION COMPLETE")
    print(f"{'=' * 60}\n")

    return aggregated_json


def aggregate_episodes_analysis(results_root: Path, layers: list[int], output_dir: Path | None = None):
    """
    Aggregate results from multiple episodes and generate comprehensive analysis.

    Args:
        results_root: Root directory containing all processed episodes
        layers: List of layer indices to analyze
        output_dir: Output directory for aggregated results (defaults to results_root)

    Returns:
        Dictionary with aggregated statistics
    """
    if output_dir is None:
        output_dir = results_root

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("AGGREGATING MULTI-EPISODE RESULTS")
    print(f"{'=' * 60}\n")
    print(f"Results root: {results_root}")

    # Collect all episode results
    all_episode_data = []
    episode_count = {"success": 0, "failure": 0}

    for outcome in ["success", "failure"]:
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue

        # Find all JSON result files
        json_files = list(outcome_dir.rglob("h1_1_obj_attn_results.json"))
        print(f"Found {len(json_files)} episodes in {outcome}")

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)

                episode_path = json_path.parent
                episode_id = episode_path.name

                # Extract layer statistics
                if "layer_statistics" in data:
                    all_episode_data.append(
                        {
                            "outcome": outcome,
                            "episode_id": episode_id,
                            "episode_path": str(episode_path),
                            "layer_stats": data["layer_statistics"],
                            "frame_count": len(data.get("frame_results", {})),
                        }
                    )
                    episode_count[outcome] += 1

            except Exception as e:
                print(f"  Error loading {json_path}: {e}")

    if not all_episode_data:
        print("No episode data found!")
        return None

    total_episodes = sum(episode_count.values())
    print(f"\nLoaded {total_episodes} episodes:")
    print(f"  Success: {episode_count['success']}")
    print(f"  Failure: {episode_count['failure']}")

    # Aggregate statistics by layer
    layer_aggregates = {}
    for layer in layers:
        layer_key = str(layer)
        success_metrics = {"overlap": [], "concentration": [], "iou": []}
        failure_metrics = {"overlap": [], "concentration": [], "iou": []}

        for episode in all_episode_data:
            layer_stats = episode["layer_stats"]
            if layer_key not in layer_stats:
                continue

            stats = layer_stats[layer_key]
            outcome = episode["outcome"]
            target = success_metrics if outcome == "success" else failure_metrics

            target["overlap"].append(stats.get("overlap_ratio_mean", 0))
            target["concentration"].append(stats.get("concentration_mean", 0))
            target["iou"].append(stats.get("iou_mean", 0))

        layer_aggregates[layer] = {
            "success": success_metrics,
            "failure": failure_metrics,
        }

    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")

    # 1. Multi-panel comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    metrics = ["overlap", "concentration", "iou"]
    metric_names = ["Overlap Ratio", "Attention Concentration", "IoU"]
    colors = {"success": "steelblue", "failure": "coral"}

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Prepare data for plotting
        x_pos = np.arange(len(layers))
        width = 0.35

        success_means = []
        success_stds = []
        failure_means = []
        failure_stds = []

        for layer in layers:
            agg = layer_aggregates[layer]

            success_vals = agg["success"][metric]
            failure_vals = agg["failure"][metric]

            success_means.append(np.mean(success_vals) if success_vals else 0)
            success_stds.append(np.std(success_vals) if success_vals else 0)
            failure_means.append(np.mean(failure_vals) if failure_vals else 0)
            failure_stds.append(np.std(failure_vals) if failure_vals else 0)

        # Plot bars
        ax.bar(
            x_pos - width / 2,
            success_means,
            width,
            yerr=success_stds,
            label="Success",
            color=colors["success"],
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x_pos + width / 2,
            failure_means,
            width,
            yerr=failure_stds,
            label="Failure",
            color=colors["failure"],
            alpha=0.8,
            capsize=5,
        )

        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Add episode count annotation
        if idx == 0:
            ax.text(
                0.02,
                0.98,
                f"Success: {episode_count['success']} episodes | Failure: {episode_count['failure']} episodes",
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    axes[-1].set_xlabel("Layer", fontsize=12, fontweight="bold")
    plt.suptitle(
        "Multi-Episode Attention-Object Correlation Analysis\n(Success vs Failure)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    comparison_path = output_dir / "multi_episode_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Comparison plot: {comparison_path}")

    # 2. Distribution plots (violin plots)
    fig, axes = plt.subplots(len(metrics), len(layers), figsize=(18, 10))

    for metric_idx, metric in enumerate(metrics):
        for layer_idx, layer in enumerate(layers):
            ax = axes[metric_idx, layer_idx]
            agg = layer_aggregates[layer]

            data_to_plot = []
            labels = []

            for outcome in ["success", "failure"]:
                vals = agg[outcome][metric]
                if vals:
                    data_to_plot.append(vals)
                    labels.append(outcome.capitalize())

            if data_to_plot:
                parts = ax.violinplot(
                    data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True
                )

                # Color the violins
                for pc, outcome in zip(parts["bodies"], ["success", "failure"][: len(data_to_plot)], strict=False):
                    pc.set_facecolor(colors[outcome])
                    pc.set_alpha(0.7)

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=8)

            ax.set_title(f"L{layer}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            if layer_idx == 0:
                ax.set_ylabel(metric_names[metric_idx], fontsize=10)

    plt.suptitle("Distribution of Metrics Across Episodes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    distribution_path = output_dir / "multi_episode_distributions.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Distribution plot: {distribution_path}")

    # 3. Heatmap: Layer x Outcome
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Create matrix: rows=outcomes, cols=layers
        matrix = np.zeros((2, len(layers)))

        for layer_idx, layer in enumerate(layers):
            agg = layer_aggregates[layer]
            success_vals = agg["success"][metric]
            failure_vals = agg["failure"][metric]

            matrix[0, layer_idx] = np.mean(success_vals) if success_vals else 0
            matrix[1, layer_idx] = np.mean(failure_vals) if failure_vals else 0

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Success", "Failure"])
        ax.set_title(metric_name, fontsize=12, fontweight="bold")

        # Add text annotations
        for i in range(2):
            for j in range(len(layers)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Heatmap: Attention-Object Metrics by Outcome", fontsize=14, fontweight="bold")
    plt.tight_layout()

    heatmap_path = output_dir / "multi_episode_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap: {heatmap_path}")

    # 4. Generate summary report
    report_lines = [
        "# Multi-Episode Attention-Object Correlation Report\n",
        f"## Dataset: {results_root.name}\n",
        f"**Total Episodes**: {total_episodes}\n",
        f"- Success: {episode_count['success']}\n",
        f"- Failure: {episode_count['failure']}\n",
        f"\n**Layers Analyzed**: {layers}\n",
        "\n## Summary Statistics\n",
    ]

    for layer in layers:
        report_lines.append(f"\n### Layer {layer}\n")
        agg = layer_aggregates[layer]

        for outcome in ["success", "failure"]:
            report_lines.append(f"\n**{outcome.capitalize()}**:\n")
            for metric, metric_name in zip(metrics, metric_names):
                vals = agg[outcome][metric]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    median = np.median(vals)
                    report_lines.append(
                        f"- {metric_name}: {mean:.4f} ± {std:.4f} (median: {median:.4f}, n={len(vals)})\n"
                    )
                else:
                    report_lines.append(f"- {metric_name}: No data\n")

    # Key findings
    report_lines.append("\n## Key Findings\n")

    # Find best layers
    best_layers = {"overlap": None, "concentration": None, "iou": None}
    for metric in metrics:
        max_val = 0
        best_layer = None
        for layer in layers:
            success_vals = layer_aggregates[layer]["success"][metric]
            if success_vals:
                mean_val = np.mean(success_vals)
                if mean_val > max_val:
                    max_val = mean_val
                    best_layer = layer
        best_layers[metric] = (best_layer, max_val)

    report_lines.append(f"\n**Best Performing Layers (Success Episodes)**:\n")
    for metric, metric_name in zip(metrics, metric_names):
        layer, val = best_layers[metric]
        report_lines.append(f"- {metric_name}: Layer {layer} ({val:.4f})\n")

    # Success vs Failure comparison
    report_lines.append(f"\n**Success vs Failure Comparison**:\n")
    for layer in layers:
        report_lines.append(f"\nLayer {layer}:\n")
        for metric, metric_name in zip(metrics, metric_names):
            success_vals = layer_aggregates[layer]["success"][metric]
            failure_vals = layer_aggregates[layer]["failure"][metric]

            if success_vals and failure_vals:
                success_mean = np.mean(success_vals)
                failure_mean = np.mean(failure_vals)
                diff = success_mean - failure_mean
                diff_pct = (diff / success_mean * 100) if success_mean != 0 else 0

                report_lines.append(f"  - {metric_name}: Success={success_mean:.4f}, Failure={failure_mean:.4f}, ")
                report_lines.append(f"Δ={diff:+.4f} ({diff_pct:+.1f}%)\n")

    report_path = output_dir / "multi_episode_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"  ✓ Report: {report_path}")

    # Save aggregated data as JSON
    aggregated_json = {
        "total_episodes": total_episodes,
        "episode_count": episode_count,
        "layers": list(layers),  # Convert range to list for JSON serialization
        "layer_aggregates": {
            str(layer): {
                outcome: {
                    metric: {
                        "values": layer_aggregates[layer][outcome][metric],
                        "mean": float(np.mean(layer_aggregates[layer][outcome][metric]))
                        if layer_aggregates[layer][outcome][metric]
                        else 0,
                        "std": float(np.std(layer_aggregates[layer][outcome][metric]))
                        if layer_aggregates[layer][outcome][metric]
                        else 0,
                        "count": len(layer_aggregates[layer][outcome][metric]),
                    }
                    for metric in metrics
                }
                for outcome in ["success", "failure"]
            }
            for layer in layers
        },
    }

    json_path = output_dir / "multi_episode_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(aggregated_json, f, indent=2)
    print(f"  ✓ JSON data: {json_path}")

    print(f"\n{'=' * 60}")
    print("AGGREGATION COMPLETE")
    print(f"{'=' * 60}\n")

    return aggregated_json


def batch_generate_counterfactual_videos(results_root: Path, layers: list[int], fps: int = 5):
    """
    Batch generate counterfactual videos for all processed episodes.

    This function can be run independently to generate videos for episodes
    that already have counterfactual analysis results but are missing videos.

    Args:
        results_root: Root directory containing all processed episodes
        layers: List of layer indices to create videos for
        fps: Frames per second for output videos

    Returns:
        Statistics dictionary
    """
    print(f"\n{'=' * 60}")
    print("BATCH GENERATING COUNTERFACTUAL VIDEOS")
    print(f"{'=' * 60}\n")
    print(f"Results root: {results_root}")
    print(f"Layers: {list(layers)}")
    print(f"FPS: {fps}\n")

    total_episodes = 0
    processed_episodes = 0
    skipped_episodes = 0
    error_episodes = 0

    for outcome in ["success", "failure"]:
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing outcome: {outcome.upper()}")
        print(f"{'=' * 60}\n")

        # Find all episodes with counterfactual results
        cf_json_files = list(outcome_dir.rglob("h2_1_counterfactual_results.json"))
        print(f"Found {len(cf_json_files)} episodes with counterfactual results")

        for cf_json_path in cf_json_files:
            episode_dir = cf_json_path.parent
            episode_id = episode_dir.name
            total_episodes += 1

            print(f"\n[{total_episodes}] Episode: {episode_id}")

            # Check if videos already exist
            video_dir = episode_dir / "videos_counterfactual"
            if video_dir.exists() and list(video_dir.glob("*.webm")):
                existing_videos = list(video_dir.glob("*.webm"))
                print(f"  ✓ Videos already exist ({len(existing_videos)} files), skipping")
                skipped_episodes += 1
                continue

            # Check if counterfactual directory exists
            cf_dir = episode_dir / "counterfactual"
            if not cf_dir.exists():
                print(f"  ✗ Counterfactual directory not found, skipping")
                error_episodes += 1
                continue

            try:
                # Generate videos
                print(f"  Generating counterfactual videos...")
                video_paths = create_all_counterfactual_videos(episode_dir, layers=layers, fps=fps, add_text=True)

                if video_paths:
                    print(f"  ✓ Created {len(video_paths)} videos")
                    processed_episodes += 1
                else:
                    print(f"  ✗ No videos created")
                    error_episodes += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_episodes += 1
                import traceback

                traceback.print_exc()

    # Final summary
    print(f"\n{'=' * 60}")
    print("BATCH VIDEO GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes found: {total_episodes}")
    print(f"Successfully processed: {processed_episodes}")
    print(f"Skipped (already done): {skipped_episodes}")
    print(f"Errors: {error_episodes}")

    return {
        "total": total_episodes,
        "processed": processed_episodes,
        "skipped": skipped_episodes,
        "errors": error_episodes,
    }


@timer
def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python viz/object_pipeline.py <DATA_ROOT>")
    #     print("Example: python viz/object_pipeline.py /data3/tonyw/aawr_offline/dual/")
    #     sys.exit(1)
    # TEST_CASE = "/data3/tonyw/aawr_offline/gold/"
    # TEST_CASE = "/data3/tonyw/aawr_offline/dual/"
    TEST_CASE = "/data3/tonyw/aawr_offline/bookshelf_d/"
    DATA_ROOT = Path(TEST_CASE)
    RESULTS_ROOT = Path("/data3/tonyw/aawr_offline/pi05/") / DATA_ROOT.name / CAMERA

    print(f"Data root: {DATA_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Camera: {CAMERA}")
    print(f"Layers: {LAYERS}")
    print()

    # Load Policy
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    print(f"Loading policy from {checkpoint_dir}...")
    policy = attn_map.get_policy(checkpoint_dir)
    print("Policy loaded.\n")

    # Statistics
    total_episodes = 0
    processed_episodes = 0
    skipped_episodes = 0
    error_episodes = 0

    for outcome in ["success", "failure"]:
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            print(f"Skipping outcome: {outcome} (directory not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing outcome: {outcome.upper()}")
        print(f"{'=' * 60}\n")

        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for h5_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = h5_path.parent
                episode_id = data_dir.name
                total_episodes += 1

                print(f"\n[{total_episodes}] Episode: {outcome}/{date_dir.name}/{episode_id}")

                # Setup Output Dir: results_object/{dataset_name}/{camera}/{outcome}/{date}/{episode}
                rel_path = data_dir.relative_to(DATA_ROOT)
                episode_dir = RESULTS_ROOT / rel_path
                episode_dir.mkdir(parents=True, exist_ok=True)

                # Skip if already processed
                marker_file = episode_dir / "object_detection.md"
                # if marker_file.exists():
                #     print(f"  ✓ Already processed, skipping")
                #     skipped_episodes += 1
                #     continue

                # Check if object masks exist
                white_list = get_white_list_frames(data_dir)
                if not white_list:
                    print(f"  ✗ No white list found, skipping")
                    skipped_episodes += 1
                    continue

                print(f"  White list: {len(white_list)} frames with object detections")

                # Determine total frames
                total_frames = get_video_length(data_dir)
                if total_frames == 0:
                    print(f"  ✗ No video frames found, skipping")
                    skipped_episodes += 1
                    continue

                # Use white list frames (they already have object detections)
                keyframes = white_list
                print(f"  Processing {len(keyframes)} keyframes: {keyframes[:5]}{'...' if len(keyframes) > 5 else ''}")

                all_results = {}
                successful_frames = 0

                # Storage for counterfactual results
                all_cf_results = {}

                # Process Keyframes
                for frame_idx in keyframes:
                    try:
                        print(f"\n  Frame {frame_idx}:")

                        # Load example
                        example = load_toy_example(data_dir, frame_idx, camera=CAMERA)

                        # ============================================================
                        # OBJECT DETECTION ANALYSIS
                        # ============================================================
                        if ENABLE_OBJECT_DETECTION:
                            # Run inference to generate attention maps
                            print(f"    Running inference (object detection)...")
                            _ = policy.infer(example)

                            # Run object detection analysis
                            print(f"    Analyzing attention-object correlation...")
                            frame_results = run_object_detection(
                                policy,
                                example,
                                frame_idx,
                                str(data_dir),
                                episode_dir,
                                layers=LAYERS,
                                camera=CAMERA,
                            )

                            if frame_results:
                                all_results[frame_idx] = frame_results
                                successful_frames += 1

                                # Print brief summary
                                for layer_key, layer_data in frame_results.items():
                                    if "aggregate_metrics" in layer_data:
                                        layer_idx = layer_key.split("_")[1]
                                        metrics_list = list(layer_data["aggregate_metrics"].values())
                                        if metrics_list:
                                            avg_overlap = np.mean([m["overlap_ratio"] for m in metrics_list])
                                            print(f"      Layer {layer_idx}: Overlap={avg_overlap:.3f}")

                        # ============================================================
                        # COUNTERFACTUAL PROMPT ANALYSIS
                        # ============================================================
                        if ENABLE_COUNTERFACTUAL:
                            print(f"    Running counterfactual prompt analysis...")

                            # Extract object name from original prompt
                            # Assume prompt format: "find the <object> and pick it up"
                            original_prompt = example.get("prompt", "")
                            object_name = "object"  # Default
                            if "pineapple" in original_prompt.lower():
                                object_name = "pineapple"
                            elif "duck" in original_prompt.lower():
                                object_name = "duck toy"
                            elif "banana" in original_prompt.lower():
                                object_name = "banana"
                            # Add more object detection logic as needed

                            cf_results = run_counterfactual_analysis(
                                policy,
                                example,
                                frame_idx,
                                episode_dir,
                                layers=COUNTERFACTUAL_LAYERS,
                                prompts=COUNTERFACTUAL_PROMPTS,
                                baseline_key="baseline",
                                camera=ANALYSIS_CAMERA,
                                object_name=object_name,
                            )

                            if cf_results:
                                all_cf_results[frame_idx] = cf_results
                                print(f"      ✓ Counterfactual analysis complete ({len(cf_results)} layers)")

                    except FileNotFoundError as e:
                        print(f"    ✗ File not found: {e}")
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        import traceback

                        traceback.print_exc()

                # Generate summary if we have results
                if all_results or all_cf_results:
                    print(f"\n  Generating summaries...")

                    # ============================================================
                    # OBJECT DETECTION SUMMARY
                    # ============================================================
                    if all_results and ENABLE_OBJECT_DETECTION:
                        print(f"    Object detection: {len(all_results)} frames")

                        # Generate summary plot
                        summary_plot_path = episode_dir / "h1_1_obj_attn_summary.png"
                        layer_stats = generate_summary_plot(all_results, str(summary_plot_path), LAYERS)

                        # Save JSON results
                        results_json_path = episode_dir / "h1_1_obj_attn_results.json"

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

                        with open(results_json_path, "w") as f:
                            json.dump(
                                {
                                    "frame_results": convert_to_native(all_results),
                                    "layer_statistics": convert_to_native(layer_stats),
                                },
                                f,
                                indent=2,
                            )

                        print(f"      ✓ Summary plot: {summary_plot_path.name}")
                        print(f"      ✓ JSON results: {results_json_path.name}")

                        # Generate videos
                        print(f"    Generating object detection videos...")
                        video_paths = create_all_layer_videos(episode_dir, layers=LAYERS, fps=FPS_VIDEO, add_text=True)
                        if video_paths:
                            print(f"      ✓ Created {len(video_paths)} videos")

                    # ============================================================
                    # COUNTERFACTUAL SUMMARY
                    # ============================================================
                    if all_cf_results and ENABLE_COUNTERFACTUAL:
                        print(f"    Counterfactual: {len(all_cf_results)} frames")

                        # Save counterfactual results
                        cf_json_path = episode_dir / "h2_1_counterfactual_results.json"

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

                        with open(cf_json_path, "w") as f:
                            json.dump(
                                {
                                    "frame_results": convert_to_native(all_cf_results),
                                    "prompts": COUNTERFACTUAL_PROMPTS,
                                    "layers": list(COUNTERFACTUAL_LAYERS),
                                    "camera": ANALYSIS_CAMERA,
                                },
                                f,
                                indent=2,
                            )

                        print(f"      ✓ JSON results: {cf_json_path.name}")

                        # Generate counterfactual videos
                        print(f"    Generating counterfactual videos...")
                        cf_video_paths = create_all_counterfactual_videos(
                            episode_dir, layers=list(COUNTERFACTUAL_LAYERS), fps=FPS_VIDEO, add_text=True
                        )
                        if cf_video_paths:
                            print(f"      ✓ Created {len(cf_video_paths)} videos")

                        # Generate counterfactual report
                        cf_report_path = episode_dir / "h2_1_counterfactual_report.md"
                        with open(cf_report_path, "w") as f:
                            f.write("# Counterfactual Prompt Analysis Report\n\n")
                            f.write(f"**Episode**: {episode_id}\n")
                            f.write(f"**Outcome**: {outcome}\n")
                            f.write(f"**Camera**: {ANALYSIS_CAMERA}\n")
                            f.write(f"**Frames Analyzed**: {len(all_cf_results)}\n\n")

                            f.write("## Prompts Tested\n\n")
                            for key, text in COUNTERFACTUAL_PROMPTS.items():
                                marker = " (baseline)" if key == "baseline" else ""
                                f.write(f'- **{key}**{marker}: "{text}"\n')

                            f.write("\n## Results Summary\n\n")
                            for frame_idx, frame_data in all_cf_results.items():
                                f.write(f"### Frame {frame_idx}\n\n")
                                for layer, stats in frame_data.items():
                                    f.write(f"#### Layer {layer}\n\n")
                                    f.write("| Prompt | Mean Δ | |Δ| | L2 Distance | Correlation |\n")
                                    f.write("|--------|---------|------|-------------|-------------|\n")
                                    for key, stat in stats.items():
                                        f.write(
                                            f"| {key} | {stat['mean_diff']:+.4f} | {stat['abs_mean_diff']:.4f} | "
                                            f"{stat['l2_distance']:.4f} | {stat['correlation']:.4f} |\n"
                                        )
                                    f.write("\n")

                        print(f"      ✓ Report: {cf_report_path.name}")

                    # Copy instruction
                    copy_instruction(data_dir, episode_dir)

                    # Create marker file
                    marker_content = f"# Analysis Complete\n\n"
                    marker_content += f"Episode: {episode_id}\n"
                    marker_content += f"Outcome: {outcome}\n"
                    marker_content += f"Date: {date_dir.name}\n"
                    marker_content += f"Total Frames: {total_frames}\n"
                    marker_content += f"Keyframes: {keyframes}\n\n"

                    if ENABLE_OBJECT_DETECTION:
                        marker_content += f"## Object Detection\n"
                        marker_content += f"- Frames Processed: {successful_frames}/{len(keyframes)}\n"
                        marker_content += f"- Layers: {list(LAYERS)}\n"
                        marker_content += f"- Camera: {CAMERA}\n\n"

                    if ENABLE_COUNTERFACTUAL:
                        marker_content += f"## Counterfactual Analysis\n"
                        marker_content += f"- Frames Processed: {len(all_cf_results)}\n"
                        marker_content += f"- Layers: {list(COUNTERFACTUAL_LAYERS)}\n"
                        marker_content += f"- Camera: {ANALYSIS_CAMERA}\n"
                        marker_content += f"- Prompts: {list(COUNTERFACTUAL_PROMPTS.keys())}\n"

                    marker_file.write_text(marker_content)

                    processed_episodes += 1
                    print(f"  ✓ Episode processed successfully")
                else:
                    print(f"  ✗ No results generated")
                    error_episodes += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes found: {total_episodes}")
    print(f"Successfully processed: {processed_episodes}")
    print(f"Skipped (already done): {skipped_episodes}")
    print(f"Errors: {error_episodes}")
    print(f"\nResults saved to: {RESULTS_ROOT}")
    print(f"\nAnalysis Modes:")
    print(f"  - Object Detection: {'✓ Enabled' if ENABLE_OBJECT_DETECTION else '✗ Disabled'}")
    print(f"  - Counterfactual Prompt: {'✓ Enabled' if ENABLE_COUNTERFACTUAL else '✗ Disabled'}")

    # Generate multi-episode aggregated analysis
    if processed_episodes > 0 or skipped_episodes > 0:
        # Object Detection aggregation
        if ENABLE_OBJECT_DETECTION:
            print("\n" + "=" * 60)
            print("Generating multi-episode aggregated analysis (Object Detection)...")
            print("=" * 60)
            aggregate_episodes_analysis(RESULTS_ROOT, LAYERS, output_dir=RESULTS_ROOT)

        # Counterfactual aggregation
        if ENABLE_COUNTERFACTUAL:
            print("\n" + "=" * 60)
            print("Generating multi-episode aggregated analysis (Counterfactual)...")
            print("=" * 60)
            aggregate_counterfactual_analysis(
                RESULTS_ROOT, COUNTERFACTUAL_LAYERS, COUNTERFACTUAL_PROMPTS, output_dir=RESULTS_ROOT
            )


if __name__ == "__main__":
    main()
