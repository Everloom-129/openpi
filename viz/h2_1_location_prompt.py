"""
Hypothesis 2.1: Text Prompt Controls Visual Attention (Cross-Modal Grounding)

Tests whether changing the object name in the text prompt shifts attention focus.
Uses counterfactual prompts to verify semantic grounding capability.

Experiment Design:
1. Same visual scene (pineapple on table)
2. Multiple prompts: "pick up the pineapple", "pick up the duck", "pick up the banana", etc.
3. Compare attention maps across different prompts
4. Expected: Attention should shift based on prompt, even if object doesn't exist
"""

import json
import os
from pathlib import Path

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.shared import image_tools
from openpi.training import config as _config


def load_example_with_prompt(camera: str = "left", index: int = 0, prompt: str = None):
    """Load example with custom prompt"""

    if camera == "left":
        camera = "varied_camera_1"
    elif camera == "right":
        camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

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

    # Use provided prompt or default
    if prompt is None:
        prompt = "find the pineapple toy and pick it up"

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": prompt,
    }


def extract_attention_map(policy, example, layer: int, camera: str = "wrist"):
    """
    Extract attention map for a specific layer and camera.

    Args:
        policy: Trained policy
        example: Input example
        layer: Layer index
        camera: 'wrist' or 'exterior'

    Returns:
        Attention map (16x16) or None if not found
    """
    # Run inference to generate attention maps
    _ = policy.infer(example)

    # Load attention map
    attn_path = Path("results") / "layers_prefix" / f"attn_map_layer_{layer}.npy"
    if not attn_path.exists():
        print(f"Warning: Attention map not found for Layer {layer}")
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
        print(f"Warning: Not prefix attention for Layer {layer}")
        return None

    # Max over text tokens
    text_attn = attn_avg[total_img:, :total_img].max(axis=0)

    # Select camera
    if camera == "wrist":
        attn_cam = text_attn[num_img:total_img].reshape(16, 16)
    else:  # exterior
        attn_cam = text_attn[:num_img].reshape(16, 16)

    return attn_cam


def visualize_prompt_comparison(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    camera: str = "wrist",
):
    """
    Visualize attention maps for different prompts side-by-side.

    Args:
        attention_maps: Dict mapping prompt_key -> attention_map (16x16)
        prompts: Dict mapping prompt_key -> prompt_text
        reference_image: Original image
        output_path: Save path
        layer: Layer index
        camera: Camera name
    """
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
    print(f"Saved comparison: {output_path}")


def visualize_difference_maps(
    attention_maps: dict,
    prompts: dict,
    reference_image: np.ndarray,
    output_path: str,
    layer: int,
    baseline_key: str,
    camera: str = "wrist",
):
    """
    Visualize difference maps: M_counterfactual - M_baseline.

    Positive values: More attention with counterfactual prompt
    Negative values: Less attention with counterfactual prompt
    """
    if baseline_key not in attention_maps:
        print(f"Warning: Baseline key '{baseline_key}' not found")
        return

    baseline_attn = attention_maps[baseline_key]
    counterfactual_keys = [k for k in attention_maps.keys() if k != baseline_key]

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
        # Positive (red): More attention with counterfactual
        # Negative (blue): Less attention with counterfactual
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
    print(f"Saved difference maps: {output_path}")


def compute_attention_statistics(attention_maps: dict, prompts: dict, baseline_key: str):
    """
    Compute statistics for attention shift analysis.

    Returns:
        Dictionary with metrics per counterfactual prompt
    """
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


def main():
    # 1. Config & Policy
    print("Loading Policy...")
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 2. Configuration
    FRAME_IDX = 50  # Frame with clear pineapple
    CAMERA_VIEW = "right"  # Camera for loading images
    ANALYSIS_CAMERA = "wrist"  # Which camera's attention to analyze
    LAYERS = [5, 10, 17]  # Key layers to analyze

    # 3. Define prompts (counterfactual)
    PROMPTS = {
        "pineapple": "find the pineapple toy and pick it up",
        "duck": "find the duck toy and pick it up",
        "banana": "find the banana and pick it up",
        "cat": "find the cat toy and pick it up",
        "bottle": "find the bottle and pick it up",
    }
    BASELINE_KEY = "pineapple"  # Ground truth object

    # 4. Output directory
    output_dir = Path("results/h2_1_prompt_control")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Hypothesis 2.1: Text Prompt Controls Visual Attention")
    print(f"{'=' * 60}")
    print(f"Frame: {FRAME_IDX}")
    print(f"Camera: {ANALYSIS_CAMERA}")
    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Baseline: {BASELINE_KEY}")
    print()

    # 5. Load reference image
    example_ref = load_example_with_prompt(camera=CAMERA_VIEW, index=FRAME_IDX, prompt=PROMPTS[BASELINE_KEY])
    if ANALYSIS_CAMERA == "wrist":
        reference_image = example_ref["observation/wrist_image_left"]
    else:
        reference_image = example_ref["observation/exterior_image_1_left"]

    # Save reference image
    cv2.imwrite(
        str(output_dir / f"frame_{FRAME_IDX:05d}_{ANALYSIS_CAMERA}_reference.jpg"),
        cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR),
    )

    # 6. Process each layer
    all_layer_results = {}

    for layer in LAYERS:
        print(f"\n{'=' * 60}")
        print(f"Processing Layer {layer}")
        print(f"{'=' * 60}")

        attention_maps = {}

        # Extract attention for each prompt
        for prompt_key, prompt_text in PROMPTS.items():
            print(f"  Prompt: '{prompt_text}'")

            # Load example with this prompt
            example = load_example_with_prompt(camera=CAMERA_VIEW, index=FRAME_IDX, prompt=prompt_text)

            # Extract attention
            attn_map = extract_attention_map(policy, example, layer, camera=ANALYSIS_CAMERA)

            if attn_map is not None:
                attention_maps[prompt_key] = attn_map
                print(f"    ✓ Extracted attention (mean: {attn_map.mean():.4f}, max: {attn_map.max():.4f})")
            else:
                print(f"    ✗ Failed to extract attention")

        if not attention_maps:
            print(f"  No attention maps extracted for Layer {layer}, skipping")
            continue

        # Visualize comparison
        comparison_path = output_dir / f"L{layer:02d}_prompt_comparison_{ANALYSIS_CAMERA}.png"
        visualize_prompt_comparison(
            attention_maps, PROMPTS, reference_image, str(comparison_path), layer, ANALYSIS_CAMERA
        )

        # Visualize difference maps
        if BASELINE_KEY in attention_maps and len(attention_maps) > 1:
            diff_path = output_dir / f"L{layer:02d}_difference_maps_{ANALYSIS_CAMERA}.png"
            visualize_difference_maps(
                attention_maps, PROMPTS, reference_image, str(diff_path), layer, BASELINE_KEY, ANALYSIS_CAMERA
            )

            # Compute statistics
            stats = compute_attention_statistics(attention_maps, PROMPTS, BASELINE_KEY)
            all_layer_results[layer] = stats

            print(f"\n  Statistics (vs baseline '{PROMPTS[BASELINE_KEY]}'):")
            for key, stat in stats.items():
                print(f"    {key}:")
                print(f"      Mean Δ: {stat['mean_diff']:+.4f}")
                print(f"      |Δ|: {stat['abs_mean_diff']:.4f}")
                print(f"      L2 distance: {stat['l2_distance']:.4f}")
                print(f"      Correlation: {stat['correlation']:.4f}")

    # 7. Save summary report
    report_path = output_dir / "h2_1_report.md"
    with open(report_path, "w") as f:
        f.write("# Hypothesis 2.1: Text Prompt Controls Visual Attention\n\n")
        f.write(f"**Frame**: {FRAME_IDX}\n")
        f.write(f"**Camera**: {ANALYSIS_CAMERA}\n")
        f.write(f'**Baseline Prompt**: "{PROMPTS[BASELINE_KEY]}"\n\n')

        f.write("## Prompts Tested\n\n")
        for key, text in PROMPTS.items():
            marker = " (baseline)" if key == BASELINE_KEY else ""
            f.write(f'- **{key}**{marker}: "{text}"\n')

        f.write("\n## Results by Layer\n\n")
        for layer, stats in all_layer_results.items():
            f.write(f"### Layer {layer}\n\n")
            f.write("| Prompt | Mean Δ | |Δ| | L2 Distance | Correlation |\n")
            f.write("|--------|---------|------|-------------|-------------|\n")
            for key, stat in stats.items():
                f.write(
                    f"| {key} | {stat['mean_diff']:+.4f} | {stat['abs_mean_diff']:.4f} | "
                    f"{stat['l2_distance']:.4f} | {stat['correlation']:.4f} |\n"
                )
            f.write("\n")

        f.write("## Interpretation\n\n")
        f.write("- **Mean Δ > 0**: Counterfactual prompt increases overall attention\n")
        f.write("- **|Δ|**: Average absolute difference (sensitivity to prompt change)\n")
        f.write("- **L2 Distance**: Euclidean distance between attention maps\n")
        f.write("- **Correlation**: Spatial similarity (1.0 = identical, 0.0 = uncorrelated)\n\n")
        f.write("**Expected**: Low correlation and high L2 distance indicate strong prompt control.\n")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Visualizations: {len(LAYERS) * 2} images generated")


if __name__ == "__main__":
    main()
