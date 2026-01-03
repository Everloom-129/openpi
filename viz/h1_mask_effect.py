"""
Hypothesis: Attention Causal Fidelity Test

This script tests whether attention maps have causal fidelity by masking regions
and measuring the impact on model outputs.

Hypothesis:
    If we mask (occlude) the high-attention regions identified by the Attention Map,
    the model's output Actions should deviate significantly (i.e., error increases).
    Conversely, masking low-attention regions should cause minimal change in Actions.

Experimental Design (Input Perturbation):
    1. Baseline: Record the original Action A_orig generated from the original image I.

    2. Mask Relevant: Based on the extracted Attention Map, generate a binary mask
       that blacks out or adds Gaussian noise to high-attention regions (e.g., top 10%
       of pixels), then feed to the model to get A_mask_high.

    3. Mask Irrelevant: Randomly mask an equal area of low-attention regions,
       feed to the model to get A_mask_low.

    4. Metric: Calculate the difference in Actions (e.g., MSE or Trajectory Distance).

       Fidelity Score = MSE(A_orig, A_mask_high) - MSE(A_orig, A_mask_low)

Expected Result:
    Fidelity Score should be significantly greater than 0, indicating that
    high-attention regions are causally important for the model's predictions.
"""

import cv2
import numpy as np
from pathlib import Path

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

from attn_map import load_duck_example, select_best_gpu


def map_mask_to_raw(mask_224, raw_shape):
    """Maps a 224x224 binary mask back to the raw image coordinate space."""
    raw_h, raw_w = raw_shape[:2]
    target_h, target_w = 224, 224

    # Re-calculate padding parameters used in resize_with_pad
    # Note: openpi uses: ratio = max(cur_width / width, cur_height / height)
    # resized_height = int(cur_height / ratio)
    scale = min(target_w / raw_w, target_h / raw_h)
    new_w = int(raw_w * scale)
    new_h = int(raw_h * scale)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    # 1. Crop valid region from 224 mask
    # Valid region: [pad_h : pad_h+new_h, pad_w : pad_w+new_w]
    # We need to handle potential off-by-one errors in rounding carefully,
    # but slicing handles bounds gracefully.
    mask_cropped = mask_224[pad_h : pad_h + new_h, pad_w : pad_w + new_w]

    if mask_cropped.size == 0:
        return np.zeros((raw_h, raw_w), dtype=bool)

    # 2. Resize back to raw
    mask_raw = cv2.resize(mask_cropped.astype(np.uint8), (raw_w, raw_h), interpolation=cv2.INTER_NEAREST)

    return mask_raw > 0


def create_masked_example(example, attn_ext, attn_wrist, mask_type="high", percentile=90):
    """
    Creates a new example dict with images masked based on attention.
    mask_type: "high" (mask top N%) or "low" (mask random N% from low attention areas)
    """
    ex_new = example.copy()

    def process_image(img_key, attn_map_16x16):
        img_raw = ex_new[img_key]

        # Upsample attention to 224x224
        attn_224 = cv2.resize(attn_map_16x16, (224, 224), interpolation=cv2.INTER_CUBIC)

        threshold = np.percentile(attn_224, percentile)
        high_mask_224 = attn_224 > threshold

        if mask_type == "high":
            final_mask_224 = high_mask_224
        else:  # mask_type == "low"
            # Strategy: Randomly mask an EQUAL AREA of low-attention regions
            area_pixels = np.sum(high_mask_224)

            # Low candidates: bottom 50%
            median_val = np.percentile(attn_224, 50)
            low_candidates_mask = attn_224 < median_val

            flat_indices = np.flatnonzero(low_candidates_mask)

            final_mask_224 = np.zeros_like(attn_224, dtype=bool)
            if len(flat_indices) >= area_pixels:
                selected_indices = np.random.choice(flat_indices, int(area_pixels), replace=False)
                np.put(final_mask_224, selected_indices, True)
            else:
                # If not enough low pixels (unlikely), just take all of them
                final_mask_224 = low_candidates_mask

        # Map mask to raw image
        mask_raw = map_mask_to_raw(final_mask_224, img_raw.shape)

        # Apply mask (black out)
        img_masked = img_raw.copy()
        img_masked[mask_raw] = 0  # Set to black

        return img_masked, mask_raw

    # Process Exterior
    img_ext_masked, mask_ext = process_image("observation/exterior_image_1_left", attn_ext)
    ex_new["observation/exterior_image_1_left"] = img_ext_masked

    # Process Wrist
    img_wrist_masked, mask_wrist = process_image("observation/wrist_image_left", attn_wrist)
    ex_new["observation/wrist_image_left"] = img_wrist_masked

    return ex_new, mask_ext, mask_wrist


def run_fidelity_test(
    policy,
    example,
    action_orig,
    output_dir,
    layers=None,
    attn_root_dir="results",
    mask_percentile=90,
    save_vis=True,
):
    """
    Runs the fidelity test for specified layers on a single example.

    Args:
        policy: The loaded policy object.
        example: The input example dict.
        action_orig: The original action prediction (baseline).
        output_dir: Directory to save results/images.
        layers: List of layer indices to test. If None, uses range(18).
        attn_root_dir: Root directory where attention maps are stored.
        mask_percentile: Percentile for high-attention masking.
        save_vis: Whether to save visualization images.

    Returns:
        List of results dicts.
    """
    if layers is None:
        layers = range(18)

    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ATTN_MODE = "prefix"
    # Assuming standard structure: attn_root_dir/layers_prefix/attn_map_layer_X.npy
    # If attn_root_dir points directly to where files are, adjust accordingly.
    # But usually attn_map.py writes to "results/layers_prefix".

    # We'll assume attn_root_dir is the base "results" folder unless it ends with "layers_prefix"
    attn_base = Path(attn_root_dir)
    if not str(attn_base).endswith(f"layers_{ATTN_MODE}"):
        attn_base = attn_base / f"layers_{ATTN_MODE}"

    for layer_idx in layers:
        print(f"Processing Fidelity Layer {layer_idx}...")

        attn_path = attn_base / f"attn_map_layer_{layer_idx}.npy"

        if not attn_path.exists():
            print(f"  Warning: Attention map not found for Layer {layer_idx} at {attn_path}. Skipping.")
            continue

        attn_map = np.load(attn_path)  # [Batch, Heads, Seq, Seq]
        if attn_map.ndim == 4:
            attn_avg = attn_map[0].max(axis=0)  # Use MAX across heads to capture peaks
        else:
            attn_avg = attn_map

        # Extract Image Attention (Text -> Image)
        # Tokens 0-255: Ext Image, 256-511: Wrist Image, 512+: Text
        num_img = 256
        total_img = 512
        text_attn = attn_avg[total_img:, :total_img].max(axis=0)  # Max over text tokens

        attn_ext = text_attn[:num_img].reshape(16, 16)
        attn_wrist = text_attn[num_img:].reshape(16, 16)

        # Generate Masked Examples
        ex_high, _, _ = create_masked_example(
            example, attn_ext, attn_wrist, mask_type="high", percentile=mask_percentile
        )
        ex_low, _, _ = create_masked_example(example, attn_ext, attn_wrist, mask_type="low", percentile=mask_percentile)

        # Save visualization
        if save_vis:
            layer_out_dir = output_dir / f"L{layer_idx:02d}"
            layer_out_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(layer_out_dir / "mask_high.jpg"),
                cv2.cvtColor(ex_high["observation/exterior_image_1_left"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(layer_out_dir / "mask_low.jpg"),
                cv2.cvtColor(ex_low["observation/exterior_image_1_left"], cv2.COLOR_RGB2BGR),
            )

        # Inference
        result_high = policy.infer(ex_high)
        action_high = result_high["actions"]

        result_low = policy.infer(ex_low)
        action_low = result_low["actions"]

        # Metrics
        mse_high = np.mean((action_orig - action_high) ** 2)
        mse_low = np.mean((action_orig - action_low) ** 2)
        fidelity_score = mse_high - mse_low

        res = {"layer": layer_idx, "mse_high": mse_high, "mse_low": mse_low, "fidelity": fidelity_score}
        results.append(res)
        print(f"  Layer {layer_idx}: Fidelity = {fidelity_score:.6f}")

    return results


def run_experiment_all_layers():
    # 1. Config & Policy
    print("Loading Policy...")
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 2. Load Data (Keyframe 0)
    KEYFRAME = 0
    CAMERA = "left"
    example = load_duck_example(camera=CAMERA, index=KEYFRAME)

    output_dir = Path("results/h1_fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    cv2.imwrite(
        str(output_dir / "sideview_orig.jpg"),
        cv2.cvtColor(example["observation/exterior_image_1_left"], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output_dir / "wristview_orig.jpg"),
        cv2.cvtColor(example["observation/wrist_image_left"], cv2.COLOR_RGB2BGR),
    )

    # 3. Baseline Inference
    print("Running Baseline Inference...")
    result_orig = policy.infer(example)
    action_orig = result_orig["actions"]

    # 4. Run Fidelity Test
    # Note: policy.infer above should have generated the attention maps in results/layers_prefix
    device_id = str(select_best_gpu())
    results = run_fidelity_test(
        policy,
        example,
        action_orig,
        output_dir=output_dir,
        attn_root_dir=f"attn/{device_id}/layers_prefix",
        save_vis=True,
    )

    # 5. Generate Report
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("# Hypothesis 1: Attention Causal Fidelity Report\n\n")
        f.write(
            "**Goal**: Verify if masking high-attention regions causes higher error than masking low-attention regions.\n\n"
        )
        f.write(f"**Metric**: `Fidelity = MSE_High_Mask - MSE_Low_Mask` (Higher/Positive is better)\n\n")

        # Best Layer
        if results:
            best_layer = max(results, key=lambda x: x["fidelity"])
            f.write(f"## 🏆 Best Layer: {best_layer['layer']} (Score: {best_layer['fidelity']:.6f})\n\n")

        # Table
        f.write("## Layer-wise Results\n\n")
        f.write("| Layer | Fidelity Score | MSE High (Occluded) | MSE Low (Control) | Result |\n")
        f.write("|-------|----------------|---------------------|-------------------|--------|\n")

        for res in results:
            indicator = "✅" if res["fidelity"] > 0 else "❌"
            f.write(
                f"| {res['layer']} | {res['fidelity']:.6f} | {res['mse_high']:.6f} | {res['mse_low']:.6f} | {indicator} |\n"
            )

        f.write("\n## Visualizations (Best Layer)\n\n")
        if results:
            L = best_layer["layer"]
            f.write(f"### Layer {L}\n")
            f.write(f"| Mask High (Target) | Mask Low (Control) |\n")
            f.write(f"|-------------------|--------------------|\n")
            f.write(
                f"| <img src='L{L:02d}/mask_high.jpg' width='300'> | <img src='L{L:02d}/mask_low.jpg' width='300'> |\n"
            )

    print(f"\nReport generated at {report_path}")


if __name__ == "__main__":
    run_experiment_all_layers()
