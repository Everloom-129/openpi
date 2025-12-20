import dataclasses
import os
from PIL import Image
import cv2

import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import matplotlib.pyplot as plt


def load_duck_example(camera: str = "left", index: int = 0):
    """Load example from duck dataset."""

    if camera == "left":
        camera = "varied_camera_1"
    elif camera == "right":
        camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    if not 0 <= index <= 90:
        raise ValueError("index must be between 0 and 90")

    data_dir = "data/visualization/duck/frames"

    # Load images
    ext_path = os.path.join(data_dir, camera, f"{index:05d}.jpg")
    hand_path = os.path.join(data_dir, "hand_camera", f"{index:05d}.jpg")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": np.random.rand(7),  # Placeholder
        "observation/gripper_position": np.random.rand(1),  # Placeholder
        "prompt": "place the duck toy into the pink bowl",
    }


def vis_example(example: dict, name: str):
    os.makedirs(f"results/{name}", exist_ok=True)
    fig = plt.figure(figsize=(12, 8))

    plt.suptitle(example["prompt"], fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(example["observation/exterior_image_1_left"])
    plt.title("Exterior Camera")

    plt.subplot(2, 2, 2)
    plt.imshow(example["observation/wrist_image_left"])
    plt.title("Wrist Camera")

    plt.subplot(2, 2, 3)
    plt.bar(range(len(example["observation/joint_position"])), example["observation/joint_position"])
    plt.title("Joint Positions")

    plt.subplot(2, 2, 4)
    plt.bar(["Gripper"], example["observation/gripper_position"])
    plt.title("Gripper Position")

    plt.tight_layout()
    fig_path = f"results/{name}/origin_{example['prompt']}.jpg"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved example to {fig_path}")


def visualize_attention(example: dict, name: str, attn_mode: str, mode: str = "avg", layer_idx: int = 20):
    """Visualizes the attention map overlaid on the images."""
    attn_path = f"results/layers_{attn_mode}/attn_map_layer_{layer_idx}.npy"
    if not os.path.exists(attn_path):
        print(f"Warning: Attention file {attn_path} not found. Skipping visualization.")
        return

    # Load Attention Map: [Batch, NumHeads, SeqLen, SeqLen]
    # We take the first Batch and average over Heads
    attn_map = np.load(attn_path)
    if attn_map.ndim == 4:
        attn_avg = attn_map[0].mean(axis=0)  # [SeqLen, SeqLen]
    elif attn_map.ndim == 3:
        attn_avg = attn_map.mean(axis=0)  # [SeqLen, SeqLen]
    else:
        attn_avg = attn_map

    print(f"Loaded attention map shape: {attn_map.shape}, averaged shape: {attn_avg.shape}")

    # Pi05 DROID models usually have two images: exterior_image_1_left and wrist_image_left
    # Each image 224x224, Patch Size 14 -> 16x16 = 256 tokens
    num_image_tokens = 256
    num_images = 2
    total_image_tokens = num_image_tokens * num_images  # 512

    # Determine Text Token position
    # PaliGemma Structure: [Image1, Image2, ..., Text...]
    # We take the Attention of Text Tokens (usually starting after images) to all Image Tokens
    text_start_idx = total_image_tokens

    # Logic to handle both Prefix (Square matrix) and Suffix (Rectangular matrix) attention
    # Prefix: [SeqLen, SeqLen] -> Text tokens are at the end, attending to Image tokens at start
    # Suffix: [ActionLen, TotalLen] -> Action tokens (rows) attending to Image tokens (cols at start)

    if attn_avg.shape[0] < total_image_tokens:
        # Suffix Case: Query length is short (actions), Key length is long (includes images)
        print(f"Visualizing Suffix Attention (Action -> Image). Shape: {attn_avg.shape}")
        # Aggregate attention from all action tokens (rows) to image tokens (cols)
        if mode == "avg":
            text_attn_to_image = attn_avg[:, :total_image_tokens].mean(axis=0)
        elif mode == "max":
            text_attn_to_image = attn_avg[:, :total_image_tokens].max(axis=0)
    else:
        # Prefix Case
        if attn_avg.shape[0] <= text_start_idx:
            print("Error: Attention map sequence length is smaller than expected image tokens.")
            return

        # Get attention from Text Tokens to Image regions
        if mode == "avg":
            text_attn_to_image = attn_avg[text_start_idx:, :total_image_tokens].mean(axis=0)
        elif mode == "max":
            text_attn_to_image = attn_avg[text_start_idx:, :total_image_tokens].max(axis=0)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # Split for two images
    img1_attn = text_attn_to_image[:num_image_tokens].reshape(16, 16)
    img2_attn = text_attn_to_image[num_image_tokens:].reshape(16, 16)

    # Helper for overlay
    def overlay_heatmap(img, heatmap):
        from openpi.shared import image_tools
        import jax.numpy as jnp

        # 1. Resize image to 224x224 with pad (Simulating model preprocessing)
        img_jax = jnp.array(img)
        img_resized = image_tools.resize_with_pad(img_jax, 224, 224)
        img_resized = np.array(img_resized).astype(np.uint8)  # Ensure uint8 for opencv

        # 2. Resize heatmap to 224x224 (matching the resized image)
        # heatmap is 16x16
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Normalize to 0-255
        heatmap_norm = np.uint8(255 * heatmap_resized / (np.max(heatmap_resized) + 1e-8))
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        # Overlay
        # img is RGB, cv2 uses BGR for some ops, but applyColorMap returns BGR
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        return img_resized, cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

    img1_resized, img1_overlay = overlay_heatmap(example["observation/exterior_image_1_left"], img1_attn)
    img2_resized, img2_overlay = overlay_heatmap(example["observation/wrist_image_left"], img2_attn)

    # Plot - Compact Layout
    # 2 Rows x 2 Cols
    # Row 1: Original Images (Resized)
    # Row 2: Attention Heatmaps
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"wspace": 0.05, "hspace": 0.1})

    # Set Main Title
    plt.suptitle(f'Attention ({attn_mode} L{layer_idx} {mode}) for: "{example["prompt"]}"', fontsize=16, y=0.98)

    # Row 1: Originals
    axs[0, 0].imshow(img1_resized)
    axs[0, 0].set_title("Exterior Camera", fontsize=12)
    axs[0, 1].imshow(img2_resized)
    axs[0, 1].set_title("Wrist Camera", fontsize=12)

    # Row 2: Attention Overlays
    axs[1, 0].imshow(img1_overlay)
    axs[1, 0].set_title("Exterior Attention", fontsize=12)
    axs[1, 1].imshow(img2_overlay)
    axs[1, 1].set_title("Wrist Attention", fontsize=12)

    # Remove axes for all subplots
    for ax in axs.flatten():
        ax.axis("off")

    out_path = f"results/{name}/{attn_mode}_L{layer_idx}_attn_vis_{mode}.jpg"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved attention visualization to {out_path}")


if __name__ == "__main__":
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"

    # Create a trained policy (automatically detects PyTorch format)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    camera = "left"
    for index in [0, 30, 60, 85, 90]:
        print(f"VLA index {index}")
        example = load_duck_example(camera=camera, index=index)
        # vis_example(example, f"duck_{camera}_{index}")

        # 1. Run Inference (this will save the .npy files)
        result = policy.infer(example)
        print("Actions shape:", result["actions"].shape)

        # 2. Visualize Attention using the saved .npy
        # Try multiple layers to find the best one
        for layer in range(18):
            for mode in ["avg", "max"]:
                visualize_attention(example, f"duck_{camera}_{index}", attn_mode="prefix", mode=mode, layer_idx=layer)
                visualize_attention(example, f"duck_{camera}_{index}", attn_mode="suffix", mode=mode, layer_idx=layer)

    del policy
