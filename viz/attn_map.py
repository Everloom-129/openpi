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


from openpi.shared import image_tools
import jax.numpy as jnp

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

    instruction = "place the duck toy into the pink bowl"
    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": np.random.rand(7),  # Placeholder # TODO read from real data
        "observation/gripper_position": np.random.rand(1),  # Placeholder
        "prompt": instruction,
    }


# TODO: refactor this, outdated now
def vis_example(example: dict, name: str, base_dir: str = "results"):
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
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
    fig_path = os.path.join(out_dir, f"origin_{example['prompt']}.jpg")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved example to {fig_path}")


def visualize_attention(
    example: dict,
    name: str,
    attn_mode: str,
    mode: str = "avg",
    layer_idx: int = 20,
    input_dir: str = "results",
    output_dir: str = "results",
):
    """Visualizes the attention map overlaid on the images (Summary View)."""

    attn_path = os.path.join(input_dir, "layers_" + attn_mode, f"attn_map_layer_{layer_idx}.npy")
    if not os.path.exists(attn_path):
        print(f"Warning: Attention file {attn_path} not found. Skipping visualization.")
        return

    # Load Attention Map: [Batch, NumHeads, SeqLen, SeqLen]
    attn_map = np.load(attn_path)
    if attn_map.ndim == 4:
        attn_avg = attn_map[0].mean(axis=0)  # [SeqLen, SeqLen]
    elif attn_map.ndim == 3:
        attn_avg = attn_map.mean(axis=0)  # [SeqLen, SeqLen]
    else:
        attn_avg = attn_map

    # Pi05 DROID models usually have two images: exterior_image_1_left and wrist_image_left
    # Each image 224x224, Patch Size 14 -> 16x16 = 256 tokens
    num_image_tokens = 256
    num_images = 2
    total_image_tokens = num_image_tokens * num_images  # 512

    text_start_idx = total_image_tokens

    if attn_avg.shape[0] < total_image_tokens:
        # Suffix Case
        if mode == "avg":
            text_attn_to_image = attn_avg[:, :total_image_tokens].mean(axis=0)
        elif mode == "max":
            text_attn_to_image = attn_avg[:, :total_image_tokens].max(axis=0)
    else:
        # Prefix Case
        if attn_avg.shape[0] <= text_start_idx:
            return

        if mode == "avg":
            text_attn_to_image = attn_avg[text_start_idx:, :total_image_tokens].mean(axis=0)
        elif mode == "max":
            text_attn_to_image = attn_avg[text_start_idx:, :total_image_tokens].max(axis=0)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    img1_attn = text_attn_to_image[:num_image_tokens].reshape(16, 16)
    img2_attn = text_attn_to_image[num_image_tokens:].reshape(16, 16)

    def overlay_heatmap(img, heatmap):
        img_jax = jnp.array(img)
        img_resized = image_tools.resize_with_pad(img_jax, 224, 224)
        img_resized = np.array(img_resized).astype(np.uint8)

        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

        heatmap_norm = np.uint8(255 * heatmap_resized / (np.max(heatmap_resized) + 1e-8))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        return img_resized, cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

    img1_resized, img1_overlay = overlay_heatmap(example["observation/exterior_image_1_left"], img1_attn)
    img2_resized, img2_overlay = overlay_heatmap(example["observation/wrist_image_left"], img2_attn)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"wspace": 0.05, "hspace": 0.1})

    plt.suptitle(f'Attention ({attn_mode} L{layer_idx} {mode}) for: "{example["prompt"]}"', fontsize=16, y=0.98)

    axs[0, 0].imshow(img1_resized)
    axs[0, 0].set_title("Exterior Camera", fontsize=12)
    axs[0, 1].imshow(img2_resized)
    axs[0, 1].set_title("Wrist Camera", fontsize=12)

    axs[1, 0].imshow(img1_overlay)
    axs[1, 0].set_title("Exterior Attention", fontsize=12)
    axs[1, 1].imshow(img2_overlay)
    axs[1, 1].set_title("Wrist Attention", fontsize=12)

    for ax in axs.flatten():
        ax.axis("off")

    # out_path = f"results/{name}/{attn_mode}_L{layer_idx}_attn_vis_{mode}.jpg"
    final_out_dir = os.path.join(output_dir, name)
    os.makedirs(final_out_dir, exist_ok=True)
    out_path = os.path.join(final_out_dir, f"{attn_mode}_L{layer_idx}_attn_vis_{mode}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    # print(f"Saved summary visualization to {out_path}")


def visualize_heads(
    example: dict, name: str, attn_mode: str, layer_idx: int, input_dir: str = "results", output_dir: str = "results"
):
    """Visualizes attention maps for EACH HEAD separately (Detailed View)."""
    from openpi.shared import image_tools
    import jax.numpy as jnp

    attn_path = os.path.join(input_dir, "layers_" + attn_mode, f"attn_map_layer_{layer_idx}.npy")
    if not os.path.exists(attn_path):
        return

    # Load: [Batch, NumHeads, SeqLen, SeqLen]
    attn_map = np.load(attn_path)
    if attn_map.ndim == 4:
        attn_map = attn_map[0]  # [NumHeads, SeqLen, SeqLen]

    num_heads = attn_map.shape[0]

    # Setup output dir
    # head_dir = f"results/{name}/L{layer_idx}_{attn_mode}_heads"
    head_dir = os.path.join(output_dir, name, f"L{layer_idx}_{attn_mode}_heads")
    os.makedirs(head_dir, exist_ok=True)

    num_image_tokens = 256
    total_image_tokens = 512
    text_start_idx = total_image_tokens

    # Prepare reference images (Resized to 224x224)
    img_ext_jax = jnp.array(example["observation/exterior_image_1_left"])
    img_wrist_jax = jnp.array(example["observation/wrist_image_left"])

    img_ext_resized = np.array(image_tools.resize_with_pad(img_ext_jax, 224, 224)).astype(np.uint8)
    img_wrist_resized = np.array(image_tools.resize_with_pad(img_wrist_jax, 224, 224)).astype(np.uint8)

    print(f"Visualizing {num_heads} heads for Layer {layer_idx} ({attn_mode})...")

    for head in range(num_heads):
        head_attn = attn_map[head]  # [SeqLen, SeqLen]

        if head_attn.shape[0] < total_image_tokens:
            # Suffix: Max over action tokens
            attn_val = head_attn[:, :total_image_tokens].max(axis=0)
        else:
            # Prefix: Max over text tokens
            if head_attn.shape[0] <= text_start_idx:
                continue
            attn_val = head_attn[text_start_idx:, :total_image_tokens].max(axis=0)

        img1_attn = attn_val[:num_image_tokens].reshape(16, 16)
        img2_attn = attn_val[num_image_tokens:].reshape(16, 16)

        # Plot Layout: 3 Columns [Attention 1, Attention 2, Original Images Stacked]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Heatmap 1 (Exterior)
        im1 = axs[0].imshow(img1_attn, cmap="viridis", interpolation="nearest", vmin=0)
        axs[0].set_title(f"Head {head:02d} - Exterior Attn")
        axs[0].axis("off")

        # Heatmap 2 (Wrist)
        im2 = axs[1].imshow(img2_attn, cmap="viridis", interpolation="nearest", vmin=0)
        axs[1].set_title(f"Head {head:02d} - Wrist Attn")
        axs[1].axis("off")

        # Column 3: Original Images (Stacked)
        combined_img = np.vstack([img_ext_resized, img_wrist_resized])
        axs[2].imshow(combined_img)
        axs[2].set_title("Reference Images")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{head_dir}/head_{head:02d}.jpg", bbox_inches="tight", dpi=100)
        plt.close()

    print(f"Saved head visualizations to {head_dir}")


def visualize_tokenizer(example: dict, name: str, output_dir: str = "results"):
    """Visualizes the tokenization of the prompt:
    Row 1: The reconstructed sentence with tokens colored.
    Row 2: The list of tokens (ID and text) also colored.
    """
    import matplotlib.pyplot as plt
    from openpi.models.tokenizer import PaligemmaTokenizer

    try:
        tokenizer = PaligemmaTokenizer()

        joint_pos = example.get("observation/joint_position", np.zeros(7))
        gripper_pos = example.get("observation/gripper_position", np.zeros(1))
        state = np.concatenate([joint_pos, gripper_pos])

        instruction = example["prompt"].strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "

        token_ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        tokens = [tokenizer._tokenizer.id_to_piece(i) for i in token_ids]
    except Exception as e:
        print(f"Tokenizer visualization failed: {e}")
        return

    # Colors (Pastel for readability)
    cmap = plt.get_cmap("Set3")
    colors = [cmap(i % 12) for i in range(len(tokens))]

    # Black background, compact size
    fig = plt.figure(figsize=(16, 9), facecolor="black")

    # Helper to plot tokens with wrapping
    def plot_tokens(ax, title, token_list, color_list=None, use_ids=False):
        ax.set_title(title, fontsize=16, loc="left", pad=10, color="white")
        ax.axis("off")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        # Layout parameters
        fontsize = 12
        fontname = "monospace"

        char_w_frac = 0.011

        # Estimate number of lines needed
        total_chars = sum(len(str(item)) for item in token_list)
        chars_per_line = int(0.97 / char_w_frac)  # Approximate chars that fit in one line
        estimated_lines = max(1, total_chars // chars_per_line + 2)

        # Dynamically adjust line height based on estimated lines
        line_h_frac = min(0.12, 0.85 / estimated_lines)

        x_start, y_start = 0.01, 0.95
        x, y = x_start, y_start
        x_limit = 0.98

        for i, token_item in enumerate(token_list):
            if use_ids:
                # token_item is (id, text)
                tid, text = token_item
                display_str = f"[{tid}: '{text}']"
                w = len(display_str) * char_w_frac
                bg_color = color_list[i] if color_list else None
                text_color = "black" if bg_color else "white"
            else:
                display_str = token_item.replace("\n", "\\n")
                w = len(display_str) * char_w_frac
                bg_color = color_list[i] if color_list else None
                text_color = "black" if bg_color else "white"

            # Check wrap
            is_newline = token_item == "\n" or token_item == "\\n"
            if (x + w > x_limit) or is_newline:
                x = x_start
                y -= line_h_frac
            if is_newline and not use_ids:
                pass
            # Check overflow
            if y < 0:
                ax.text(x, y + line_h_frac, "...", fontsize=fontsize, fontname=fontname, color="white")
                break
            # Draw
            kwargs = dict(
                fontsize=fontsize,
                fontname=fontname,
                va="top",
                color=text_color,
            )
            if bg_color:
                kwargs["backgroundcolor"] = bg_color
                kwargs["bbox"] = dict(facecolor=bg_color, edgecolor="none", pad=1.5)

            ax.text(x, y, display_str, **kwargs)

            x += w + 0.001  # Tighter gap

    # Use GridSpec for custom height ratios: ax1 and ax2 each get 1/5, ax3 gets 3/5
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 3, 0, 0], hspace=0.01)

    # 1. Original Prompt (White text, no background)
    ax1 = fig.add_subplot(gs[0])
    state_tokens = state_str.split(" ")
    display_tokens_1 = ["Task:", instruction, "\n", "State:"] + state_tokens + ["\n", "Action:"]

    plot_tokens(ax1, "Original Prompt Structure", display_tokens_1, color_list=None, use_ids=False)

    # 2. Reconstructed Prompt (Colored)
    ax2 = fig.add_subplot(gs[1])
    plot_tokens(ax2, "Reconstructed Prompt (Colored by Token)", tokens, colors, use_ids=False)

    # 3. Token List
    ax3 = fig.add_subplot(gs[2])
    token_pairs = list(zip(token_ids, tokens))
    plot_tokens(ax3, "Token List (IDs)", token_pairs, colors, use_ids=True)

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokenizer_vis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved tokenizer visualization to {out_path}")


def get_keyframes(total_frames: int, horizon: int) -> list[int]:
    """
    Returns a list of keyframe indices where the model inference actually happens.
    In Open-Loop Control / Action Chunking, the model predicts an action chunk of size `horizon`.
    So we only run inference at t=0, t=horizon, t=2*horizon, etc.
    """
    return list(range(0, total_frames, horizon))


def get_policy(checkpoint_dir: str):
    config = _config.get_config("pi05_droid")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    return policy


def process_episode(policy, example, output_dir, name, layers=[1, 4, 5, 7, 10]):
    # 0. Visualize Tokenizer (Once per episode)
    visualize_tokenizer(example, name, output_dir)

    # 1. Run Inference
    # This generates .npy files in results/layers_prefix/... (hardcoded in model)
    result = policy.infer(example)

    # 2. Visualize
    for layer in layers:
        # A. Summary View (Overlay) - Best layers
        for mode in ["max"]:
            visualize_attention(
                example,
                name,
                attn_mode="prefix",
                mode=mode,
                layer_idx=layer,
                input_dir="results",
                output_dir=output_dir,
            )
        # B. Detailed Head View (Side-by-Side) - All layers
        visualize_heads(example, name, attn_mode="prefix", layer_idx=layer, input_dir="results", output_dir=output_dir)

    return result


if __name__ == "__main__":
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    camera = "left"
    TOTAL_FRAMES = 90
    ACTION_HORIZON = 8  # Pi0 default action horizon

    keyframes = get_keyframes(TOTAL_FRAMES, ACTION_HORIZON)[:3]
    print(f"Running visualization on Keyframes: {keyframes}")

    # Visualize for keyframes only
    for index in keyframes:
        print(f"VLA index {index}")
        example = load_duck_example(camera=camera, index=index)
        # print out the joint post
        vis_example(example, f"duck_{camera}_{index}")
        process_episode(policy, example, "results", f"duck_{camera}_{index}")

    del policy
