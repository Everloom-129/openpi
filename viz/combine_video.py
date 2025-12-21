import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
from attn_map import get_keyframes

# Configuration
BASE_DIR = "results"
CAMERA = "left"
ATTN_MODE = "prefix"  # or "suffix"
TOTAL_FRAMES = 90
ACTION_HORIZON = 8
FPS_VIDEO = 5


TIMESTEPS = range(TOTAL_FRAMES)
# TIMESTEPS = get_keyframes(TOTAL_FRAMES, ACTION_HORIZON)


def get_image_path(timestep_idx, layer, head):
    """Construct path to the visualization image."""
    folder_name = f"duck_{CAMERA}_{timestep_idx}"
    filename = f"head_{head:02d}.jpg"
    # Path: results/duck_left_0/L15_prefix_heads/head_03.jpg
    path = os.path.join(BASE_DIR, folder_name, f"L{layer}_{ATTN_MODE}_heads", filename)
    return path


def get_summary_image_path(timestep_idx, layer, mode="max"):
    """Construct path to the summary overlay image."""
    folder_name = f"duck_{CAMERA}_{timestep_idx}"
    # Example: results/duck_left_0/prefix_L15_attn_vis_max.jpg
    filename = f"{ATTN_MODE}_L{layer}_attn_vis_{mode}.jpg"
    path = os.path.join(BASE_DIR, folder_name, filename)
    return path


def calculate_focus_score(image_path):
    """
    Calculate a score indicating how 'focused' the attention is.
    We use the Variance of the heatmap region as a proxy.
    Higher Variance = More focused spots (High peaks vs Low background).
    Lower Variance = Diffuse/Uniform attention.
    """
    if not os.path.exists(image_path):
        return 0.0

    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    # Image layout is 3 columns: [Ext Attn] [Wrist Attn] [Ref Images]
    # We focus on the first column (Exterior Attention) for scoring
    height, width, _ = img.shape
    one_third_width = width // 3

    # Crop the Exterior Attention Heatmap part
    ext_attn_img = img[:, :one_third_width, :]

    # Convert to grayscale to measure intensity variance
    gray = cv2.cvtColor(ext_attn_img, cv2.COLOR_BGR2GRAY)

    # Calculate Variance
    score = np.var(gray)
    return score


def rank_best_heads(reference_timestep=0):
    """
    Scans all layers and heads for a specific timestep and ranks them by focus score.
    """
    print(f"Analyzing heads for timestep {reference_timestep} to find the best ones...")
    scores = []

    # Assume 18 layers and 8 heads (standard for PaliGemma/Pi0)
    for layer in range(18):
        for head in range(8):
            path = get_image_path(reference_timestep, layer, head)
            score = calculate_focus_score(path)
            if score > 0:
                scores.append(((layer, head), score))

    # Sort by score descending (High variance first)
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 Most Focused Heads (Layer, Head):")
    for (layer, head), score in scores[:10]:
        print(f"Layer {layer:02d}, Head {head:02d} | Score: {score:.2f}")

    return [x[0] for x in scores[:10]]


def create_video_for_head(layer, head, output_filename=None):
    """Creates a video sequence for a specific layer/head across all timesteps."""

    # Try VP90 (VP9) - Newer standard, often better supported
    fourcc_code = "VP90"
    ext = ".webm"

    if output_filename is None:
        os.makedirs("results/videos", exist_ok=True)
        output_filename = f"results/videos/L{layer:02d}_H{head:02d}_{ATTN_MODE}{ext}"

    frames = []
    for t in TIMESTEPS:
        path = get_image_path(t, layer, head)
        if os.path.exists(path):
            img = cv2.imread(path)
            # Add text annotation for timestep
            cv2.putText(img, f"Timestep: {t}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(img)
        else:
            print(f"Warning: Missing frame for T={t} at {path}")

    if not frames:
        print(f"No frames found for L{layer} H{head}, skipping video.")
        return

    height, width, _ = frames[0].shape

    # Try creating video writer
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    fps = FPS_VIDEO
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Fallback to MJPG (.avi) if WebM fails
    if not out.isOpened():
        print(f"Warning: Could not open video writer with {fourcc_code}. Falling back to MJPG (.avi).")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_filename = output_filename.replace(ext, ".avi")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_filename}")


def create_summary_video(layer, mode="max"):
    """Creates a video sequence for the summary overlay of a specific layer."""
    output_dir = "results/videos/summary"
    os.makedirs(output_dir, exist_ok=True)

    # Try VP90 (VP9) - Newer standard, often better supported
    fourcc_code = "VP90"
    ext = ".webm"
    output_filename = os.path.join(output_dir, f"L{layer:02d}_{ATTN_MODE}_{mode}{ext}")

    frames = []
    for t in TIMESTEPS:
        path = get_summary_image_path(t, layer, mode)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                # Add text annotation for timestep
                cv2.putText(
                    img, f"Timestep: {t}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )
                frames.append(img)
        else:
            # print(f"Warning: Missing summary frame for T={t} at {path}")
            pass

    if not frames:
        print(f"No summary frames found for L{layer}, skipping video.")
        return

    height, width, _ = frames[0].shape

    # Try creating video writer
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    fps = FPS_VIDEO
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Fallback to MJPG (.avi) if WebM fails
    if not out.isOpened():
        print(f"Warning: Could not open video writer with {fourcc_code}. Falling back to MJPG (.avi).")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_filename = output_filename.replace(ext, ".avi")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved summary video to {output_filename}")


if __name__ == "__main__":
    # 1. Analyze and find best heads based on the first timestep (or index 30 which might be more active)
    # Using index 0 as baseline
    # top_heads = rank_best_heads(reference_timestep=0)

    # 2. Generate videos for the Top 5 heads
    # print("\nGenerating videos for top 5 heads...")
    # for layer, head in top_heads[:5]:
    #     create_video_for_head(layer, head)

    # Optional: Manually generate for specific layers you are interested in (e.g. L15)
    for layer in [1, 4, 5, 7, 10]:
        print(f"Generating videos for layer {layer}")
        create_summary_video(layer, mode="max")
        # for head in range(8):
        #     create_video_for_head(layer, head)
