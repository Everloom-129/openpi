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


def get_duck_image_path(timestep_idx, layer, head):
    """Construct path to the visualization image."""
    folder_name = f"duck_{CAMERA}_{timestep_idx}"
    filename = f"head_{head:02d}.jpg"
    # Path: results/duck_left_0/L15_prefix_heads/head_03.jpg
    path = os.path.join(BASE_DIR, folder_name, f"L{layer}_{ATTN_MODE}_heads", filename)
    return path


def get_image_path(timestep_idx, layer, head, base_dir=BASE_DIR, camera=CAMERA):
    """Construct path to the visualization image."""
    # This assumes the structure is base_dir/duck_{camera}_{timestep}/...
    # But in the new pipeline, base_dir is already .../episode
    # and the frames are in .../episode/{timestep:05d}/...

    # Check if base_dir has "duck" in it, otherwise assume it's the episode dir
    folder_name = f"duck_{camera}_{timestep_idx}"
    path_duck = os.path.join(base_dir, folder_name, f"L{layer}_{ATTN_MODE}_heads", f"head_{head:02d}.jpg")

    if os.path.exists(path_duck):
        return path_duck

    # New structure: base_dir/{timestep:05d}/L{layer}_prefix_heads/head_{head:02d}.jpg
    # where base_dir is results_toy/success/.../episode

    folder_name_new = f"{timestep_idx:05d}"
    path_new = os.path.join(base_dir, folder_name_new, f"L{layer}_{ATTN_MODE}_heads", f"head_{head:02d}.jpg")

    return path_new


def get_summary_image_path(timestep_idx, layer, mode="max", base_dir=BASE_DIR, camera=CAMERA):
    """Construct path to the summary overlay image."""
    folder_name = f"duck_{camera}_{timestep_idx}"
    filename = f"{ATTN_MODE}_L{layer}_attn_vis_{mode}.jpg"

    path_duck = os.path.join(base_dir, folder_name, filename)
    if os.path.exists(path_duck):
        return path_duck

    # New structure: base_dir/{timestep:05d}/prefix_L{layer}_attn_vis_max.jpg
    folder_name_new = f"{timestep_idx:05d}"
    path_new = os.path.join(base_dir, folder_name_new, filename)
    return path_new


def create_video_for_head(
    layer, head, fps=FPS_VIDEO, timesteps=TIMESTEPS, output_dir="results/videos", input_dir=BASE_DIR
):
    """Creates a video sequence for a specific layer/head across all timesteps."""

    # Try VP90 (VP9) - Newer standard, often better supported
    fourcc_code = "VP90"
    ext = ".webm"

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"L{layer:02d}_H{head:02d}_{ATTN_MODE}{ext}")

    frames = []
    for t in timesteps:
        path = get_image_path(t, layer, head, base_dir=input_dir)
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
    # fourcc = cv2.VideoWriter_fourcc(*fourcc_code)

    # Check extension to decide codec
    ext = os.path.splitext(output_filename)[1]
    if ext == ".webm":
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
    elif ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Fallback to MJPG (.avi) if writer fails
    if not out.isOpened():
        print(f"Warning: Could not open video writer for {output_filename}. Falling back to MJPG (.avi).")
        output_filename = os.path.splitext(output_filename)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_filename}")


def create_summary_video(
    layer, mode="max", fps=FPS_VIDEO, output_dir="results/videos/summary", timesteps=TIMESTEPS, input_dir=BASE_DIR
):
    """Creates a video sequence for the summary overlay of a specific layer."""
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"L{layer:02d}_{ATTN_MODE}_{mode}.webm")

    frames = []
    for t in timesteps:
        path = get_summary_image_path(t, layer, mode, base_dir=input_dir)
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
    ext = os.path.splitext(output_filename)[1]

    if ext == ".webm":
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
    elif ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Fallback to MJPG (.avi) if WebM fails
    if not out.isOpened():
        print(f"Warning: Could not open video writer for {output_filename}. Falling back to MJPG (.avi).")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_filename = os.path.splitext(output_filename)[0] + ".avi"
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
        create_summary_video(layer, mode="max", timesteps=TIMESTEPS)
        # for head in range(8):
        #     create_video_for_head(layer, head, timesteps=TIMESTEPS)
