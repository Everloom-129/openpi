import os

import cv2

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
    ext = ".webm"
    layer_dir = os.path.join(output_dir, f"L{layer}")
    os.makedirs(layer_dir, exist_ok=True)
    output_filename = os.path.join(layer_dir, f"H{head:02d}_{ATTN_MODE}{ext}")

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

    # Check extension to decide codec
    ext = os.path.splitext(output_filename)[1]
    if ext == ".webm":
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
    elif ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
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

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved summary video to {output_filename}")


if __name__ == "__main__":
    for layer in [1, 4, 5, 7, 10]:
        print(f"Generating videos for layer {layer}")
        create_summary_video(layer, mode="max", timesteps=TIMESTEPS)
        # for head in range(8):
        #     create_video_for_head(layer, head, timesteps=TIMESTEPS)
