import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from ultralytics import YOLO
import sys

# Ensure we can import from the current directory if needed
sys.path.append(os.getcwd())
viz_dir = os.path.join(os.getcwd(), "viz")
if viz_dir not in sys.path:
    sys.path.append(viz_dir)

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

try:
    from viz.attn_map import load_duck_example
except ImportError:
    from attn_map import load_duck_example


@dataclass
class ROI:
    name: str
    camera: str  # "exterior" or "wrist"
    # Bounding box in 224x224 coordinates: (x, y, w, h)
    bbox: tuple[int, int, int, int]


def detect_and_visualize_rois(image_np, output_dir="results_analysis"):
    """
    Use YOLO to detect objects, return ROIs, and save visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Running YOLO detection...")
    try:
        model = YOLO("yolo11n.pt")
    except Exception:
        print("Could not load yolo11n.pt, trying yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # Run inference
    results = model(image_np, verbose=False)
    result = results[0]

    # Prepare visualization image (resize to 224x224 to match Attention Map)
    vis_img = cv2.resize(image_np.copy(), (224, 224))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    rois = []

    # Scaling factors (Original -> 224x224)
    orig_h, orig_w = image_np.shape[:2]
    scale_x = 224.0 / orig_w
    scale_y = 224.0 / orig_h

    print(f"Detected objects in exterior image:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])

        if conf < 0.2:  # Threshold
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Scale to 224x224
        sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
        sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
        w, h = sx2 - sx1, sy2 - sy1

        roi_name = None
        # Heuristic mapping for Duck/Bowl task
        if cls_name in ["bird", "teddy bear", "sheep", "dog", "cat", "duck"]:
            roi_name = f"Object ({cls_name})"
        elif cls_name in ["bowl", "cup", "potted plant", "vase", "bottle"]:
            roi_name = f"Goal ({cls_name})"

        if roi_name:
            print(f" - {cls_name} ({conf:.2f}) -> {roi_name}")
            rois.append(ROI(name=roi_name, camera="exterior", bbox=(sx1, sy1, w, h)))

            # Draw on visualization
            color = (0, 255, 0) if "Object" in roi_name else (0, 0, 255)
            cv2.rectangle(vis_img, (sx1, sy1), (sx1 + w, sy1 + h), color, 2)
            cv2.putText(vis_img, roi_name, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Always add Gripper (Approximation)
    gripper_roi = ROI(name="Gripper", camera="wrist", bbox=(80, 80, 64, 144))
    rois.append(gripper_roi)

    # Save visualization
    vis_path = os.path.join(output_dir, "roi_detection_vis.jpg")
    cv2.imwrite(vis_path, vis_img)
    print(f"Saved ROI visualization to {vis_path}")

    return rois


def clean_suffix_results():
    if os.path.exists("results/layers_suffix"):
        shutil.rmtree("results/layers_suffix")
    os.makedirs("results/layers_suffix", exist_ok=True)


def run_inference(policy, example):
    print("Running inference...")
    clean_suffix_results()
    result = policy.infer(example)
    print("Inference complete.")
    return result


def load_suffix_attention(layer_idx=10):
    # 1. Try loading the new fused format first
    pattern_full = "results/layers_suffix/attn_map_full_step_*.npy"
    files_full = glob.glob(pattern_full)

    if files_full:
        # Load from fused file: [NumLayers, Batch, Heads, QLen, KLen]
        f = files_full[0]
        # print(f"Loading fused attention from {f} for Layer {layer_idx}")
        data = np.load(f)

        # Select Layer
        # data shape: [NumLayers, ...]
        if layer_idx >= data.shape[0]:
            print(f"Error: Layer {layer_idx} out of bounds (Max {data.shape[0] - 1})")
            return [], []

        layer_data = data[layer_idx]  # [Batch, Heads, QLen, KLen]
        if layer_data.ndim == 4:
            layer_data = layer_data[0]  # [Heads, QLen, KLen]

        num_heads, q_len, k_len = layer_data.shape
        steps = list(range(q_len))
        attns = [layer_data[:, t, :] for t in range(q_len)]
        return steps, attns

    # 2. Fallback to old split format
    pattern = f"results/layers_suffix/attn_map_layer_{layer_idx}_step_*.npy"
    files = glob.glob(pattern)

    if not files:
        print(f"No attention files found for layer {layer_idx}")
        return [], []

    # Assuming single batch/file for now
    data = np.load(files[0])  # [Batch, Heads, QLen, KLen] or [Heads, QLen, KLen]
    if data.ndim == 4:
        data = data[0]

    num_heads, q_len, k_len = data.shape
    steps = list(range(q_len))

    # List of [Heads, KLen] for each step
    attns = [data[:, t, :] for t in range(q_len)]
    return steps, attns


def get_roi_mask(roi: ROI, grid_size=(16, 16)):
    scale_x = grid_size[0] / 224.0
    scale_y = grid_size[1] / 224.0
    x, y, w, h = roi.bbox

    x1 = int(x * scale_x)
    y1 = int(y * scale_y)
    x2 = int((x + w) * scale_x)
    y2 = int((y + h) * scale_y)

    # Clip to grid
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(grid_size[0], x2), min(grid_size[1], y2)

    mask = np.zeros(grid_size, dtype=float)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def analyze_temporal_shift(layer_idx, rois):
    steps, attns = load_suffix_attention(layer_idx)
    if not steps:
        return None, None

    results = {roi.name: [] for roi in rois}
    NUM_IMG_TOKENS = 256
    TOTAL_IMG_TOKENS = 512

    for attn_head in attns:
        # Average over heads -> [KLen]
        attn = attn_head.mean(axis=0)

        # Attention from Action Token to Image Tokens
        img_attn = attn[:TOTAL_IMG_TOKENS]
        ext_attn = img_attn[:NUM_IMG_TOKENS].reshape(16, 16)
        wrist_attn = img_attn[NUM_IMG_TOKENS:].reshape(16, 16)

        for roi in rois:
            mask = get_roi_mask(roi)
            val = np.sum(ext_attn * mask) if roi.camera == "exterior" else np.sum(wrist_attn * mask)
            results[roi.name].append(val)

    return steps, results


def plot_temporal_shift(steps, results, layer_idx, output_dir="results_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for name, values in results.items():
        plt.plot(steps, values, label=name, marker="o", markersize=4)

    plt.title(f"Temporal Shift of Attention (Layer {layer_idx})")
    plt.xlabel("Action Token Index")
    plt.ylabel("Integrated Attention")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"temporal_shift_L{layer_idx}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()


def main():
    # 1. Setup Policy
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    print(f"Loading policy from {checkpoint_dir}...")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 2. Load Example & Detect ROIs
    example = load_duck_example(camera="left", index=0)
    rois = detect_and_visualize_rois(example["observation/exterior_image_1_left"])

    if not rois:
        print("No ROIs detected, skipping analysis.")
        return

    # 3. Run Inference (Generates Attention Maps)
    run_inference(policy, example)

    # 4. Analyze & Plot
    for layer in [10, 17]:
        print(f"Analyzing Layer {layer}...")
        steps, results = analyze_temporal_shift(layer, rois)
        if steps:
            plot_temporal_shift(steps, results, layer)


if __name__ == "__main__":
    main()
