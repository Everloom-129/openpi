"""
Wrist-Camera Object-Attention Correlation Pipeline

For every episode in DATA_ROOT that has perception data, computes the correlation
between each object's wrist-camera segmentation mask and the top-10% attention
patches per layer, using the original task instruction only.

Matching rules (which detected object is the "grasp target" vs "place target")
are configured in viz/config/object_matching.yaml.

Perception data layout (produced by perception_pipeline.py):
    {episode}/perception/{frame:05d}/perception.h5
        wrist/masks         (N, H, W)  uint8 binary
        wrist/bboxes/labels (N,)       bytes
        wrist/image         (H, W, 3)

Outputs per episode (in RESULTS_ROOT mirroring DATA_ROOT):
    h1_wrist_corr.json          — frame-level metrics + matched roles
    vis/{frame:05d}_attn.jpg    — per-frame attention panel
    vis/episode_timeline.png    — attention-on-target vs frame index

Aggregate outputs (RESULTS_ROOT/aggregate/):
    h1_wrist_corr_by_role.png   — grasp / place / other × success / failure
    h1_wrist_corr_per_layer.png — line plot across all 18 layers
    h1_wrist_corr_aggregate.json

Usage:
    # H5 mode (fast — reads pre-computed pipeline.py attention files):
    uv run python viz/h1_wrist_object_corr.py <DATA_ROOT> <RESULTS_ROOT> --from-h5

    # Inference mode (runs policy live):
    uv run python viz/h1_wrist_object_corr.py <DATA_ROOT> <RESULTS_ROOT>

    # Re-generate plots only:
    uv run python viz/h1_wrist_object_corr.py <DATA_ROOT> <RESULTS_ROOT> --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

# ── Path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from attn_map import get_policy, select_best_gpu
from openpi.models_pytorch import gemma_pytorch as _gpt

# ── Constants ─────────────────────────────────────────────────────────────────
OPEN_LOOP_HORIZON = 8
NUM_LAYERS = 18
CAMERA_EXT = "right"            # "right" → varied_camera_2
COMPLETION_MARKER = "h1_wrist_corr.json"

# Attention token layout: [ext(0:256)|wrist(256:512)|zero_pad(512:768)|text(768:N)|action]
NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512
WRIST_START = 256
WRIST_END = 512
TEXT_ROW_START = 768

TOP_PERCENT = 0.10   # "top 10%" attention threshold
PATCH_GRID = 16      # 16×16 = 256 patches per camera

DEFAULT_CHECKPOINT = "./checkpoints/viz/pi05_droid_pytorch"
DEFAULT_MATCHING_CONFIG = Path(__file__).parent / "config" / "object_matching.yaml"

ROLES = ("grasp", "place", "other")
ROLE_COLORS_MPL = {"grasp": "#1976D2", "place": "#F57C00", "other": "#757575"}


# ── Config loading ────────────────────────────────────────────────────────────

def load_matching_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Object matching ───────────────────────────────────────────────────────────

def match_target_objects(
    instruction: str,
    labels: list[str],
    cfg: dict,
) -> dict[str, str | None]:
    """Rule-based matching of instruction to detected object labels.

    Splits the instruction on the first placement preposition into a grasp-part
    (before) and a place-part (after), then scores each label by token overlap
    against its respective part. Ties broken by object-type priors.

    Returns: {"grasp": label_or_None, "place": label_or_None}
    """
    stop = set(cfg.get("stopwords", []))
    split_preps = cfg.get("split_prepositions", [])
    grasp_types = [t.lower() for t in cfg.get("grasp_types", [])]
    place_types = [t.lower() for t in cfg.get("place_types", [])]
    min_score = cfg.get("min_match_score", 0.0)

    instr_lower = instruction.lower()

    # Find first split preposition (word-boundary match, longest-first to avoid
    # "on" matching inside "onto" etc. — yaml list is already ordered longest-first)
    split_pos: int | None = None
    for prep in split_preps:
        m = re.search(r"\b" + re.escape(prep) + r"\b", instr_lower)
        if m:
            split_pos = m.start()
            break

    grasp_text = instr_lower[:split_pos] if split_pos is not None else instr_lower
    place_text = instr_lower[split_pos:] if split_pos is not None else ""

    def token_score(label: str, text: str) -> float:
        lw = set(label.lower().split()) - stop
        tw = set(text.lower().split()) - stop
        if not lw:
            return 0.0
        return len(lw & tw) / len(lw)

    def type_priority(label: str, preferred_types: list[str]) -> int:
        ll = label.lower()
        return sum(1 for t in preferred_types if t in ll)

    def best_label(text: str, exclude: str | None, preferred_types: list[str]) -> str | None:
        if not text.strip():
            return None
        candidates = [(lbl, token_score(lbl, text)) for lbl in labels if lbl != exclude]
        if not candidates:
            return None
        best_score = max(s for _, s in candidates)
        if best_score < min_score:
            return None
        tied = [lbl for lbl, s in candidates if s == best_score]
        if len(tied) == 1:
            return tied[0]
        # Tie-break: prefer labels whose type matches the role
        typed = sorted(tied, key=lambda l: type_priority(l, preferred_types), reverse=True)
        return typed[0]

    grasp = best_label(grasp_text, exclude=None, preferred_types=grasp_types)
    place = best_label(place_text, exclude=grasp, preferred_types=place_types)

    return {"grasp": grasp, "place": place}


def label_to_role(label: str, matched: dict[str, str | None]) -> str:
    """Return 'grasp', 'place', or 'other' for a detected label."""
    if label == matched.get("grasp"):
        return "grasp"
    if label == matched.get("place"):
        return "place"
    return "other"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_example(data_dir: Path, index: int, camera: str = "right") -> dict:
    """Load one frame from a DROID-format episode directory."""
    side_camera = "varied_camera_2" if camera == "right" else "varied_camera_1"
    frames_dir = data_dir / "recordings" / "frames"

    ext_path = frames_dir / side_camera / f"{index:05d}.jpg"
    hand_path = frames_dir / "hand_camera" / f"{index:05d}.jpg"

    if not ext_path.exists():
        raise FileNotFoundError(f"Exterior image not found: {ext_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"Hand image not found: {hand_path}")

    ext_img = np.array(Image.open(ext_path).convert("RGB"))
    hand_img = np.array(Image.open(hand_path).convert("RGB"))

    instruction_path = data_dir / "instruction.txt"
    instruction = instruction_path.read_text().strip() if instruction_path.exists() else ""

    traj_path = data_dir / "trajectory.h5"
    with h5py.File(traj_path, "r") as f:
        joint_position = f["observation/robot_state/joint_positions"][index].astype(np.float64)
        gripper_position = f["observation/robot_state/gripper_position"][index : index + 1].astype(np.float64)

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": instruction,
    }


def load_wrist_image(episode_dir: Path, frame_idx: int) -> np.ndarray | None:
    """Load raw wrist-camera image for visualization."""
    path = episode_dir / "recordings" / "frames" / "hand_camera" / f"{frame_idx:05d}.jpg"
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def load_perception(perc_h5: Path) -> list[dict]:
    """Load wrist-camera object data from a perception.h5 file.

    Returns list of dicts with keys: label, mask_raw (H×W bool), bbox.
    """
    objects = []
    with h5py.File(perc_h5, "r") as f:
        if "wrist" not in f:
            return objects
        masks = f["wrist/masks"][:]
        labels = f["wrist/bboxes/labels"][:]
        bboxes = f["wrist/bboxes/box_2d"][:]

    for i in range(len(labels)):
        label = labels[i].decode() if isinstance(labels[i], bytes) else str(labels[i])
        objects.append({
            "label": label,
            "mask_raw": masks[i].astype(bool),
            "bbox": bboxes[i].tolist(),
        })
    return objects


# ── Mask processing ───────────────────────────────────────────────────────────

def resize_mask_to_224(mask_raw: np.ndarray) -> np.ndarray:
    """Resize raw mask to 224×224 matching openpi's resize_with_pad."""
    raw_h, raw_w = mask_raw.shape[:2]
    target = 224
    scale = min(target / raw_w, target / raw_h)
    new_w, new_h = int(raw_w * scale), int(raw_h * scale)
    resized = cv2.resize(mask_raw.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((target, target), dtype=np.uint8)
    py, px = (target - new_h) // 2, (target - new_w) // 2
    out[py : py + new_h, px : px + new_w] = resized
    return out.astype(bool)


def resize_image_to_224(img: np.ndarray) -> np.ndarray:
    """Resize raw image to 224×224 matching openpi's resize_with_pad."""
    raw_h, raw_w = img.shape[:2]
    target = 224
    scale = min(target / raw_w, target / raw_h)
    new_w, new_h = int(raw_w * scale), int(raw_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((target, target, 3), dtype=np.uint8)
    py, px = (target - new_h) // 2, (target - new_w) // 2
    out[py : py + new_h, px : px + new_w] = resized
    return out


def mask_224_to_16x16(mask_224: np.ndarray) -> np.ndarray:
    """Downsample 224×224 bool mask to 16×16 (any-pixel-in-patch rule)."""
    patch_size = 224 // PATCH_GRID  # 14
    mask_16 = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            block = mask_224[r * patch_size:(r + 1) * patch_size,
                             c * patch_size:(c + 1) * patch_size]
            mask_16[r, c] = block.any()
    return mask_16


# ── Attention extraction ──────────────────────────────────────────────────────

def extract_wrist_attn(attn_buffer: dict, layer: int) -> np.ndarray | None:
    """Extract text→wrist attention from a live inference buffer. Returns (16,16) or None."""
    if layer not in attn_buffer:
        return None
    attn = attn_buffer[layer]
    if attn.ndim == 4:
        attn = attn[0]
    attn_avg = attn.mean(axis=0)
    if attn_avg.shape[0] <= TEXT_ROW_START:
        return None
    wrist_attn = attn_avg[TEXT_ROW_START:, WRIST_START:WRIST_END].max(axis=0)
    return wrist_attn.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)


def load_wrist_attn_from_h5(h5_path: Path) -> dict[int, np.ndarray]:
    """Read text→wrist attention from a pre-computed pipeline.py H5 file.

    Returns {layer: (16,16) float32} for all available layers.
    """
    result: dict[int, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for layer in range(NUM_LAYERS):
            key = f"prefix/layer_{layer}/text_to_img"
            if key not in f:
                continue
            t2i = f[key][:]  # (8, n_text, 512)
            wrist = t2i[:, :, WRIST_START:WRIST_END].mean(axis=0).max(axis=0)
            result[layer] = wrist.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return result


# ── Metric computation ────────────────────────────────────────────────────────

def compute_top10_metrics(attn_16: np.ndarray, obj_mask_16: np.ndarray) -> dict[str, float]:
    """Compute correlation between top-10% attention patches and an object mask."""
    threshold = np.percentile(attn_16.flatten(), (1 - TOP_PERCENT) * 100)
    top10 = attn_16 >= threshold
    obj = obj_mask_16.astype(bool)

    n_top10 = top10.sum()
    n_obj = obj.sum()
    n_inter = (top10 & obj).sum()
    n_union = (top10 | obj).sum()

    attn_norm = attn_16 / (attn_16.sum() + 1e-8)
    mean_in = attn_norm[obj].mean() if n_obj > 0 else 0.0
    mean_out = attn_norm[~obj].mean() if (~obj).sum() > 0 else 0.0

    return {
        "top10_precision":   float(n_inter / (n_top10 + 1e-8)),
        "top10_recall":      float(n_inter / (n_obj + 1e-8)),
        "top10_iou":         float(n_inter / (n_union + 1e-8)),
        "attn_on_obj_ratio": float(attn_norm[obj].sum()),
        "attn_concentration": float(mean_in / (mean_out + 1e-8)),
        "obj_patch_count":   int(n_obj),
    }


def _compute_frame_metrics(
    attn_per_layer: dict[int, np.ndarray],
    objects: list[dict],
) -> dict:
    """Returns {layer_idx: {obj_label: metrics_dict}}."""
    obj_masks = [
        (obj["label"], mask_224_to_16x16(resize_mask_to_224(obj["mask_raw"])))
        for obj in objects
    ]
    results = {}
    for layer, attn_16 in attn_per_layer.items():
        layer_results = {}
        for label, m16 in obj_masks:
            if m16.any():
                layer_results[label] = compute_top10_metrics(attn_16, m16)
        if layer_results:
            results[layer] = layer_results
    return results


def process_frame_from_buffer(attn_buffer: dict, objects: list[dict]) -> dict:
    attn_per_layer = {l: extract_wrist_attn(attn_buffer, l) for l in range(NUM_LAYERS)}
    attn_per_layer = {k: v for k, v in attn_per_layer.items() if v is not None}
    return _compute_frame_metrics(attn_per_layer, objects)


def process_frame_from_h5(h5_path: Path, objects: list[dict]) -> tuple[dict, dict]:
    """Returns (frame_metrics, attn_per_layer)."""
    attn_per_layer = load_wrist_attn_from_h5(h5_path)
    if not attn_per_layer:
        return {}, {}
    return _compute_frame_metrics(attn_per_layer, objects), attn_per_layer


# ── Visualization ─────────────────────────────────────────────────────────────

def _make_attn_overlay(wrist_224: np.ndarray, attn_16: np.ndarray, alpha: float) -> np.ndarray:
    """Blend attention heatmap onto a 224×224 BGR wrist image."""
    attn_up = cv2.resize(attn_16, (224, 224), interpolation=cv2.INTER_LINEAR)
    attn_norm = ((attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)  # BGR
    base = cv2.cvtColor(wrist_224, cv2.COLOR_RGB2BGR).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _draw_mask_contour(img_bgr: np.ndarray, mask_224: np.ndarray, color_bgr: list, thickness: int = 2) -> np.ndarray:
    """Draw mask contours on a BGR image in-place. Returns the image."""
    contours, _ = cv2.findContours(mask_224.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours, -1, tuple(color_bgr), thickness)
    return img_bgr


def _draw_top10_patches(img_bgr: np.ndarray, attn_16: np.ndarray, color_bgr: tuple = (0, 0, 255)) -> np.ndarray:
    """Outline top-10% attention patches as rectangles on a BGR image."""
    threshold = np.percentile(attn_16.flatten(), (1 - TOP_PERCENT) * 100)
    patch_px = 224 // PATCH_GRID  # 14
    img = img_bgr.copy()
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            if attn_16[r, c] >= threshold:
                x1, y1 = c * patch_px, r * patch_px
                x2, y2 = x1 + patch_px - 1, y1 + patch_px - 1
                cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 1)
    return img


def visualize_frame_attention(
    wrist_img: np.ndarray,
    attn_per_layer: dict[int, np.ndarray],
    objects: list[dict],
    matched: dict[str, str | None],
    output_path: Path,
    key_layers: list[int],
    frame_idx: int,
    instruction: str,
    vis_cfg: dict,
):
    """Save a multi-panel figure showing attention vs target masks at key layers.

    Layout: rows = [grasp target, place target] (only rows with a matched label),
            cols = [reference image | key_layer_0 | ... | key_layer_N].

    Each attention cell shows:
      - Wrist image blended with JET attention heatmap
      - Target mask drawn as a coloured contour
      - Top-10% attention patches outlined in red
    """
    grasp_label = matched.get("grasp")
    place_label = matched.get("place")
    alpha = vis_cfg.get("attn_alpha", 0.5)
    grasp_color = vis_cfg.get("grasp_color", [0, 220, 0])
    place_color = vis_cfg.get("place_color", [255, 140, 0])
    dpi = vis_cfg.get("fig_dpi", 150)

    # Build role → mask_224 mapping
    role_masks: dict[str, np.ndarray | None] = {"grasp": None, "place": None}
    for obj in objects:
        lbl = obj["label"]
        if lbl == grasp_label:
            role_masks["grasp"] = resize_mask_to_224(obj["mask_raw"])
        if lbl == place_label:
            role_masks["place"] = resize_mask_to_224(obj["mask_raw"])

    # Only show rows for roles that have a matched label
    active_roles = [r for r in ("grasp", "place") if matched.get(r) is not None]
    if not active_roles:
        return  # Nothing matched — skip visualization

    wrist_224 = resize_image_to_224(wrist_img)  # (224, 224, 3) RGB

    n_rows = len(active_roles)
    n_cols = 1 + len(key_layers)  # reference + one per key layer
    cell_size = 224
    fig_w = n_cols * cell_size / dpi + 1.5
    fig_h = n_rows * cell_size / dpi + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w * dpi / 72, fig_h * dpi / 72),
                             dpi=dpi, squeeze=False)

    for row_idx, role in enumerate(active_roles):
        mask_224 = role_masks[role]
        color_bgr = grasp_color if role == "grasp" else place_color
        target_label = matched.get(role, "")

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx == 0:
                # Reference: image + mask contour only
                ref = cv2.cvtColor(wrist_224, cv2.COLOR_RGB2BGR).copy()
                if mask_224 is not None:
                    _draw_mask_contour(ref, mask_224, color_bgr, thickness=3)
                ax.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{role.upper()}\n{target_label}", fontsize=7, pad=2)
            else:
                layer = key_layers[col_idx - 1]
                attn_16 = attn_per_layer.get(layer)
                if attn_16 is None:
                    ax.set_title(f"L{layer}\n(missing)", fontsize=7, pad=2)
                    continue

                cell = _make_attn_overlay(wrist_224, attn_16, alpha)
                if mask_224 is not None:
                    _draw_mask_contour(cell, mask_224, color_bgr, thickness=2)
                _draw_top10_patches(cell, attn_16, color_bgr=(0, 0, 200))
                ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))

                # Metrics for this role at this layer
                if mask_224 is not None:
                    m16 = mask_224_to_16x16(mask_224)
                    if m16.any():
                        m = compute_top10_metrics(attn_16, m16)
                        ax.set_title(
                            f"L{layer}  prec={m['top10_precision']:.2f}  rec={m['top10_recall']:.2f}",
                            fontsize=6, pad=2,
                        )
                    else:
                        ax.set_title(f"L{layer}", fontsize=7, pad=2)
                else:
                    ax.set_title(f"L{layer}", fontsize=7, pad=2)

    fig.suptitle(
        f"Frame {frame_idx:05d} — \"{instruction}\"",
        fontsize=8, y=1.01,
    )
    plt.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_episode_timeline(
    episode_results: dict,
    matched: dict[str, str | None],
    output_path: Path,
    key_layers: list[int],
):
    """Plot attention-on-target vs frame index for key layers.

    Subplots: one per active role (grasp / place).
    Lines: one per key layer.
    Metric shown: top10_precision (fraction of peak attention on target object).
    """
    active_roles = [r for r in ("grasp", "place") if matched.get(r) is not None]
    if not active_roles:
        return

    # Collect data: role → layer → [(frame_idx, top10_precision)]
    series: dict[str, dict[int, list[tuple[int, float]]]] = {
        role: {l: [] for l in key_layers} for role in active_roles
    }
    for frame_str, frame_data in episode_results.items():
        frame_idx = int(frame_str)
        for layer in key_layers:
            layer_data = frame_data.get(str(layer), {})
            for role in active_roles:
                label = matched[role]
                if label and label in layer_data:
                    val = layer_data[label].get("top10_precision", 0.0)
                    series[role][layer].append((frame_idx, val))

    layer_cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(key_layers)))

    n_rows = len(active_roles)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True, squeeze=False)

    for row, role in enumerate(active_roles):
        ax = axes[row, 0]
        for li, layer in enumerate(key_layers):
            pts = sorted(series[role][layer])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=layer_cmap[li], label=f"L{layer}",
                        linewidth=1.5, markersize=4, alpha=0.85)
        ax.set_ylabel("Top-10% Precision", fontsize=10)
        ax.set_title(f"{role.capitalize()} target: {matched[role]!r}", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="upper right")

    axes[-1, 0].set_xlabel("Frame index", fontsize=10)
    plt.suptitle("Attention on Target Object across Episode", fontsize=11, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Episode processing ────────────────────────────────────────────────────────

def get_perception_frames(episode_dir: Path) -> list[int]:
    perc_root = episode_dir / "perception"
    if not perc_root.exists():
        return []
    frames = []
    for d in perc_root.iterdir():
        if d.is_dir() and (d / "perception.h5").exists():
            try:
                frames.append(int(d.name))
            except ValueError:
                pass
    return sorted(frames)


def process_episode(
    episode_dir: Path,
    output_dir: Path,
    *,
    matching_cfg: dict,
    policy=None,
    attn_root: Path | None = None,
    visualize: bool = True,
) -> dict | None:
    """Process one episode. Returns result dict or None if skipped."""
    if (policy is None) == (attn_root is None):
        raise ValueError("Provide exactly one of: policy or attn_root.")

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / COMPLETION_MARKER
    if marker.exists():
        print(f"  [skip] {episode_dir.name} (already done)")
        return json.loads(marker.read_text())

    perc_frames = get_perception_frames(episode_dir)
    if not perc_frames:
        print(f"  [skip] {episode_dir.name} (no perception data)")
        return None

    instruction_path = episode_dir / "instruction.txt"
    instruction = instruction_path.read_text().strip() if instruction_path.exists() else ""
    mode = "H5" if attn_root else "inference"
    print(f"  [{mode}] {episode_dir.name!r}  \"{instruction}\"")

    vis_cfg = matching_cfg.get("vis", {})
    key_layers: list[int] = vis_cfg.get("key_layers", [0, 4, 7, 10, 13, 17])

    # Collect all labels across the episode's first perception frame for matching
    # (labels are usually stable; use first available frame)
    all_labels: list[str] = []
    for fi in perc_frames:
        perc_h5 = episode_dir / "perception" / f"{fi:05d}" / "perception.h5"
        objs = load_perception(perc_h5)
        if objs:
            all_labels = [o["label"] for o in objs]
            break

    matched = match_target_objects(instruction, all_labels, matching_cfg)
    print(f"    Matched → grasp: {matched['grasp']!r}  place: {matched['place']!r}")

    episode_results: dict[str, dict] = {}

    for frame_idx in perc_frames:
        print(f"    Frame {frame_idx:05d}...", end=" ", flush=True)

        perc_h5 = episode_dir / "perception" / f"{frame_idx:05d}" / "perception.h5"
        objects = load_perception(perc_h5)
        if not objects:
            print("no objects")
            continue

        if attn_root is not None:
            h5_path = attn_root / f"{frame_idx:05d}" / f"{frame_idx:05d}.h5"
            if not h5_path.exists():
                print(f"H5 missing")
                continue
            frame_metrics, attn_per_layer = process_frame_from_h5(h5_path, objects)
        else:
            try:
                example = load_example(episode_dir, frame_idx, camera=CAMERA_EXT)
            except FileNotFoundError as e:
                print(f"missing image: {e}")
                continue
            _gpt.enable_attn_buffer()
            try:
                policy.infer(example)
                attn_buf = _gpt.get_attn_buffer()
            finally:
                _gpt.clear_attn_buffer()
            attn_per_layer = {l: extract_wrist_attn(attn_buf, l) for l in range(NUM_LAYERS)}
            attn_per_layer = {k: v for k, v in attn_per_layer.items() if v is not None}
            frame_metrics = _compute_frame_metrics(attn_per_layer, objects)

        if not frame_metrics:
            print("no attention data")
            continue

        # Convert int keys to str for JSON
        episode_results[str(frame_idx)] = {
            str(layer): layer_data for layer, layer_data in frame_metrics.items()
        }
        print(f"{len(frame_metrics)} layers, {len(objects)} objects")

        # Per-frame visualization
        if visualize and attn_per_layer:
            wrist_img = load_wrist_image(episode_dir, frame_idx)
            if wrist_img is not None:
                vis_path = output_dir / "vis" / f"{frame_idx:05d}_attn.jpg"
                try:
                    visualize_frame_attention(
                        wrist_img=wrist_img,
                        attn_per_layer=attn_per_layer,
                        objects=objects,
                        matched=matched,
                        output_path=vis_path,
                        key_layers=key_layers,
                        frame_idx=frame_idx,
                        instruction=instruction,
                        vis_cfg=vis_cfg,
                    )
                except Exception as e:
                    print(f"      [vis warn] {e}")

    if not episode_results:
        return None

    # Episode timeline
    if visualize:
        try:
            plot_episode_timeline(
                episode_results, matched,
                output_path=output_dir / "vis" / "episode_timeline.png",
                key_layers=key_layers,
            )
        except Exception as e:
            print(f"  [timeline warn] {e}")

    output = {"instruction": instruction, "matched": matched, "frames": episode_results}
    marker.write_text(json.dumps(output, indent=2))
    print(f"  Saved → {marker}")
    return output


# ── Aggregation ───────────────────────────────────────────────────────────────

_METRICS = ["top10_precision", "top10_recall", "top10_iou", "attn_on_obj_ratio", "attn_concentration"]
_METRIC_LABELS = {
    "top10_precision":   "Top-10% Precision\n(peak-attn patches on object)",
    "top10_recall":      "Top-10% Recall\n(object covered by peak attention)",
    "top10_iou":         "Top-10% IoU",
    "attn_on_obj_ratio": "Attn Mass on Object",
    "attn_concentration":"Attn Concentration\n(inside/outside ratio)",
}


def aggregate(results_root: Path, output_dir: Path):
    """Collect per-episode JSONs and generate aggregate plots split by role and outcome."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # {outcome: {role: {layer: {metric: [values]}}}}
    agg: dict = {
        outcome: {
            role: {layer: {m: [] for m in _METRICS} for layer in range(NUM_LAYERS)}
            for role in ROLES
        }
        for outcome in ("success", "failure")
    }
    episode_counts: dict[str, int] = {"success": 0, "failure": 0}

    for outcome in ("success", "failure"):
        outcome_dir = results_root / outcome
        if not outcome_dir.exists():
            continue
        json_files = list(outcome_dir.rglob(COMPLETION_MARKER))
        print(f"{outcome}: {len(json_files)} episodes")
        for jf in json_files:
            try:
                ep = json.loads(jf.read_text())
            except Exception as e:
                print(f"  [warn] {jf}: {e}")
                continue
            episode_counts[outcome] += 1
            matched = ep.get("matched", {})
            for _frame, frame_data in ep.get("frames", {}).items():
                for layer_str, layer_data in frame_data.items():
                    layer = int(layer_str)
                    for label, obj_metrics in layer_data.items():
                        role = label_to_role(label, matched)
                        for m in _METRICS:
                            if m in obj_metrics:
                                agg[outcome][role][layer][m].append(obj_metrics[m])

    total = sum(episode_counts.values())
    if total == 0:
        print("No data found — run the pipeline first.")
        return
    print(f"Aggregating {total} episodes "
          f"({episode_counts['success']} success, {episode_counts['failure']} failure)")

    layers = list(range(NUM_LAYERS))
    outcome_ls = {"success": "-", "failure": "--"}
    outcome_marker = {"success": "o", "failure": "s"}

    # ── Plot 1: top10_precision — grasp+place roles, success vs failure ───────
    # One subplot per role (grasp / place), lines = success vs failure
    active_roles = [r for r in ("grasp", "place") if
                    any(agg[o][r][l]["top10_precision"] for o in ("success", "failure") for l in layers)]

    if active_roles:
        fig, axes = plt.subplots(len(active_roles), 1,
                                 figsize=(14, 4 * len(active_roles)), sharex=True)
        if len(active_roles) == 1:
            axes = [axes]
        for ax, role in zip(axes, active_roles):
            for outcome in ("success", "failure"):
                means = [np.mean(agg[outcome][role][l]["top10_precision"])
                         if agg[outcome][role][l]["top10_precision"] else np.nan
                         for l in layers]
                stds  = [np.std(agg[outcome][role][l]["top10_precision"])
                         if agg[outcome][role][l]["top10_precision"] else 0.0
                         for l in layers]
                means, stds = np.array(means), np.array(stds)
                n = episode_counts[outcome]
                c = "steelblue" if outcome == "success" else "coral"
                ax.plot(layers, means, linestyle=outcome_ls[outcome],
                        marker=outcome_marker[outcome], color=c,
                        label=f"{outcome} (n={n})", linewidth=2, markersize=5)
                ax.fill_between(layers, means - stds, means + stds, color=c, alpha=0.12)
            ax.set_title(f"{role.capitalize()} target — Top-10% Precision", fontsize=11)
            ax.set_ylabel("Precision", fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        axes[-1].set_xlabel("Layer", fontsize=11)
        axes[-1].set_xticks(layers)
        axes[-1].set_xticklabels([str(l) for l in layers])
        plt.suptitle(f"Top-10% Precision by Role — {total} episodes", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = output_dir / "h1_wrist_corr_by_role.png"
        plt.savefig(p, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

    # ── Plot 2: all metrics × all roles, success vs failure (line plot) ───────
    n_metrics = len(_METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 3.5 * n_metrics), sharex=True)
    role_styles = {"grasp": "-", "place": "--", "other": ":"}
    for ax, metric in zip(axes, _METRICS):
        for outcome in ("success", "failure"):
            c = "steelblue" if outcome == "success" else "coral"
            for role in ROLES:
                vals_per_layer = agg[outcome][role]
                means = [np.mean(vals_per_layer[l][metric]) if vals_per_layer[l][metric] else np.nan
                         for l in layers]
                if all(np.isnan(m) for m in means):
                    continue
                label_str = f"{outcome[:4]}/{role}"
                ax.plot(layers, means, color=c, linestyle=role_styles[role],
                        linewidth=1.5, label=label_str, alpha=0.85)
        ax.set_ylabel(_METRIC_LABELS[metric], fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=3)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    axes[-1].set_xlabel("Layer", fontsize=11)
    axes[-1].set_xticks(layers)
    axes[-1].set_xticklabels([str(l) for l in layers])
    plt.suptitle(f"All Metrics by Role (solid=grasp, dashed=place, dot=other) — {total} episodes",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = output_dir / "h1_wrist_corr_per_layer.png"
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _summarise(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    agg_json = {
        "episode_counts": episode_counts,
        "top_percent": TOP_PERCENT,
        "data": {
            outcome: {
                role: {
                    str(layer): {m: _summarise(agg[outcome][role][layer][m]) for m in _METRICS}
                    for layer in layers
                }
                for role in ROLES
            }
            for outcome in ("success", "failure")
        },
    }
    p = output_dir / "h1_wrist_corr_aggregate.json"
    p.write_text(json.dumps(agg_json, indent=2))
    print(f"Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root",    type=Path, help="Root of DROID episodes (success/ and failure/ subdirs)")
    parser.add_argument("results_root", type=Path, help="Output root (and H5 source in --from-h5 mode)")
    parser.add_argument("--checkpoint",      default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu",             default=None)
    parser.add_argument("--matching-config", type=Path, default=DEFAULT_MATCHING_CONFIG,
                        help="Path to object_matching.yaml (default: viz/config/object_matching.yaml)")
    parser.add_argument("--from-h5",         action="store_true",
                        help="Read pre-computed pipeline.py H5 files instead of re-running inference")
    parser.add_argument("--no-vis",          action="store_true",
                        help="Skip per-frame visualization (faster batch processing)")
    parser.add_argument("--force",           action="store_true", help="Reprocess completed episodes")
    parser.add_argument("--aggregate-only",  action="store_true",
                        help="Skip processing; re-generate aggregate plots only")
    args = parser.parse_args()

    matching_cfg = load_matching_config(args.matching_config)
    agg_dir = args.results_root / "aggregate"

    if args.aggregate_only:
        aggregate(args.results_root, agg_dir)
        return

    policy = None
    if not args.from_h5:
        device = args.gpu or f"cuda:{select_best_gpu()}"
        print(f"Using device: {device}")
        print(f"Loading policy from {args.checkpoint} ...")
        policy = get_policy(args.checkpoint, device=device)
        print("Policy loaded.")
    else:
        print("H5 mode: reading pre-computed attention from results_root.")

    for outcome in ("success", "failure"):
        outcome_src = args.data_root / outcome
        if not outcome_src.exists():
            print(f"[warn] {outcome_src} does not exist, skipping.")
            continue
        for date_dir in sorted(d for d in outcome_src.iterdir() if d.is_dir()):
            for ep_dir in sorted(d for d in date_dir.iterdir() if d.is_dir()):
                out_dir = args.results_root / outcome / date_dir.name / ep_dir.name

                if args.force and (out_dir / COMPLETION_MARKER).exists():
                    (out_dir / COMPLETION_MARKER).unlink()

                print(f"\n[{outcome}] {date_dir.name}/{ep_dir.name}")
                try:
                    attn_root = out_dir if args.from_h5 else None
                    process_episode(
                        ep_dir, out_dir,
                        matching_cfg=matching_cfg,
                        policy=policy,
                        attn_root=attn_root,
                        visualize=not args.no_vis,
                    )
                except Exception as e:
                    import traceback
                    print(f"  [error] {e}")
                    traceback.print_exc()

    print(f"\n{'='*60}\nAggregating results...")
    aggregate(args.results_root, agg_dir)
    print("Done.")


if __name__ == "__main__":
    main()
