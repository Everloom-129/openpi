"""Standalone test for JAX inference pipeline with attention visualization.

Usage:
    uv run python viz/test_jax_inference.py

    # To restrict to a single GPU (if others are busy):
    CUDA_VISIBLE_DEVICES=0 uv run python viz/test_jax_inference.py
"""
from __future__ import annotations

import os
import sys
import traceback

# Prevent JAX from pre-allocating all GPU memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
# Full tracebacks from JAX for debugging.
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

JAX_CHECKPOINT_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache/openpi/openpi-assets/checkpoints"
)
CHECKPOINT_NAME = "pi05_droid"
CHECKPOINT_DIR = os.path.join(JAX_CHECKPOINT_ROOT, CHECKPOINT_NAME)
FRAME_IDX = 40
SAVE_DIR = os.path.join(_PROJECT_ROOT, "attn_jax")

NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512
PATCH_GRID = 16


# ── Visualization helpers ─────────────────────────────────────────────────────

def attn_to_heatmap(attn_512: np.ndarray, camera: str = "exterior") -> np.ndarray:
    """Extract 16x16 patch grid for one camera, return as float32 heatmap."""
    if camera == "exterior":
        patches = attn_512[:NUM_IMAGE_TOKENS]
    else:
        patches = attn_512[NUM_IMAGE_TOKENS:TOTAL_IMAGE_TOKENS]
    return patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)


def overlay_heatmap(img: np.ndarray, hmap_16: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a 16x16 heatmap on an image. Returns BGR uint8."""
    h, w = img.shape[:2]
    hmap_up = cv2.resize(hmap_16, (w, h), interpolation=cv2.INTER_LINEAR)
    hmin, hmax = hmap_up.min(), hmap_up.max()
    hmap_norm = (hmap_up - hmin) / (hmax - hmin + 1e-8)
    hmap_u8 = (hmap_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    # Convert img to BGR if needed
    if img.shape[-1] == 3 and img.dtype == np.uint8:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)
    return blended


def make_layer_grid(
    prefix: dict,
    token_idx: int,
    camera_img: np.ndarray,
    camera: str,
    layers: list[int],
    token_label: str = "",
) -> np.ndarray:
    """Build a grid image: rows=layers, cols=8 heads. Returns BGR uint8."""
    cell_h, cell_w = 112, 112
    n_layers = len(layers)
    n_heads = 8
    margin = 2

    grid_h = n_layers * (cell_h + margin) + margin
    grid_w = n_heads * (cell_w + margin) + margin
    canvas = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)  # dark bg

    img_small = cv2.resize(
        cv2.cvtColor(camera_img, cv2.COLOR_RGB2BGR) if camera_img.shape[-1] == 3 else camera_img,
        (cell_w, cell_h),
    )

    for row, layer in enumerate(layers):
        key = f"layer_{layer}"
        if key not in prefix:
            continue
        t2i = prefix[key]["text_to_img"]  # (8, n_text, 512)
        for head in range(min(n_heads, t2i.shape[0])):
            if token_idx >= t2i.shape[1]:
                continue
            attn_vec = t2i[head, token_idx]  # (512,)
            hmap = attn_to_heatmap(attn_vec, camera)
            cell = overlay_heatmap(camera_img, hmap, alpha=0.5)
            cell = cv2.resize(cell, (cell_w, cell_h))

            y = row * (cell_h + margin) + margin
            x = head * (cell_w + margin) + margin
            canvas[y : y + cell_h, x : x + cell_w] = cell

    return canvas


def make_token_summary(
    prefix: dict,
    camera_img: np.ndarray,
    camera: str,
    token_texts: list[str],
    layer: int = 7,
) -> np.ndarray:
    """One row per token, mean across heads. Returns BGR uint8."""
    key = f"layer_{layer}"
    if key not in prefix:
        return np.zeros((100, 400, 3), dtype=np.uint8)
    t2i = prefix[key]["text_to_img"]  # (8, n_text, 512)
    t2i_mean = t2i.mean(axis=0)  # (n_text, 512)
    n_tokens = min(len(token_texts), t2i_mean.shape[0])

    cell_h, cell_w = 80, 80
    label_w = 120
    cols_per_row = 1
    row_h = cell_h + 4
    canvas_h = n_tokens * row_h + 4
    canvas_w = label_w + cell_w + 8
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for i in range(n_tokens):
        attn_vec = t2i_mean[i]
        hmap = attn_to_heatmap(attn_vec, camera)
        cell = overlay_heatmap(camera_img, hmap, alpha=0.5)
        cell = cv2.resize(cell, (cell_w, cell_h))

        y = i * row_h + 4
        # Draw token label
        label = token_texts[i][:14]
        cv2.putText(canvas, label, (4, y + cell_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # Draw cell
        x = label_w + 4
        canvas[y : y + cell_h, x : x + cell_w] = cell

    return canvas


def make_full_attn_matrix(prefix: dict, layer: int, head: int) -> np.ndarray:
    """Render full attention matrix for one layer/head as a heatmap image."""
    key = f"layer_{layer}"
    if key not in prefix:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    full = prefix[key]["full"]  # (8, seq, seq)
    mat = full[head].astype(np.float32)
    # Normalize
    vmin, vmax = mat.min(), mat.max()
    mat_norm = (mat - vmin) / (vmax - vmin + 1e-8)
    mat_u8 = (mat_norm * 255).astype(np.uint8)
    # Resize for display
    display_size = min(800, max(256, mat_u8.shape[0]))
    mat_resized = cv2.resize(mat_u8, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(mat_resized, cv2.COLORMAP_VIRIDIS)
    return colored


# ── Main pipeline ─────────────────────────────────────────────────────────────

def step(name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


# ── Step 1: Load example ─────────────────────────────────────────────────────
step("Load duck example (frame 40)")
try:
    from attn_map import load_duck_example

    example = load_duck_example(camera="left", index=FRAME_IDX)
    example["prompt"] = "place the duck toy into the pink bowl"
    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]
    print(f"  prompt: {example['prompt']}")
    print(f"  exterior: {ext_img.shape}, wrist: {wrist_img.shape}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 2: Load model ───────────────────────────────────────────────────────
step("Load JAX model")
try:
    import pathlib
    from openpi.training import config as _cfg
    from openpi.models import model as _model
    from openpi.shared import download

    raw = os.path.basename(os.path.normpath(CHECKPOINT_DIR))
    config_name = "pi05_droid" if "pi05" in raw.lower() else "pi0_droid"
    is_pi05 = "pi05" in config_name

    config = _cfg.get_config(config_name)
    ckpt = pathlib.Path(str(download.maybe_download(CHECKPOINT_DIR)))
    params_path = ckpt / "params" if (ckpt / "params").is_dir() else ckpt
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    model = config.model.load(params)
    model.eval()
    print(f"  config: {config_name}, model: {type(model).__name__}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 3: Build observation ─────────────────────────────────────────────────
step("Build observation & tokenize")
try:
    import jax
    import jax.numpy as jnp

    images = {}
    image_masks = {}
    for img_key, model_key in [
        ("observation/exterior_image_1_left", "base_0_rgb"),
        ("observation/wrist_image_left", "left_wrist_0_rgb"),
    ]:
        img = example[img_key]
        if img.ndim == 3:
            img = img[None]
        images[model_key] = jnp.array(img, dtype=jnp.float32) / 255.0
        image_masks[model_key] = jnp.ones((img.shape[0],), dtype=jnp.bool_)

    dummy = jnp.zeros_like(next(iter(images.values())))
    images["right_wrist_0_rgb"] = dummy
    image_masks["right_wrist_0_rgb"] = jnp.zeros((dummy.shape[0],), dtype=jnp.bool_)

    state = jnp.array(example["observation/joint_position"], dtype=jnp.float32)
    if state.ndim == 1:
        state = state[None]

    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer()
    prompt = example["prompt"]
    max_token_len = getattr(model, "max_token_len", 48)
    token_ids, token_mask = tokenizer.tokenize(prompt, state=state[0] if is_pi05 else None)
    token_ids = np.array(token_ids[:max_token_len], dtype=np.int32)
    token_mask = np.array(token_mask[:max_token_len], dtype=bool)
    n_real_tokens = int(token_mask.sum())
    token_texts = [tokenizer._tokenizer.id_to_piece(int(i)) for i in token_ids[:n_real_tokens]]

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=jnp.array(token_ids[None]),
        tokenized_prompt_mask=jnp.array(token_mask[None], dtype=bool),
    )
    print(f"  tokens ({n_real_tokens}): {' '.join(token_texts[:15])}...")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 4: Run inference ─────────────────────────────────────────────────────
step("Run forward_with_attention()")
try:
    rng = jax.random.key(0)
    actions, prefix_attn, suffix_attn = model.forward_with_attention(rng, observation)
    actions_np = np.array(actions)
    print(f"  actions: {actions_np.shape}")
    print(f"  prefix_attn layers: {len(prefix_attn)}")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 5: Build result dict ─────────────────────────────────────────────────
step("Build prefix dict")
try:
    from viz.dashboard.loader import TEXT_START_IDX, TOTAL_IMAGE_TOKENS

    prefix = {}
    seq_len = 0
    if prefix_attn:
        seq_len = next(iter(prefix_attn.values())).shape[-1]
    for layer_idx, attn in prefix_attn.items():
        attn_np = np.array(attn)
        if attn_np.ndim == 4:
            attn_np = attn_np[0]
        attn_np = attn_np.astype(np.float32)
        t2i = attn_np[:, TEXT_START_IDX: TEXT_START_IDX + n_real_tokens, :TOTAL_IMAGE_TOKENS]
        prefix[f"layer_{layer_idx}"] = {"text_to_img": t2i, "full": attn_np}
    print(f"  {len(prefix)} layers, seq_len={seq_len}")
    if prefix:
        sample = prefix["layer_0"]["text_to_img"]
        print(f"  text_to_img shape: {sample.shape}  (heads, n_text, 512)")
except Exception:
    traceback.print_exc()
    sys.exit(1)


# ── Step 6: Visualize & save ─────────────────────────────────────────────────
step("Visualize attention & save to attn_jax/")
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    frame_dir = os.path.join(SAVE_DIR, f"frame_{FRAME_IDX:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Resize images for display
    ext_224 = cv2.resize(ext_img, (224, 224))
    wrist_224 = cv2.resize(wrist_img, (224, 224))

    # --- A) Per-token summary at layer 7 (mean across heads) ---
    for camera, cam_img in [("exterior", ext_224), ("wrist", wrist_224)]:
        summary = make_token_summary(prefix, cam_img, camera, token_texts, layer=7)
        path = os.path.join(frame_dir, f"token_summary_L7_{camera}.png")
        cv2.imwrite(path, summary)
        print(f"  Saved: {path}")

    # --- B) Layer x Head grid for key tokens ---
    vis_layers = [0, 3, 5, 7, 10, 14, 17]
    key_tokens = [0, 3, 5, 10]  # bos, "place", "duck", "bowl" (approx)
    # Find actual indices for interesting words
    for i, t in enumerate(token_texts):
        t_clean = t.strip("▁").lower()
        if t_clean == "duck" and i not in key_tokens:
            key_tokens.append(i)
        if t_clean == "bowl" and i not in key_tokens:
            key_tokens.append(i)
        if t_clean == "pink" and i not in key_tokens:
            key_tokens.append(i)

    for camera, cam_img in [("exterior", ext_224), ("wrist", wrist_224)]:
        for tok_idx in key_tokens:
            if tok_idx >= n_real_tokens:
                continue
            label = token_texts[tok_idx].strip("▁") if tok_idx < len(token_texts) else f"tok{tok_idx}"
            grid = make_layer_grid(prefix, tok_idx, cam_img, camera, vis_layers, label)
            # Add title
            title_bar = np.full((30, grid.shape[1], 3), 30, dtype=np.uint8)
            title = f'Token "{label}" (idx={tok_idx}) | {camera} | layers {vis_layers}'
            cv2.putText(title_bar, title, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            grid = np.vstack([title_bar, grid])

            path = os.path.join(frame_dir, f"grid_{camera}_tok{tok_idx:02d}_{label}.png")
            cv2.imwrite(path, grid)
            print(f"  Saved: {path}")

    # --- C) Full attention matrix (layer 7, head 0) ---
    for layer in [7, 14]:
        for head in [0, 4]:
            mat_img = make_full_attn_matrix(prefix, layer, head)
            path = os.path.join(frame_dir, f"attn_matrix_L{layer}_H{head}.png")
            cv2.imwrite(path, mat_img)
            print(f"  Saved: {path}")

    # --- D) Show interactive cv2 window (skip if no display) ---
    has_display = os.environ.get("DISPLAY") is not None
    if has_display:
        print(f"\n  Opening cv2 window (press any key to cycle, 'q' to quit)...")

        duck_idx = next((i for i, t in enumerate(token_texts) if "duck" in t.lower()), 3)
        show_tokens = [duck_idx] + [i for i in key_tokens if i != duck_idx and i < n_real_tokens]
        idx_ptr = 0

        def _draw(tok_idx):
            grid = make_layer_grid(prefix, tok_idx, ext_224, "exterior", vis_layers)
            bar = np.full((30, grid.shape[1], 3), 30, dtype=np.uint8)
            label = token_texts[tok_idx].strip("▁") if tok_idx < len(token_texts) else f"tok{tok_idx}"
            cv2.putText(bar, f'Token "{label}" (idx={tok_idx}) | exterior | layers {vis_layers}',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            return np.vstack([bar, grid])

        try:
            cv2.imshow("JAX Attention", _draw(show_tokens[0]))
            print(f"  Showing token '{token_texts[show_tokens[0]]}' — press key to cycle, 'q' to quit")
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    break
                idx_ptr = (idx_ptr + 1) % len(show_tokens)
                cv2.imshow("JAX Attention", _draw(show_tokens[idx_ptr]))
            cv2.destroyAllWindows()
        except cv2.error:
            print("  cv2.imshow not available (no GTK/display). Skipping interactive window.")
    else:
        print(f"\n  No DISPLAY set — skipping cv2 window. View saved images in {frame_dir}/")

except Exception:
    traceback.print_exc()
    sys.exit(1)


print(f"\n{'='*60}")
print(f"  ALL DONE — results saved to {SAVE_DIR}")
print(f"{'='*60}")
