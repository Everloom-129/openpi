"""H7.1 — Word-Specific Prefix Attention Visualization.

For each target word (e.g. "cube", "bowl", "place"), finds its subword token
indices in the full prompt, extracts the corresponding rows of the prefix
attention matrix, and overlays the aggregated attention as a heatmap on both
camera images.

Usage (standalone):
    python viz/h_word_attention.py
"""
from __future__ import annotations

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import jax.numpy as jnp
from openpi.shared import image_tools


# ── Token layout constants ────────────────────────────────────────────────────
NUM_IMAGE_TOKENS = 256          # patches per camera  (16×16)
# Pi05/DROID uses 3 image slots: exterior, wrist, and a zero-padded dummy camera
# (right_wrist_0_rgb = np.zeros_like, image_mask=False).  All three occupy 256
# tokens each in the attention sequence, so text tokens start at 768, not 512.
NUM_CAMERAS_REAL = 2            # exterior + wrist (have real pixel content)
NUM_CAMERAS_TOTAL = 3           # includes the masked zero-padding slot
TOTAL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_REAL   # 512  — real cameras only (for vis)
ALL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_TOTAL    # 768  — full image region in sequence
TEXT_START_IDX = ALL_IMAGE_TOKENS                          # 768  — where text tokens begin


# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def build_full_prompt(example: dict) -> str:
    """Reconstruct the exact prompt the model sees."""
    joint_pos = example.get("observation/joint_position", np.zeros(7))
    gripper_pos = example.get("observation/gripper_position", np.zeros(1))
    state = np.concatenate([joint_pos, gripper_pos])
    instruction = example["prompt"].strip().replace("_", " ").replace("\n", " ")
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    return f"Task: {instruction}, State: {state_str};\nAction: "


def tokenize_prompt(example: dict):
    """Return (token_ids, token_texts) for the full prompt."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer()
    full_prompt = build_full_prompt(example)
    token_ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
    token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in token_ids]
    return token_ids, token_texts


def find_word_token_indices(token_texts: list[str], target_word: str) -> list[int]:
    """Find local token indices (within the text tokens) for a target word.

    SentencePiece uses "▁" (U+2581) as a word-boundary prefix.  We reconstruct
    the text and map character positions back to token indices.

    Returns local indices relative to the *text* token region (i.e. index 0 =
    first token after the image tokens).  Add TEXT_START_IDX to get global
    attention-matrix row indices.
    """
    # Build reconstructed string and char→token map
    char_to_tok: list[int] = []
    text = ""
    for tok_idx, tok in enumerate(token_texts):
        if tok.startswith("▁"):
            piece = " " + tok[1:]
        else:
            piece = tok
        text += piece
        char_to_tok.extend([tok_idx] * len(piece))

    target_lower = target_word.lower()
    text_lower = text.lower()

    indices: list[int] = []
    pos = 0
    while True:
        start = text_lower.find(target_lower, pos)
        if start == -1:
            break
        end = start + len(target_lower)
        indices.extend(char_to_tok[start:end])
        pos = end

    # deduplicate, preserve order
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


# ── Attention loading ─────────────────────────────────────────────────────────

def load_prefix_attn(layer_idx: int, input_dir: str = "results/layers_prefix") -> np.ndarray | None:
    """Load prefix attention map → [heads, seq_len, seq_len]."""
    path = os.path.join(input_dir, f"attn_map_layer_{layer_idx}.npy")
    if not os.path.exists(path):
        print(f"[h_word_attention] Attention file not found: {path}")
        return None
    attn = np.load(path)
    if attn.ndim == 4:
        attn = attn[0]   # remove batch dim → [heads, seq, seq]
    return attn.astype(np.float32)


# ── Heatmap overlay ───────────────────────────────────────────────────────────

def overlay_heatmap(img_np: np.ndarray, heatmap_16x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (resized_img, blended_overlay), both uint8 RGB 224×224."""
    img_jax = jnp.array(img_np)
    img_224 = np.array(image_tools.resize_with_pad(img_jax, 224, 224)).astype(np.uint8)

    hmap = cv2.resize(heatmap_16x16.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    hmap_u8 = np.uint8(255 * hmap / (np.max(hmap) + 1e-8))
    hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    hmap_color = cv2.cvtColor(hmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_224, 0.6, hmap_color, 0.4, 0)
    return img_224, blended


# ── Main visualization function ───────────────────────────────────────────────

def visualize_word_attention(
    example: dict,
    layer_idx: int,
    target_words: list[str],
    input_dir: str = "results/layers_prefix",
    output_dir: str = "results",
    name: str = "default",
    head_agg: str = "max",   # "max" | "mean"
):
    """Visualize per-word attention heatmaps overlaid on camera images.

    For each target word the function:
      1. Finds the global attention-matrix rows for that word's subword tokens.
      2. Averages (over subwords) the attention to all image patches → [512].
      3. Splits into camera-1 [256] and camera-2 [256], reshapes to 16×16.
      4. Aggregates over attention heads (max or mean).
      5. Overlays as a heatmap on the original images.

    Args:
        example:      Dict with image arrays and "prompt" key.
        layer_idx:    Which transformer layer to visualize.
        target_words: List of words to analyse (e.g. ["cube", "bowl"]).
        input_dir:    Directory containing `attn_map_layer_{i}.npy` files.
        output_dir:   Root output directory.
        name:         Sub-folder name for this episode/frame.
        head_agg:     How to aggregate over attention heads ("max" or "mean").
    """
    # ── Load attention ───────────────────────────────────────────────────────
    attn = load_prefix_attn(layer_idx, input_dir)
    if attn is None:
        return

    _, seq_len, _ = attn.shape
    if seq_len <= TEXT_START_IDX:
        print(f"[h_word_attention] Sequence length {seq_len} ≤ {TEXT_START_IDX}; no text tokens.")
        return

    # ── Tokenize ─────────────────────────────────────────────────────────────
    try:
        _, token_texts = tokenize_prompt(example)
    except Exception as exc:
        print(f"[h_word_attention] Tokenizer failed: {exc}")
        return

    # Token texts cover the *full* prompt including BOS.  The global attention
    # rows for text tokens start at TEXT_START_IDX, so local token index 0
    # inside token_texts maps to attention row TEXT_START_IDX.
    text_tokens = token_texts   # all tokens; first 1 is BOS, rest is prompt

    # ── Per-word visualization ───────────────────────────────────────────────
    ext_img = example["observation/exterior_image_1_left"]
    wrist_img = example["observation/wrist_image_left"]

    n_words = len(target_words)
    # Layout: n_words rows × 4 cols  [ext_raw | ext_attn | wrist_raw | wrist_attn]
    fig, axes = plt.subplots(
        n_words + 1, 4,
        figsize=(16, 4 * (n_words + 1)),
        gridspec_kw={"wspace": 0.05, "hspace": 0.3},
    )
    if n_words + 1 == 1:
        axes = axes[np.newaxis, :]

    # Row 0: raw images (reference)
    ext_raw_224, _ = overlay_heatmap(ext_img, np.ones((16, 16)))
    wrist_raw_224, _ = overlay_heatmap(wrist_img, np.ones((16, 16)))
    for col, img in zip([0, 2], [ext_raw_224, wrist_raw_224]):
        axes[0, col].imshow(img)
        axes[0, col + 1].axis("off")
    axes[0, 0].set_title("Exterior (raw)", fontsize=10)
    axes[0, 2].set_title("Wrist (raw)", fontsize=10)
    axes[0, 1].axis("off")
    axes[0, 3].axis("off")
    axes[0, 0].axis("off")
    axes[0, 2].axis("off")

    word_results: dict[str, dict] = {}

    for row_idx, word in enumerate(target_words, start=1):
        local_indices = find_word_token_indices(text_tokens, word)
        if not local_indices:
            print(f"[h_word_attention] Word '{word}' not found in token list.")
            for col in range(4):
                axes[row_idx, col].axis("off")
                axes[row_idx, col].set_title(f"'{word}' — NOT FOUND", fontsize=9)
            continue

        # Global attention-matrix row indices for this word's tokens
        global_indices = [TEXT_START_IDX + li for li in local_indices]
        # Clamp to valid range
        global_indices = [gi for gi in global_indices if gi < seq_len]
        if not global_indices:
            print(f"[h_word_attention] Word '{word}' global indices out of range.")
            continue

        # Extract: [heads, n_word_tokens, TOTAL_IMAGE_TOKENS]
        word_attn = attn[:, global_indices, :TOTAL_IMAGE_TOKENS]  # [heads, k, 512]
        # Average over subword tokens
        word_attn = word_attn.mean(axis=1)   # [heads, 512]

        # Aggregate over heads
        if head_agg == "max":
            word_attn_agg = word_attn.max(axis=0)   # [512]
        else:
            word_attn_agg = word_attn.mean(axis=0)  # [512]

        ext_hmap = word_attn_agg[:NUM_IMAGE_TOKENS].reshape(16, 16)
        wrist_hmap = word_attn_agg[NUM_IMAGE_TOKENS:].reshape(16, 16)

        # Entropy (per head, averaged over this word's rows)
        word_entropy_per_head = -np.sum(
            word_attn * np.log(word_attn + 1e-10), axis=-1
        )  # [heads]
        mean_entropy = float(word_entropy_per_head.mean())

        word_results[word] = {
            "local_token_indices": local_indices,
            "global_token_indices": global_indices,
            "token_texts": [text_tokens[li] for li in local_indices if li < len(text_tokens)],
            "mean_entropy": mean_entropy,
        }

        # Overlay
        _, ext_overlay = overlay_heatmap(ext_img, ext_hmap)
        _, wrist_overlay = overlay_heatmap(wrist_img, wrist_hmap)

        token_str = " ".join(word_results[word]["token_texts"])
        row_title = f"'{word}'  tokens=[{token_str}]  H={mean_entropy:.2f}"

        axes[row_idx, 0].imshow(ext_raw_224); axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(ext_overlay); axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(wrist_raw_224); axes[row_idx, 2].axis("off")
        axes[row_idx, 3].imshow(wrist_overlay); axes[row_idx, 3].axis("off")
        axes[row_idx, 0].set_title(f"Ext raw", fontsize=8)
        axes[row_idx, 1].set_title(f"Ext attn — {row_title}", fontsize=8)
        axes[row_idx, 2].set_title(f"Wrist raw", fontsize=8)
        axes[row_idx, 3].set_title(f"Wrist attn", fontsize=8)

    plt.suptitle(
        f'H7.1 Word Attention  |  Layer {layer_idx}  |  "{example["prompt"]}"',
        fontsize=13, y=1.01,
    )

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    words_slug = "_".join(target_words)[:40]
    out_path = os.path.join(out_dir, f"h7_word_attn_L{layer_idx}_{words_slug}.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_word_attention] Saved → {out_path}")

    # ── Print summary ────────────────────────────────────────────────────────
    for word, info in word_results.items():
        print(
            f"  '{word}': tokens={info['token_texts']} "
            f"global_idx={info['global_token_indices']} "
            f"entropy={info['mean_entropy']:.3f}"
        )

    return word_results


# ── Entropy comparison helper ─────────────────────────────────────────────────

def compare_word_entropy(
    example: dict,
    target_words: list[str],
    layers: list[int],
    input_dir: str = "results/layers_prefix",
    output_dir: str = "results",
    name: str = "default",
):
    """Plot entropy vs layer for each target word.

    Low entropy = focused attention; high entropy = diffuse attention.
    """
    try:
        _, token_texts = tokenize_prompt(example)
    except Exception as exc:
        print(f"[h_word_attention] Tokenizer failed: {exc}")
        return

    entropies: dict[str, list[float]] = {w: [] for w in target_words}
    valid_layers: list[int] = []

    for layer_idx in layers:
        attn = load_prefix_attn(layer_idx, input_dir)
        if attn is None:
            continue
        valid_layers.append(layer_idx)
        _, seq_len, _ = attn.shape

        for word in target_words:
            local_indices = find_word_token_indices(token_texts, word)
            global_indices = [TEXT_START_IDX + li for li in local_indices if (TEXT_START_IDX + li) < seq_len]
            if not global_indices:
                entropies[word].append(float("nan"))
                continue
            word_attn = attn[:, global_indices, :TOTAL_IMAGE_TOKENS].mean(axis=1)  # [heads, 512]
            ent = -np.sum(word_attn * np.log(word_attn + 1e-10), axis=-1).mean()
            entropies[word].append(float(ent))

    if not valid_layers:
        print("[h_word_attention] No attention files found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for word, ent_vals in entropies.items():
        ax.plot(valid_layers, ent_vals, marker="o", label=f"'{word}'")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy (bits)")
    ax.set_title(f'H7.1 Word Entropy vs Layer  |  "{example["prompt"]}"')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "h7_word_entropy_vs_layer.jpg")
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[h_word_attention] Entropy plot saved → {out_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from attn_map import load_duck_example

    example = load_duck_example(camera="left", index=0)

    TARGET_WORDS = ["duck", "pink", "bowl", "place"]
    LAYERS = [1, 4, 5, 7, 10]
    INPUT_DIR = "results/layers_prefix"
    OUTPUT_DIR = "results"
    NAME = "duck_left_0"

    print("=== H7.1 Word Attention Visualization ===")
    print(f"Prompt: {example['prompt']}")
    print(f"Target words: {TARGET_WORDS}")
    print()

    for layer in LAYERS:
        visualize_word_attention(
            example,
            layer_idx=layer,
            target_words=TARGET_WORDS,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            name=NAME,
        )

    compare_word_entropy(
        example,
        target_words=TARGET_WORDS,
        layers=LAYERS,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        name=NAME,
    )
