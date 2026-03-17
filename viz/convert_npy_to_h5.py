"""Convert existing attn/*.npy attention maps to compressed HDF5 format.

Source layout:
    attn/{checkpoint_id}/layers_prefix/attn_map_layer_{i}.npy   shape (1, 8, seq, seq)

Output layout:
    attn_h5/{checkpoint_id}/{episode_name}/{frame_idx:05d}.h5

Usage:
    python viz/convert_npy_to_h5.py --src attn/ --dst attn_h5/
    python viz/convert_npy_to_h5.py --src attn/ --dst attn_h5/ --episode duck_0 --frame 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


# ── Token layout (Pi05/DROID) ─────────────────────────────────────────────────
NUM_IMAGE_TOKENS = 256       # 16×16 patches per camera
NUM_CAMERAS_REAL = 2         # exterior + wrist (real pixel content)
NUM_CAMERAS_TOTAL = 3        # includes zero-padded dummy slot
TOTAL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_REAL   # 512
ALL_IMAGE_TOKENS = NUM_IMAGE_TOKENS * NUM_CAMERAS_TOTAL    # 768
TEXT_START_IDX = ALL_IMAGE_TOKENS                          # 768

# Layers to also save full (seq×seq) attention matrix
FULL_MATRIX_LAYERS = {1, 4, 5, 7, 10}
NUM_LAYERS = 18


def load_image_224(path: str | None) -> np.ndarray | None:
    """Load and resize image to 224×224 RGB uint8."""
    if path is None or not os.path.exists(path):
        return None
    img = np.array(Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR))
    return img.astype(np.uint8)


def tokenize_instruction(instruction: str) -> tuple[list[int], list[str]]:
    """Tokenize instruction with zero state to get real token texts."""
    import sys
    project_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer()
        # Use zeros for state — instruction tokens will be exact, state labels approximate
        state = np.zeros(8)
        disc = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, disc))
        full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "
        ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        texts = [tokenizer._tokenizer.id_to_piece(i) for i in ids]
        return ids, texts
    except Exception as e:
        print(f"[convert] Tokenizer unavailable ({e}), using placeholder labels.")
        return [], []


def convert_episode(
    src_prefix_dir: str,
    dst_h5_path: str,
    ext_img_path: str | None = None,
    wrist_img_path: str | None = None,
    instruction: str = "",
    token_texts: list[str] | None = None,
    token_ids: list[int] | None = None,
    frame_idx: int = 0,
) -> None:
    """Convert one episode's attention .npy files to a single .h5 file."""

    os.makedirs(os.path.dirname(dst_h5_path), exist_ok=True)

    # Detect seq_len from first available layer
    seq_len = None
    for i in range(NUM_LAYERS):
        p = os.path.join(src_prefix_dir, f"attn_map_layer_{i}.npy")
        if os.path.exists(p):
            arr = np.load(p, mmap_mode="r")
            seq_len = arr.shape[-1]
            break

    if seq_len is None:
        print(f"[convert] No attention files found in {src_prefix_dir}", file=sys.stderr)
        return

    n_text = seq_len - TEXT_START_IDX

    # Build token metadata — tokenize if instruction given but texts not provided
    if token_texts is None and instruction:
        real_ids, real_texts = tokenize_instruction(instruction)
        if real_texts:
            token_ids = real_ids
            token_texts = real_texts
    if token_texts is None:
        token_texts = [f"tok_{i}" for i in range(n_text)]
    if token_ids is None:
        token_ids = list(range(n_text))

    n_real_tokens = min(len(token_texts), n_text)
    n_text_actual = min(n_text, len(token_texts))

    with h5py.File(dst_h5_path, "w") as f:
        # ── /meta ──────────────────────────────────────────────────────────
        meta = f.create_group("meta")
        meta.create_dataset("prefix_len", data=np.int32(TEXT_START_IDX))
        meta.create_dataset("frame_idx", data=np.int32(frame_idx))
        meta.create_dataset("seq_len", data=np.int32(seq_len))
        meta.create_dataset("n_real_tokens", data=np.int32(n_real_tokens))
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("instruction", data=instruction)
        meta.create_dataset(
            "token_texts",
            data=np.array(token_texts[:n_text_actual], dtype=object),
            dtype=dt,
        )
        meta.create_dataset(
            "token_ids",
            data=np.array(token_ids[:n_text_actual], dtype=np.int32),
        )

        # ── /images ────────────────────────────────────────────────────────
        imgs = f.create_group("images")
        ext_arr = load_image_224(ext_img_path)
        wrist_arr = load_image_224(wrist_img_path)
        if ext_arr is not None:
            imgs.create_dataset("exterior", data=ext_arr, compression="gzip", compression_opts=4)
        if wrist_arr is not None:
            imgs.create_dataset("wrist", data=wrist_arr, compression="gzip", compression_opts=4)

        # ── /prefix ────────────────────────────────────────────────────────
        prefix_grp = f.create_group("prefix")

        for i in range(NUM_LAYERS):
            npy_path = os.path.join(src_prefix_dir, f"attn_map_layer_{i}.npy")
            if not os.path.exists(npy_path):
                continue

            print(f"  layer {i:2d} ... ", end="", flush=True)
            attn = np.load(npy_path)
            if attn.ndim == 4:
                attn = attn[0]  # remove batch → (8, seq, seq)
            attn = attn.astype(np.float32)

            layer_grp = prefix_grp.create_group(f"layer_{i}")

            # text_to_img: (8, n_text, 512) float32
            t2i = attn[:, TEXT_START_IDX:TEXT_START_IDX + n_text_actual, :TOTAL_IMAGE_TOKENS]
            t2i_f32 = t2i.astype(np.float32)
            layer_grp.create_dataset(
                "text_to_img",
                data=t2i_f32,
                compression="gzip",
                compression_opts=4,
                chunks=(1, min(64, n_text_actual), TOTAL_IMAGE_TOKENS),
            )
            print(f"text_to_img {t2i_f32.shape}", end="")

            # full matrix for key layers
            if i in FULL_MATRIX_LAYERS:
                full_f32 = attn.astype(np.float32)
                chunk_s = min(256, seq_len)
                layer_grp.create_dataset(
                    "full",
                    data=full_f32,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, chunk_s, chunk_s),
                )
                print(f"  full {full_f32.shape}", end="")

            print()

        print(f"[convert] Written → {dst_h5_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert attn .npy → HDF5")
    parser.add_argument("--src", default="attn", help="Source attn/ root directory")
    parser.add_argument("--dst", default="attn_h5", help="Destination attn_h5/ root directory")
    parser.add_argument("--episode", default="episode_0", help="Episode name label")
    parser.add_argument("--frame", type=int, default=0, help="Frame index label")
    parser.add_argument(
        "--ext-img",
        default=None,
        help="Path to exterior camera image (224×224 RGB jpeg)",
    )
    parser.add_argument(
        "--wrist-img",
        default=None,
        help="Path to wrist camera image (224×224 RGB jpeg)",
    )
    parser.add_argument("--instruction", default="", help="Text instruction for this episode")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    # Auto-detect ext/wrist images from duck dataset if not provided
    default_ext = "data/visualization/duck/frames/varied_camera_1/00000.jpg"
    default_wrist = "data/visualization/duck/frames/hand_camera/00000.jpg"
    ext_img = args.ext_img or (default_ext if os.path.exists(default_ext) else None)
    wrist_img = args.wrist_img or (default_wrist if os.path.exists(default_wrist) else None)

    checkpoints = sorted([d.name for d in src_root.iterdir() if d.is_dir()])
    if not checkpoints:
        print(f"[convert] No checkpoint directories found in {src_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found checkpoints: {checkpoints}")
    for ckpt in checkpoints:
        prefix_dir = src_root / ckpt / "layers_prefix"
        if not prefix_dir.exists():
            print(f"[convert] Skipping {ckpt}: no layers_prefix/ directory")
            continue

        dst_h5 = dst_root / ckpt / args.episode / f"{args.frame:05d}.h5"
        print(f"\n[convert] Checkpoint {ckpt} → {dst_h5}")
        convert_episode(
            src_prefix_dir=str(prefix_dir),
            dst_h5_path=str(dst_h5),
            ext_img_path=ext_img,
            wrist_img_path=wrist_img,
            instruction=args.instruction,
            frame_idx=args.frame,
        )

    print("\n[convert] Done.")


if __name__ == "__main__":
    main()
