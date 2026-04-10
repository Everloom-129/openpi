"""Write prefix-attention data to a compressed HDF5 file.

Produces the schema expected by viz/dashboard/loader.py:

    /meta
        prefix_len      int32   — TEXT_START_IDX (768)
        frame_idx       int32
        seq_len         int32
        n_real_tokens   int32
        instruction     str
        token_texts     str[]
        token_ids       int32[]
    /images
        exterior        uint8[224,224,3]   gzip-4
        wrist           uint8[224,224,3]   gzip-4
    /prefix
        layer_{i}/
            text_to_img float32[8, n_text, 512]   gzip-4   (every layer)
            full        float32[8, seq_len, seq_len] gzip-4  (every layer)
    /gt_action          float32[8, action_dim]   gzip-4
        rows = OPEN_LOOP_HORIZON action steps (NaN-padded at end of episode)
        cols = [joint_velocity×7, gripper_position×1]  (action_dim=8)
    /pred_action        float32[8, action_dim]   gzip-4
        Pi0.5 predicted actions for the same horizon (DroidOutputs[:, :8])
    /suffix_denoising
        n_steps         int32
        action_traj     float32[n_steps, action_horizon, action_dim]   gzip-4
            x_t after each Euler step; index 0 = mostly noise, index -1 = clean action
        layer_{i}/
            action_to_img_steps  float32[n_steps, 8, 512]   gzip-4
            group_masses         float32[n_steps, 4]         gzip-4

Primary entry point (pipeline.py):
    from attn_h5_writer import write_attn_h5_from_buffer

    ok = write_attn_h5_from_buffer(
        attn_buffer=_gpt.get_attn_buffer(),   # dict[int, ndarray]
        h5_path=frame_dir / "00000.h5",
        ext_img=example["observation/exterior_image_1_left"],
        wrist_img=example["observation/wrist_image_left"],
        instruction=example["prompt"],
        frame_idx=0,
    )

Legacy entry point (convert_npy_to_h5.py):
    from attn_h5_writer import write_attn_h5
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# ── Token-layout constants (must match loader.py) ─────────────────────────────
NUM_IMAGE_TOKENS = 256       # 16×16 patches per camera
TOTAL_IMAGE_TOKENS = 512     # ext (0:256) + wrist (256:512)
ALL_IMAGE_TOKENS = 768       # includes zero-padded dummy camera slot
TEXT_START_IDX = ALL_IMAGE_TOKENS
NUM_LAYERS = 18


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resize_224(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    return np.array(pil.resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def _tokenize_instruction(instruction: str, is_pi05: bool = True) -> tuple[list[int], list[str]]:
    """Tokenize *instruction* for token-label metadata.

    For pi0.5 (is_pi05=True): uses "Task: ..., State: ...;\\nAction: " format with zeroed state.
    For pi0   (is_pi05=False): uses plain instruction + "\\n" (state is a continuous suffix token).
    Falls back to placeholder labels if the tokenizer is unavailable.
    """
    try:
        import sys
        src_dir = str(Path(__file__).resolve().parent.parent / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from openpi.models.tokenizer import PaligemmaTokenizer

        tokenizer = PaligemmaTokenizer()
        if is_pi05:
            state = np.zeros(8)
            disc = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, disc))
            full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "
            ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # pi0: instruction only; state is injected as a continuous suffix token
            cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
            ids = tokenizer._tokenizer.encode(cleaned, add_bos=True) + tokenizer._tokenizer.encode("\n")
        texts = [tokenizer._tokenizer.id_to_piece(i) for i in ids]
        return ids, texts
    except Exception:
        return [], []


def _write_h5_core(
    attn_arrays: dict[int, np.ndarray],
    h5_path: Path,
    ext_img: np.ndarray | None,
    wrist_img: np.ndarray | None,
    instruction: str,
    frame_idx: int,
    suffix_arrays: dict[int, np.ndarray] | None = None,
    suffix_steps_arrays: list[dict[int, np.ndarray]] | None = None,
    gt_action: np.ndarray | None = None,
    pred_action: np.ndarray | None = None,
    action_traj: list[np.ndarray] | None = None,
    is_pi05: bool = True,
) -> bool:
    """Write a single HDF5 from an in-memory layer dict.

    *attn_arrays* maps layer_idx → ndarray of shape (1,8,seq,seq) or (8,seq,seq).
    Saves full seq×seq matrix and text→image slice for every layer present.
    Returns False if the dict is empty.
    """
    if not attn_arrays:
        return False

    # Detect seq_len from the first entry
    first = next(iter(attn_arrays.values()))
    seq_len = int(first.shape[-1])
    n_text = seq_len - TEXT_START_IDX

    # Tokenize
    token_ids, token_texts = _tokenize_instruction(instruction, is_pi05=is_pi05)
    if not token_texts:
        token_texts = [f"tok_{i}" for i in range(n_text)]
        token_ids = list(range(n_text))
    n_text_actual = min(n_text, len(token_texts))

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:

        # /meta
        meta = f.create_group("meta")
        meta.create_dataset("prefix_len",    data=np.int32(TEXT_START_IDX))
        meta.create_dataset("frame_idx",     data=np.int32(frame_idx))
        meta.create_dataset("seq_len",       data=np.int32(seq_len))
        meta.create_dataset("n_real_tokens", data=np.int32(n_text_actual))
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

        # /images
        imgs = f.create_group("images")
        if ext_img is not None:
            imgs.create_dataset(
                "exterior", data=_resize_224(ext_img), compression="gzip", compression_opts=4
            )
        if wrist_img is not None:
            imgs.create_dataset(
                "wrist", data=_resize_224(wrist_img), compression="gzip", compression_opts=4
            )

        # /prefix  — every layer, full matrix + text→image slice
        prefix_grp = f.create_group("prefix")
        chunk_s = min(256, seq_len)

        for layer_idx in sorted(attn_arrays):
            attn = attn_arrays[layer_idx]
            if attn.ndim == 4:
                attn = attn[0]              # (8, seq, seq)
            attn = attn.astype(np.float32)

            layer_grp = prefix_grp.create_group(f"layer_{layer_idx}")

            # full seq×seq matrix
            layer_grp.create_dataset(
                "full",
                data=attn,
                compression="gzip",
                compression_opts=4,
                chunks=(1, chunk_s, chunk_s),
            )

            # text→image slice: (8, n_text_actual, TOTAL_IMAGE_TOKENS)
            t2i = attn[
                :,
                TEXT_START_IDX : TEXT_START_IDX + n_text_actual,
                :TOTAL_IMAGE_TOKENS,
            ]
            layer_grp.create_dataset(
                "text_to_img",
                data=t2i,
                compression="gzip",
                compression_opts=4,
                chunks=(1, min(64, max(1, n_text_actual)), TOTAL_IMAGE_TOKENS),
            )

        # /gt_action  — ground-truth actions from trajectory.h5
        # shape: (OPEN_LOOP_HORIZON, action_dim)  e.g. (8, 8) = [joint_vel×7, gripper×1]
        # NaN-padded if the frame is within OPEN_LOOP_HORIZON of episode end.
        if gt_action is not None:
            f.create_dataset(
                "gt_action",
                data=gt_action.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )

        # /pred_action  — Pi0.5 predicted actions for this frame
        if pred_action is not None:
            f.create_dataset(
                "pred_action",
                data=np.asarray(pred_action, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )

        # /suffix  — action-token → image attention (one slice per layer)
        # suffix_arrays[layer] shape: (1, n_heads, 8_steps, prefix_seq+8) or (n_heads, 8_steps, k)
        if suffix_arrays:
            suffix_grp = f.create_group("suffix")
            for layer_idx in sorted(suffix_arrays):
                sa = suffix_arrays[layer_idx]
                if sa.ndim == 4:
                    sa = sa[0]                          # drop batch dim → (n_heads, 8, k)
                sa = sa.astype(np.float32)
                # Columns 0:TOTAL_IMAGE_TOKENS are image-patch positions (ext 0:256, wrist 256:512)
                a2i = sa[:, :, :TOTAL_IMAGE_TOKENS]     # (n_heads, 8_steps, 512)
                sg = suffix_grp.create_group(f"layer_{layer_idx}")
                sg.create_dataset(
                    "action_to_img",
                    data=a2i,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, 8, TOTAL_IMAGE_TOKENS),
                )

        # /suffix_denoising  — per-NFE-step action attention (compact)
        # Stores mean-over-heads attention for every denoising step so the
        # dashboard can plot attention-mass trajectories without re-running inference.
        #
        # Schema per layer:
        #   action_to_img_steps  float32(n_steps, 8_action, 512_patches)
        #   group_masses         float32(n_steps, 4)  — [ext, wrist, text, action_self]
        if suffix_steps_arrays:
            n_steps = len(suffix_steps_arrays)
            text_end = TEXT_START_IDX + n_text_actual
            sd_grp = f.create_group("suffix_denoising")
            sd_grp.create_dataset("n_steps", data=np.int32(n_steps))

            # Gather layer indices from first step
            layer_indices = sorted(suffix_steps_arrays[0].keys())
            for layer_idx in layer_indices:
                img_steps   = []   # (n_steps, 8, 512)
                mass_steps  = []   # (n_steps, 4)
                for step_dict in suffix_steps_arrays:
                    sa = step_dict[layer_idx]
                    if sa.ndim == 4:
                        sa = sa[0]              # (n_heads, 8, seq_len)
                    sa = sa.astype(np.float32)
                    mean_h = sa.mean(axis=0)    # (8, seq_len)
                    img_steps.append(mean_h[:, :TOTAL_IMAGE_TOKENS])  # (8, 512)
                    # Group masses — mean over action steps then sum per region
                    mean_ha = mean_h.mean(axis=0)   # (seq_len,)
                    mass_steps.append([
                        float(mean_ha[:NUM_IMAGE_TOKENS].sum()),                    # ext
                        float(mean_ha[NUM_IMAGE_TOKENS:TOTAL_IMAGE_TOKENS].sum()),  # wrist
                        float(mean_ha[TEXT_START_IDX:text_end].sum()),              # text
                        float(mean_ha[text_end:].sum()),                            # action self
                    ])

                lg = sd_grp.create_group(f"layer_{layer_idx}")
                lg.create_dataset(
                    "action_to_img_steps",
                    data=np.array(img_steps,  dtype=np.float32),   # (n_steps, 8, 512)
                    compression="gzip", compression_opts=4,
                    chunks=(1, 8, TOTAL_IMAGE_TOKENS),
                )
                lg.create_dataset(
                    "group_masses",
                    data=np.array(mass_steps, dtype=np.float32),   # (n_steps, 4)
                    compression="gzip", compression_opts=4,
                )

            # /suffix_denoising/action_traj — x_t after each Euler step
            # shape: (n_steps, action_horizon, action_dim)  e.g. (10, 15, 8)
            # index 0 = after 1st step (mostly noise), index -1 = final clean action
            if action_traj:
                traj_arr = np.stack(action_traj, axis=0).astype(np.float32)
                sd_grp.create_dataset(
                    "action_traj",
                    data=traj_arr,
                    compression="gzip", compression_opts=4,
                )

    return True


# ── Public API ─────────────────────────────────────────────────────────────────

def write_attn_h5_from_buffer(
    attn_buffer: dict[int, np.ndarray],
    h5_path: str | Path,
    ext_img: np.ndarray | None = None,
    wrist_img: np.ndarray | None = None,
    instruction: str = "",
    frame_idx: int = 0,
    suffix_attn_buffer: dict[int, np.ndarray] | None = None,
    suffix_steps_buffer: list[dict[int, np.ndarray]] | None = None,
    gt_action: np.ndarray | None = None,
    pred_action: np.ndarray | None = None,
    action_traj: list[np.ndarray] | None = None,
    is_pi05: bool = True,
) -> bool:
    """Write HDF5 directly from in-RAM attention buffers (primary API).

    *attn_buffer* — prefix attention from ``gemma_pytorch.get_attn_buffer()``.
    *suffix_attn_buffer* — averaged action attention from ``get_suffix_attn_buffer()``;
        writes ``/suffix/layer_{i}/action_to_img``.
    *suffix_steps_buffer* — per-NFE-step list from ``get_suffix_attn_steps_buffer()``;
        writes compact ``/suffix_denoising/`` group (group_masses + action_to_img_steps).
        Does not affect /prefix or /suffix groups.
    *gt_action* — float32(OPEN_LOOP_HORIZON, action_dim) from trajectory.h5.
    *pred_action* — ``result["actions"]`` from ``policy.infer()``.
    *action_traj* — list of float32(action_horizon, action_dim) from
        ``get_action_traj_buffer()``, one entry per Euler step; written to
        ``/suffix_denoising/action_traj`` as float32(n_steps, action_horizon, action_dim).
    *is_pi05* — selects tokenization format for token-label metadata.
    """
    return _write_h5_core(
        attn_buffer, Path(h5_path), ext_img, wrist_img, instruction, frame_idx,
        suffix_arrays=suffix_attn_buffer or {},
        suffix_steps_arrays=suffix_steps_buffer or [],
        gt_action=gt_action,
        pred_action=pred_action,
        action_traj=action_traj,
        is_pi05=is_pi05,
    )


def write_attn_h5(
    attn_npy_dir: str | Path,
    h5_path: str | Path,
    ext_img: np.ndarray | None = None,
    wrist_img: np.ndarray | None = None,
    instruction: str = "",
    frame_idx: int = 0,
) -> bool:
    """Legacy path-based API: load .npy files then call _write_h5_core.

    Used by convert_npy_to_h5.py for one-off conversions of existing NPY dumps.
    """
    attn_npy_dir = Path(attn_npy_dir)
    attn_arrays: dict[int, np.ndarray] = {}
    for i in range(NUM_LAYERS):
        p = attn_npy_dir / f"attn_map_layer_{i}.npy"
        if p.exists():
            attn_arrays[i] = np.load(p)
    return _write_h5_core(
        attn_arrays, Path(h5_path), ext_img, wrist_img, instruction, frame_idx
    )
