"""Counterfactual Prompting Tab.

Given the same image pair, runs inference with N different prompts and
compares the resulting attention maps side by side.

Storage layout (no conflict between rollouts):
    attn_h5/{checkpoint}/{episode}/{frame:05d}_{prompt_slug}.h5

Each slug is derived from the prompt text so different prompts never
overwrite each other.  GPU-id folders (attn/0, attn/1) are internal
write targets that get converted on-the-fly — never exposed in the UI.

Isolated: no shared state with other tabs.
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st

from viz.dashboard import loader as _loader
from viz.dashboard.views.grid_heatmap import (
    _SIMPLE_AGGS,
    _PARAM_AGGS,
    _ALL_AGG_NAMES,
    _AGG_DESCRIPTIONS,
    _make_topk_fn,
    _make_count_fn,
)

NUM_IMAGE_TOKENS = 256
PATCH_GRID = 16
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


# ── Dataset catalogue ─────────────────────────────────────────────────────────

def _catalogue() -> dict[str, dict]:
    """Discover all episode datasets under data/example/.

    Returns a dict keyed by episode name. Each entry has:
      frames_root — directory containing hand_camera/, varied_camera_1/, varied_camera_2/
      traj_h5     — path to trajectory.h5
      default_prompt — from instruction.txt or ""
      n_frames    — number of frames in hand_camera/
    Camera (ext_dir) is NOT fixed here — resolved at render time from user selection.
    """
    example_dir = os.path.join(_PROJECT_ROOT, "data/example")
    cat: dict[str, dict] = {}
    if not os.path.isdir(example_dir):
        return cat

    for ep_name in sorted(os.listdir(example_dir)):
        ep_dir = os.path.join(example_dir, ep_name)
        if not os.path.isdir(ep_dir):
            continue

        # Auto-detect frame structure
        if os.path.isdir(os.path.join(ep_dir, "recordings", "frames")):
            frames_root = os.path.join(ep_dir, "recordings", "frames")
        elif os.path.isdir(os.path.join(ep_dir, "frames")):
            frames_root = os.path.join(ep_dir, "frames")
        else:
            continue

        hand_dir = os.path.join(frames_root, "hand_camera")
        if not os.path.isdir(hand_dir):
            continue

        n = len([f for f in os.listdir(hand_dir) if f.endswith(".jpg")])
        if n == 0:
            continue

        traj_h5 = os.path.join(ep_dir, "trajectory.h5")
        instr_path = os.path.join(ep_dir, "instruction.txt")
        default_prompt = open(instr_path).read().strip() if os.path.exists(instr_path) else ""

        cat[ep_name] = {
            "frames_root": frames_root,
            "traj_h5": traj_h5,
            "default_prompt": default_prompt,
            "n_frames": n,
        }

    return cat


_CAMERA_DIR_MAP = {"right": "varied_camera_2", "left": "varied_camera_1"}


def _resolve_ext_camera_dir(frames_root: str, camera: str) -> str:
    """Return the subdirectory name for the requested exterior camera.

    Falls back gracefully: if varied_camera_2/1 don't exist, use any dir
    that isn't hand_camera.
    """
    preferred = _CAMERA_DIR_MAP.get(camera, "varied_camera_2")
    if os.path.isdir(os.path.join(frames_root, preferred)):
        return preferred
    # Fallback: first non-hand_camera dir
    for name in sorted(os.listdir(frames_root)):
        if name != "hand_camera" and os.path.isdir(os.path.join(frames_root, name)):
            return name
    return preferred  # let caller raise a clear error


def _load_frame(
    dataset: dict, frame_idx: int, camera: str = "right"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ext_img, wrist_img, joint_pos, gripper_pos).

    Args:
        dataset: catalogue entry with keys frames_root, traj_h5
        frame_idx: frame index
        camera: "right" or "left" exterior camera selection
    """
    from PIL import Image
    import h5py

    frames_root = dataset["frames_root"]
    ext_cam_dir = _resolve_ext_camera_dir(frames_root, camera)
    ext_path = os.path.join(frames_root, ext_cam_dir, f"{frame_idx:05d}.jpg")
    wrist_path = os.path.join(frames_root, "hand_camera", f"{frame_idx:05d}.jpg")
    ext_img = np.array(Image.open(ext_path).convert("RGB"))
    wrist_img = np.array(Image.open(wrist_path).convert("RGB"))

    with h5py.File(dataset["traj_h5"], "r") as f:
        joint_pos = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gripper_pos = f["observation/robot_state/gripper_position"][frame_idx : frame_idx + 1].astype(np.float64)

    return ext_img, wrist_img, joint_pos, gripper_pos


# ── Inference helpers ─────────────────────────────────────────────────────────

def _run_one_prompt(
    policy,
    ext_img: np.ndarray,
    wrist_img: np.ndarray,
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    prompt: str,
    gpu_id: int = 0,
) -> dict:
    """Run inference for one prompt, return in-memory slice dict.

    Uses the gemma_pytorch RAM buffer — no npy files written to disk.
    The raw attention buffer is stashed in slice_dict["_attn_buffer"] so
    _save_cf_h5 can write it directly via write_attn_h5_from_buffer.
    """
    from openpi.models_pytorch import gemma_pytorch as _gpt
    from viz.dashboard.loader import TEXT_START_IDX, TOTAL_IMAGE_TOKENS
    from PIL import Image as _PIL

    example = {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": prompt,
    }

    _gpt.enable_attn_buffer()
    try:
        _ = policy.infer(example)
        buf = _gpt.get_attn_buffer()
    finally:
        _gpt.clear_attn_buffer()

    if not buf:
        return {}

    first = next(iter(buf.values()))
    seq_len = int(first.shape[-1])
    n_text = seq_len - TEXT_START_IDX

    is_pi05 = bool(getattr(getattr(policy, "_model", None), "pi05", True))

    token_texts = [f"tok_{i}" for i in range(n_text)]
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer()
        instr = prompt.strip().replace("_", " ").replace("\n", " ")
        if is_pi05:
            state = np.concatenate([joint_pos, gripper_pos])
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_str = " ".join(map(str, discretized))
            full_prompt = f"Task: {instr}, State: {state_str};\nAction: "
            token_ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
        else:
            token_ids = tokenizer._tokenizer.encode(instr, add_bos=True) + tokenizer._tokenizer.encode("\n")
        token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in token_ids]
    except Exception:
        pass

    n_text_actual = min(n_text, len(token_texts))

    def _to_224(img):
        if img is None:
            return None
        return np.array(
            _PIL.fromarray(img.astype(np.uint8)).resize((224, 224), _PIL.BILINEAR),
            dtype=np.uint8,
        )

    prefix = {}
    for layer_idx, attn in buf.items():
        if attn.ndim == 4:
            attn = attn[0]
        attn = attn.astype(np.float32)
        t2i = attn[:, TEXT_START_IDX : TEXT_START_IDX + n_text_actual, :TOTAL_IMAGE_TOKENS]
        prefix[f"layer_{layer_idx}"] = {"text_to_img": t2i, "full": attn}

    return {
        "meta": {
            "prefix_len": TEXT_START_IDX,
            "seq_len": seq_len,
            "n_real_tokens": n_text_actual,
            "instruction": prompt,
            "token_texts": token_texts[:n_text_actual],
        },
        "images": {
            "exterior": _to_224(ext_img),
            "wrist":    _to_224(wrist_img),
        },
        "prefix": prefix,
        "_attn_buffer": buf,   # kept for _save_cf_h5
        "_is_pi05": is_pi05,   # kept for _save_cf_h5
    }


def _save_cf_h5(
    slice_dict: dict,
    checkpoint_id: str,
    episode_id: str,
    frame_idx: int,
    prompt_slug: str,
    attn_h5_root: str,
) -> str:
    """Persist one counterfactual rollout to disk as HDF5."""
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
    from attn_h5_writer import write_attn_h5_from_buffer  # noqa: PLC0415

    buf = slice_dict.get("_attn_buffer", {})
    images = slice_dict.get("images", {})
    instruction = slice_dict.get("meta", {}).get("instruction", "")
    is_pi05 = slice_dict.get("_is_pi05", True)
    dst = _loader.h5_path_cf(checkpoint_id, episode_id, frame_idx, prompt_slug, attn_h5_root)

    write_attn_h5_from_buffer(
        attn_buffer=buf,
        h5_path=dst,
        ext_img=images.get("exterior"),
        wrist_img=images.get("wrist"),
        instruction=instruction,
        frame_idx=frame_idx,
        is_pi05=is_pi05,
    )
    return dst


# ── Visualisation ─────────────────────────────────────────────────────────────

_IMG_SIZE = 224


def _attn_to_hmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    if camera == "exterior":
        patches = attn_512[:NUM_IMAGE_TOKENS]
    else:
        patches = attn_512[NUM_IMAGE_TOKENS : 2 * NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return cv2.resize(grid, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def _overlay(img: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img_s = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    hn = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    color = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_s, 1 - alpha, color, alpha, 0)


def _diff_overlay(img: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """diff is already _IMG_SIZE×_IMG_SIZE. Red=more, Blue=less attention vs baseline."""
    img_s = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    vmax = max(abs(diff.max()), abs(diff.min()), 1e-8)
    d_norm = np.clip((diff / vmax + 1.0) / 2.0, 0.0, 1.0)
    cmap = plt.get_cmap("RdBu_r")
    color = (cmap(d_norm)[..., :3] * 255).astype(np.uint8)
    return cv2.addWeighted(img_s, 0.55, color, 0.45, 0)


def _resolve_attn_vec(t2i: np.ndarray, agg_fn, attn_mode: str, tok_idx: int) -> np.ndarray:
    """Reduce (8, n_text, 512) → (512,) using head agg + token mode."""
    head_agg = agg_fn(t2i) if agg_fn is not None else t2i.mean(axis=0)
    if attn_mode == "whole":
        return head_agg.mean(axis=0)
    else:
        return head_agg[tok_idx]


def _build_row_png(
    items: list[tuple[str, np.ndarray, bool]],   # (label, img_224, is_baseline)
) -> bytes:
    """Render a single-row figure: one cell per prompt."""
    n = len(items)
    cell_w, cell_h = 2.6, 3.0
    fig, axes = plt.subplots(1, n, figsize=(n * cell_w, cell_h), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    if n == 1:
        axes = [axes]
    for ax, (label, img, is_base) in zip(axes, items):
        ax.imshow(img, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#00bfff" if is_base else "#333")
            sp.set_linewidth(2 if is_base else 0.5)
        short = ("★ " if is_base else "") + label[:26] + ("…" if len(label) > 26 else "")
        ax.set_title(short, color="#00bfff" if is_base else "white", fontsize=7, pad=3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_delta_png(
    ext_deltas: list[tuple[str, np.ndarray]],    # (label, diff_img_224) per non-baseline prompt
    wrist_deltas: list[tuple[str, np.ndarray]],
) -> bytes:
    """Render 2-row delta figure: row 0 = exterior Δ, row 1 = wrist Δ."""
    n = len(ext_deltas)
    if n == 0:
        return b""
    cell_w, cell_h = 2.6, 3.0
    fig, axes = plt.subplots(2, n, figsize=(n * cell_w, 2 * cell_h), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    if n == 1:
        axes = axes.reshape(2, 1)
    row_labels = ["Ext Δ", "Wrist Δ"]
    for row, (deltas, rl) in enumerate(zip([ext_deltas, wrist_deltas], row_labels)):
        for col, (label, img) in enumerate(deltas):
            ax = axes[row, col]
            ax.imshow(img, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                short = label[:26] + ("…" if len(label) > 26 else "")
                ax.set_title(short, color="white", fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(rl, color="#aaa", fontsize=8, rotation=0, labelpad=32, va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_cf_panels(
    results: list[dict],
    baseline_idx: int,
    layer: int,
    agg_fn,
    tok_idx: int,
    attn_mode: str,
) -> tuple[bytes, bytes, bytes]:
    """Returns (ext_png, wrist_png, delta_png) — three separate section images."""
    _ph = np.full((_IMG_SIZE, _IMG_SIZE, 3), 30, dtype=np.uint8)

    def get_vec(t2i):
        return _resolve_attn_vec(t2i, agg_fn, attn_mode, tok_idx)

    # Baseline hmaps for delta
    base_t2i = results[baseline_idx]["slice_dict"].get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")
    base_ext_h = base_wrist_h = None
    if base_t2i is not None:
        vec = get_vec(base_t2i)
        base_ext_h = _attn_to_hmap(vec, "exterior")
        base_wrist_h = _attn_to_hmap(vec, "wrist")

    ext_items: list[tuple[str, np.ndarray, bool]] = []
    wrist_items: list[tuple[str, np.ndarray, bool]] = []
    ext_deltas: list[tuple[str, np.ndarray]] = []
    wrist_deltas: list[tuple[str, np.ndarray]] = []

    for i, res in enumerate(results):
        t2i = res["slice_dict"].get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")
        ext_img_raw = res["slice_dict"].get("images", {}).get("exterior")
        ext_img = _ph if ext_img_raw is None else ext_img_raw
        wrist_img_raw = res["slice_dict"].get("images", {}).get("wrist")
        wrist_img = _ph if wrist_img_raw is None else wrist_img_raw
        label = res["prompt"]
        is_base = i == baseline_idx

        if t2i is not None:
            vec = get_vec(t2i)
            ext_h = _attn_to_hmap(vec, "exterior")
            wrist_h = _attn_to_hmap(vec, "wrist")
            ext_items.append((label, _overlay(ext_img, ext_h), is_base))
            wrist_items.append((label, _overlay(wrist_img, wrist_h), is_base))
            if not is_base and base_ext_h is not None:
                ext_deltas.append((label, _diff_overlay(ext_img, ext_h - base_ext_h)))
                wrist_deltas.append((label, _diff_overlay(wrist_img, wrist_h - base_wrist_h)))
        else:
            ext_items.append((label, _ph, is_base))
            wrist_items.append((label, _ph, is_base))

    return (
        _build_row_png(ext_items),
        _build_row_png(wrist_items),
        _build_delta_png(ext_deltas, wrist_deltas),
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render(
    attn_h5_root: str,
    checkpoints: list[str],
    default_checkpoint: str,
) -> None:
    """Render the Counterfactual Prompting tab."""

    st.markdown("### Counterfactual Prompting")
    st.caption(
        "Run inference on the **same image** with different prompts. "
        "Compare how text changes the attention map. Results are saved to HDF5 per prompt."
    )

    cat = _catalogue()
    if not cat:
        st.error("No local datasets found in `data/example/`.")
        return

    # ── Step 1: Image source ──────────────────────────────────────────────────
    with st.expander("① Image source & frame", expanded=True):
        dataset_name = st.selectbox("Dataset", list(cat.keys()), key="cf_dataset")
        dataset = cat[dataset_name]

        max_frame = dataset["n_frames"] - 1
        frame_idx = st.slider("Frame", 0, max_frame, 0, key="cf_frame")
        cf_camera = st.radio("Ext camera", ["right", "left"], horizontal=True, key="cf_camera")

        col_ext, col_wrist = st.columns(2)
        try:
            from PIL import Image
            ext_img, wrist_img, joint_pos, gripper_pos = _load_frame(dataset, frame_idx, camera=cf_camera)
            col_ext.image(ext_img, caption=f"Exterior ({cf_camera})", use_container_width=True)
            col_wrist.image(wrist_img, caption="Wrist", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load frame: {e}")
            return

    # ── Step 2: Prompts ───────────────────────────────────────────────────────
    with st.expander("② Define prompts", expanded=True):
        st.caption("First prompt = baseline (marked ★). Add counterfactuals below.")

        if "cf_prompts" not in st.session_state:
            st.session_state["cf_prompts"] = [
                dataset["default_prompt"],
                "pick up the banana",
                "pick up the duck",
            ]

        prompts: list[str] = st.session_state["cf_prompts"]

        updated = []
        for i, p in enumerate(prompts):
            col_inp, col_del = st.columns([9, 1])
            with col_inp:
                val = st.text_input(
                    f"{'★ Baseline' if i == 0 else f'Prompt {i}'}",
                    value=p,
                    key=f"cf_p_{i}",
                )
                updated.append(val)
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)
                if i > 0 and st.button("✕", key=f"cf_del_{i}"):
                    prompts.pop(i)
                    st.session_state["cf_prompts"] = prompts
                    st.rerun()

        st.session_state["cf_prompts"] = updated

        if st.button("＋ Add prompt", key="cf_add"):
            st.session_state["cf_prompts"].append("")
            st.rerun()

    # ── Step 3: Inference settings ────────────────────────────────────────────
    with st.expander("③ Inference settings"):
        _ckpt_root = os.path.join(_PROJECT_ROOT, "checkpoints/viz")
        _available_ckpts = sorted(
            d for d in os.listdir(_ckpt_root)
            if os.path.isdir(os.path.join(_ckpt_root, d))
        ) if os.path.isdir(_ckpt_root) else ["pi05_droid_pytorch"]
        _default_idx = (
            _available_ckpts.index("pi05_droid_pytorch")
            if "pi05_droid_pytorch" in _available_ckpts else 0
        )
        checkpoint_id = st.selectbox(
            "Checkpoint", _available_ckpts, index=_default_idx, key="cf_checkpoint"
        )
        episode_id = st.text_input(
            "Episode ID (folder name)",
            value=f"{dataset_name}_f{frame_idx}",
            key="cf_episode_id",
        )
        def _cf_gpu_devices() -> list[str]:
            try:
                import pynvml
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return [f"cuda:{i}" for i in range(n)] + ["cpu"]
            except Exception:
                return ["cuda:0", "cpu"]

        gpu_device = st.selectbox("GPU", _cf_gpu_devices(), key="cf_gpu")
        gpu_id = int(gpu_device.split(":")[-1]) if "cuda" in gpu_device else 0
        save_to_disk = st.checkbox("Save results to HDF5", value=True, key="cf_save")

    # ── Step 4: Run ───────────────────────────────────────────────────────────
    valid_prompts = [p.strip() for p in st.session_state["cf_prompts"] if p.strip()]
    if not valid_prompts:
        st.warning("Add at least one prompt.")
        return

    run_btn = st.button("▶ Run Counterfactual Inference", type="primary", key="cf_run")

    if run_btn:
        # Load model
        from viz.dashboard.inference import load_model  # noqa: PLC0415
        with st.spinner("Loading model…"):
            try:
                ckpt_path = os.path.join(_PROJECT_ROOT, "checkpoints/viz", checkpoint_id)
                policy = load_model(ckpt_path, device=gpu_device)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                return

        results = []
        progress = st.progress(0, text="Running inference…")
        for i, prompt in enumerate(valid_prompts):
            progress.progress((i) / len(valid_prompts), text=f"Prompt {i+1}/{len(valid_prompts)}: '{prompt[:40]}'")
            try:
                slice_dict = _run_one_prompt(
                    policy, ext_img, wrist_img, joint_pos, gripper_pos,
                    prompt, gpu_id=gpu_id,
                )
                slug = _loader.prompt_to_slug(prompt)
                if save_to_disk:
                    saved_path = _save_cf_h5(
                        slice_dict, checkpoint_id, episode_id, frame_idx, slug, attn_h5_root
                    )
                    st.toast(f"Saved: {os.path.basename(saved_path)}", icon="💾")
                results.append({"prompt": prompt, "slug": slug, "slice_dict": slice_dict})
            except Exception as e:
                st.error(f"Failed on prompt '{prompt}': {e}")

        progress.progress(1.0, text="Done.")
        st.session_state["cf_results"] = results
        st.session_state["cf_frame_idx"] = frame_idx

    # ── Step 5: Visualisation ─────────────────────────────────────────────────
    results: list[dict] = st.session_state.get("cf_results", [])
    if not results:
        st.info("Configure prompts above and click **▶ Run** to begin.")
        return

    st.markdown("---")
    st.markdown("### Results")

    # Check available layers
    first_prefix = results[0]["slice_dict"].get("prefix", {})
    avail_layers = sorted(int(k.split("_")[1]) for k in first_prefix if k.startswith("layer_"))
    if not avail_layers:
        st.warning("No attention data found in results.")
        return

    # Get token metadata from first result
    meta0 = results[0]["slice_dict"].get("meta", {})
    n_real = meta0.get("n_real_tokens", len(meta0.get("token_texts", [])) or 50)
    real_texts = meta0.get("token_texts", [])[:n_real]

    # ── Row 1: layer / attn-mode ──────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([2, 2])
    with ctrl1:
        layer = st.select_slider("Layer", options=avail_layers,
                                 value=avail_layers[min(2, len(avail_layers)-1)],
                                 key="cf_vis_layer")
    with ctrl2:
        attn_mode_label = st.radio(
            "Attention mode",
            ["Per token", "Whole text→image"],
            horizontal=True,
            key="cf_attn_mode",
        )
        attn_mode = "token" if attn_mode_label == "Per token" else "whole"

    # ── Row 2: head aggregation ────────────────────────────────────────────────
    # Exclude "All heads" — CF comparison needs a single map per row
    agg_opts = [a for a in _ALL_AGG_NAMES if a != "All heads"]
    agg = st.radio("Head aggregation", agg_opts, index=0, horizontal=True, key="cf_vis_agg")
    topk_k = count_pct = None
    if agg == "Top-K Focused":
        topk_k = st.slider("K (focused heads)", 1, 8, 4, key="cf_topk")
    elif agg == "Count Above Threshold":
        count_pct = st.slider("Top % threshold", 1, 50, 10, key="cf_pct")
    st.caption(_AGG_DESCRIPTIONS[agg])

    # Resolve agg_fn
    if agg in _SIMPLE_AGGS:
        agg_fn = _SIMPLE_AGGS[agg]  # may be None → _resolve_attn_vec falls back to mean
    elif agg == "Top-K Focused":
        agg_fn = _make_topk_fn(topk_k)
    else:
        agg_fn = _make_count_fn(count_pct)

    # ── Token selector (per-token mode only) ──────────────────────────────────
    tok_idx = 0
    tok_label = "all tokens"
    if attn_mode == "token":
        if real_texts:
            raw_labels = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(real_texts)]
            seen: dict[str, int] = {}
            token_labels: list[str] = []
            for i, lbl in enumerate(raw_labels):
                if raw_labels.count(lbl) > 1:
                    seen[lbl] = seen.get(lbl, 0) + 1
                    token_labels.append(f"{lbl}#{seen[lbl]}")
                else:
                    token_labels.append(lbl)

            default_tok = token_labels[min(3, n_real - 1)]
            has_dupes = any(raw_labels.count(r) > 1 for r in raw_labels)
            pill_label = (
                "Click a token to visualize its attention: (duplicate tokens are suffixed #1, #2, …)"
                if has_dupes
                else "Click a token to visualize its attention:"
            )
            selected_label = st.pills(
                pill_label, options=token_labels, default=default_tok,
                selection_mode="single", key="cf_tok_pill",
            )
            if selected_label is None:
                st.info("Click a token above to visualize its attention.")
                return
            tok_idx = token_labels.index(selected_label)
            tok_label = selected_label
        else:
            st.warning("No token texts available.")
            return

    # ── Render three subsections ──────────────────────────────────────────────
    caption_suffix = f"layer {layer} — {attn_mode_label} '{tok_label}' — {agg}"
    with st.spinner("Rendering…"):
        ext_png, wrist_png, delta_png = _build_cf_panels(
            results, baseline_idx=0,
            layer=layer, agg_fn=agg_fn,
            tok_idx=tok_idx, attn_mode=attn_mode,
        )

    st.markdown("#### Exterior Camera")
    st.image(ext_png, caption=f"Exterior — {caption_suffix}", use_container_width=True)

    st.markdown("#### Wrist Camera")
    st.image(wrist_png, caption=f"Wrist — {caption_suffix}", use_container_width=True)

    if delta_png:
        st.markdown("#### Δ vs Baseline")
        st.image(delta_png, caption=f"Δ (red=more, blue=less) — {caption_suffix}", use_container_width=True)

    # Saved slugs reference
    if save_to_disk:
        with st.expander("Saved HDF5 files"):
            for res in results:
                path = _loader.h5_path_cf(checkpoint_id, episode_id, frame_idx,
                                          res["slug"], attn_h5_root)
                exists = os.path.exists(path)
                st.code(f"{'✓' if exists else '?'} {path}")
