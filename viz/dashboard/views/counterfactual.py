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
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st

from viz.dashboard import loader as _loader

NUM_IMAGE_TOKENS = 256
PATCH_GRID = 16
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


# ── Dataset catalogue ─────────────────────────────────────────────────────────

def _catalogue() -> dict[str, dict]:
    """Built-in image datasets available for CF experiments."""
    data_root = os.path.join(_PROJECT_ROOT, "data/visualization")
    cat: dict[str, dict] = {}

    # Duck dataset
    duck_dir = os.path.join(data_root, "duck/frames")
    if os.path.isdir(duck_dir):
        n = len(list(Path(duck_dir + "/varied_camera_1").glob("*.jpg")))
        cat["duck"] = {
            "ext_dir": duck_dir + "/varied_camera_1",
            "wrist_dir": duck_dir + "/hand_camera",
            "traj_h5": os.path.join(data_root, "duck/trajectory.h5"),
            "default_prompt": "place the duck toy into the pink bowl",
            "n_frames": n,
        }

    # Pineapple dataset
    pine_dir = os.path.join(data_root, "aawr_pineapple/recordings/frames")
    if os.path.isdir(pine_dir):
        n = len(list(Path(pine_dir + "/varied_camera_2").glob("*.jpg")))
        cat["pineapple"] = {
            "ext_dir": pine_dir + "/varied_camera_2",
            "wrist_dir": pine_dir + "/hand_camera",
            "traj_h5": os.path.join(data_root, "aawr_pineapple/trajectory.h5"),
            "default_prompt": "find the pineapple toy and pick it up",
            "n_frames": n,
        }

    return cat


def _load_frame(dataset: dict, frame_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ext_img, wrist_img, joint_pos, gripper_pos)."""
    from PIL import Image
    import h5py

    ext_path = os.path.join(dataset["ext_dir"], f"{frame_idx:05d}.jpg")
    wrist_path = os.path.join(dataset["wrist_dir"], f"{frame_idx:05d}.jpg")
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
    """Run inference for one prompt, return in-memory slice dict."""
    # The model writes attn maps to attn/{gpu_id}/layers_prefix/
    # We read them back immediately and return as dict (no long-term disk use).
    example = {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": prompt,
    }
    _ = policy.infer(example)

    layers_dir = os.path.join(_PROJECT_ROOT, f"attn/{gpu_id}/layers_prefix")
    return _loader.make_slice_dict_from_npy(
        layers_prefix_dir=layers_dir,
        instruction=prompt,
        ext_img=ext_img,
        wrist_img=wrist_img,
    )


def _save_cf_h5(
    slice_dict: dict,
    checkpoint_id: str,
    episode_id: str,
    frame_idx: int,
    prompt_slug: str,
    attn_h5_root: str,
) -> str:
    """Persist one counterfactual rollout to disk as HDF5."""
    # Import lazily — only needed when saving
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
    from convert_npy_to_h5 import convert_episode  # noqa: PLC0415

    # Write .npy files to a temp dir so convert_episode can read them
    with tempfile.TemporaryDirectory() as tmp:
        prefix_dir = os.path.join(tmp, "layers_prefix")
        os.makedirs(prefix_dir)

        for layer_key, layer_data in slice_dict.get("prefix", {}).items():
            layer_idx = int(layer_key.split("_")[1])
            t2i = layer_data.get("text_to_img")
            full = layer_data.get("full")
            # Reconstruct full (seq, seq) from text_to_img + zeros for completeness
            # For CF, we only need text_to_img, so build a minimal full attn tensor
            if full is not None:
                arr = full  # (8, seq, seq)
            else:
                # Build a padded placeholder so convert_episode gets seq_len right
                seq = slice_dict["meta"]["seq_len"]
                arr = np.zeros((1, 8, seq, seq), dtype=np.float32)
                # Splice text_to_img back in
                from viz.dashboard.loader import TEXT_START_IDX, TOTAL_IMAGE_TOKENS
                n_text = t2i.shape[1]
                arr[0, :, TEXT_START_IDX : TEXT_START_IDX + n_text, :TOTAL_IMAGE_TOKENS] = t2i
            # Add batch dim if missing
            if arr.ndim == 3:
                arr = arr[np.newaxis]
            npy_path = os.path.join(prefix_dir, f"attn_map_layer_{layer_idx}.npy")
            np.save(npy_path, arr.astype(np.float32))

        dst = _loader.h5_path_cf(checkpoint_id, episode_id, frame_idx, prompt_slug, attn_h5_root)
        images = slice_dict.get("images", {})
        instruction = slice_dict.get("meta", {}).get("instruction", "")

        # Save images temporarily
        ext_tmp = wrist_tmp = None
        from PIL import Image as PILImage
        _ext = images.get("exterior")
        _wrist = images.get("wrist")
        if _ext is not None:
            ext_tmp = os.path.join(tmp, "ext.jpg")
            PILImage.fromarray(_ext).save(ext_tmp)
        if _wrist is not None:
            wrist_tmp = os.path.join(tmp, "wrist.jpg")
            PILImage.fromarray(_wrist).save(wrist_tmp)

        convert_episode(
            src_prefix_dir=prefix_dir,
            dst_h5_path=dst,
            ext_img_path=ext_tmp,
            wrist_img_path=wrist_tmp,
            instruction=instruction,
            frame_idx=frame_idx,
        )

    return dst


# ── Visualisation ─────────────────────────────────────────────────────────────

def _attn_to_hmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    if camera == "exterior":
        patches = attn_512[:NUM_IMAGE_TOKENS]
    else:
        patches = attn_512[NUM_IMAGE_TOKENS : 2 * NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return cv2.resize(grid, (112, 112), interpolation=cv2.INTER_LINEAR)


def _overlay(img: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img_s = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    hn = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    color = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_s, 1 - alpha, color, alpha, 0)


def _diff_overlay(img: np.ndarray, diff_16x16: np.ndarray) -> np.ndarray:
    """Red=more attention, Blue=less attention vs baseline."""
    img_s = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    d = cv2.resize(diff_16x16.astype(np.float32), (112, 112), interpolation=cv2.INTER_LINEAR)
    vmax = max(abs(d.max()), abs(d.min()), 1e-8)
    d_norm = (d / vmax + 1.0) / 2.0   # 0..1, 0.5 = no change
    color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_RdBu)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_s, 0.55, color, 0.45, 0)


def _build_cf_grid(
    results: list[dict],       # [{prompt, slug, slice_dict}]
    baseline_idx: int,
    layer: int,
    head_agg: str,             # "max" | "mean" | int
    camera: str,
    tok_idx: int,
    show_diff: bool,
) -> bytes:
    """Render rows=prompts × cols=(raw | diff) matplotlib grid → PNG bytes."""
    n_prompts = len(results)
    n_cols = 2 if show_diff else 1   # col 0 = attention, col 1 = diff vs baseline
    cell = 1.5
    fig = plt.figure(figsize=(n_cols * cell * 2, n_prompts * cell), dpi=110)
    fig.patch.set_facecolor("#0e1117")

    gs = gridspec.GridSpec(n_prompts, n_cols * 2, figure=fig,
                           wspace=0.05, hspace=0.12,
                           left=0.12, right=1.0, top=0.95, bottom=0.02)

    # Precompute baseline heatmap for diff
    base_hmap = None
    if show_diff:
        base_t2i = results[baseline_idx]["slice_dict"].get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")
        if base_t2i is not None:
            if head_agg == "max":
                base_agg = base_t2i.max(axis=0)
            elif head_agg == "mean":
                base_agg = base_t2i.mean(axis=0)
            else:
                base_agg = base_t2i[int(head_agg)]
            base_hmap = _attn_to_hmap(base_agg[tok_idx], camera)

    for row_i, res in enumerate(results):
        t2i = res["slice_dict"].get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")
        cam_img = res["slice_dict"].get("images", {}).get(camera)
        if cam_img is None:
            cam_img = np.full((224, 224, 3), 30, dtype=np.uint8)

        label = res["prompt"][:28] + ("…" if len(res["prompt"]) > 28 else "")
        is_base = row_i == baseline_idx

        # Col 0+1: attention overlay (spans 2 sub-cols)
        ax0 = fig.add_subplot(gs[row_i, 0:2])
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor("#00bfff" if is_base else "#444")
            sp.set_linewidth(2 if is_base else 0.5)

        if t2i is not None:
            if head_agg == "max":
                agg = t2i.max(axis=0)
            elif head_agg == "mean":
                agg = t2i.mean(axis=0)
            else:
                agg = t2i[int(head_agg)]
            hmap = _attn_to_hmap(agg[tok_idx], camera)
            cell_img = _overlay(cam_img, hmap)
            ax0.imshow(cell_img, aspect="auto")
        else:
            ax0.set_facecolor("#1a1a2e")
            ax0.text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=ax0.transAxes, color="gray", fontsize=8)

        ax0.set_ylabel(
            ("★ " if is_base else "") + label,
            color="#00bfff" if is_base else "white",
            fontsize=7, rotation=0, labelpad=4, va="center", ha="right",
        )

        # Col 2+3: difference map
        if show_diff:
            ax1 = fig.add_subplot(gs[row_i, 2:4])
            ax1.set_xticks([]); ax1.set_yticks([])
            for sp in ax1.spines.values():
                sp.set_visible(False)

            if is_base or base_hmap is None or t2i is None:
                ax1.set_facecolor("#1a1a2e")
                ax1.text(0.5, 0.5, "baseline" if is_base else "N/A",
                         ha="center", va="center", transform=ax1.transAxes,
                         color="#555", fontsize=8)
            else:
                diff = hmap - base_hmap
                diff_img = _diff_overlay(cam_img, diff)
                ax1.imshow(diff_img, aspect="auto")

            if row_i == 0:
                ax0.set_title("Attention", color="white", fontsize=8, pad=3)
                ax1.set_title("Δ vs baseline", color="white", fontsize=8, pad=3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
        st.error("No local datasets found in `data/visualization/`.")
        return

    # ── Step 1: Image source ──────────────────────────────────────────────────
    with st.expander("① Image source & frame", expanded=True):
        dataset_name = st.selectbox("Dataset", list(cat.keys()), key="cf_dataset")
        dataset = cat[dataset_name]

        max_frame = dataset["n_frames"] - 1
        frame_idx = st.slider("Frame", 0, max_frame, 0, key="cf_frame")

        col_ext, col_wrist = st.columns(2)
        try:
            from PIL import Image
            ext_img, wrist_img, joint_pos, gripper_pos = _load_frame(dataset, frame_idx)
            col_ext.image(ext_img, caption="Exterior", use_container_width=True)
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
        checkpoint_id = st.selectbox("Checkpoint", checkpoints or ["0"], key="cf_ckpt")
        episode_id = st.text_input(
            "Episode ID (folder name)",
            value=f"{dataset_name}_f{frame_idx}",
            key="cf_episode_id",
        )
        gpu_device = st.selectbox("GPU", ["cuda:0", "cuda:1", "cpu"], key="cf_gpu")
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

    # Get n_real_tokens from first result
    meta0 = results[0]["slice_dict"].get("meta", {})
    n_real = meta0.get("n_real_tokens", len(meta0.get("token_texts", [])) or 50)
    real_texts = meta0.get("token_texts", [])[:n_real]

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        layer = st.select_slider("Layer", options=avail_layers,
                                 value=avail_layers[min(2, len(avail_layers)-1)],
                                 key="cf_vis_layer")
    with ctrl2:
        head_opts = ["max", "mean"] + list(range(8))
        head_labels = ["Max", "Mean"] + [f"H{i}" for i in range(8)]
        hl = st.radio("Head", head_labels, index=0, horizontal=True, key="cf_vis_head")
        head_sel = head_opts[head_labels.index(hl)]
    with ctrl3:
        camera = st.radio("Camera", ["Exterior", "Wrist"], horizontal=True, key="cf_vis_cam")
        cam_key = "exterior" if camera == "Exterior" else "wrist"
    with ctrl4:
        show_diff = st.checkbox("Show Δ diff column", value=True, key="cf_show_diff")

    # Token selector
    if real_texts:
        tok_idx = st.slider("Token", 0, n_real - 1, min(3, n_real-1), key="cf_tok")
        tok_label = real_texts[tok_idx].replace("▁", " ").strip() or f"[{tok_idx}]"
        st.caption(f"Token: **{tok_label}** (idx {tok_idx})")
    else:
        tok_idx = 0
        tok_label = "?"

    # Render grid
    with st.spinner("Rendering…"):
        png = _build_cf_grid(
            results, baseline_idx=0,
            layer=layer, head_agg=head_sel, camera=cam_key,
            tok_idx=tok_idx, show_diff=show_diff,
        )
    st.image(png, caption=f"CF comparison — token '{tok_label}' — layer {layer} — {camera}",
             use_container_width=True)

    # Saved slugs reference
    if save_to_disk:
        with st.expander("Saved HDF5 files"):
            for res in results:
                path = _loader.h5_path_cf(checkpoint_id, episode_id, frame_idx,
                                          res["slug"], attn_h5_root)
                exists = os.path.exists(path)
                st.code(f"{'✓' if exists else '?'} {path}")
