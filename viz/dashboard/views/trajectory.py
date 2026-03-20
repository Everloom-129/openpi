"""Trajectory view: multi-frame attention grid for one episode.

Layout:
  Controls  — camera, head aggregation, layer multiselect, frame stride
  Token pills — shared selector across all frames
  Frame strip — small thumbnails of selected frames
  Main grid  — rows = layers, cols = selected frames
  CF section — expander with per-slug grids + Δ vs main
"""
from __future__ import annotations

import io
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from viz.dashboard import loader_results as _rl
from viz.dashboard.loader import list_layers_in_h5, load_images, load_meta, load_text_to_img
from viz.dashboard.views.grid_heatmap import (
    _ALL_AGG_NAMES,
    _SIMPLE_AGGS,
    _attn_to_heatmap,
    _fig_to_bytes,
    _make_count_fn,
    _make_topk_fn,
    _overlay,
)

DEFAULT_LAYERS = [1, 4, 5, 7, 10]
THUMB_SIZE = 72   # px for thumbnail strip
MAX_FRAMES_WARN = 16  # warn if more frames selected
CHUNK_SIZE = 8   # frames per row in thumbnail strip and trajectory figures


# ── Figure builder ─────────────────────────────────────────────────────────────

def _build_trajectory_figure(
    frames: list[int],
    layers: list[int],
    t2i_data: dict[tuple[int, int], np.ndarray | None],  # (frame, layer) → (8, n_text, 512)
    imgs: dict[int, np.ndarray | None],                   # frame → (H, W, 3) uint8
    tok_idx: int,
    camera: str,
    agg_fn,
    row_label_prefix: str = "",
) -> plt.Figure:
    """Build a (layers × frames) matplotlib figure.

    rows = selected layers, cols = selected frames.
    agg_fn is applied to (8, 512) → (512,).
    """
    n_rows = len(layers)
    n_cols = len(frames)
    cell_w = 1.4
    cell_h = 1.4
    label_frac = 0.45 / (n_cols * cell_w + 0.45)  # fraction for row-label column

    fig = plt.figure(figsize=(n_cols * cell_w + 0.45, n_rows * cell_h + 0.3), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.04,
        hspace=0.18,
        left=label_frac,
        right=0.99,
        top=0.92,
        bottom=0.02,
    )

    for row_i, layer in enumerate(layers):
        for col_i, frame in enumerate(frames):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            t2i = t2i_data.get((frame, layer))
            img = imgs.get(frame)
            if img is None:
                img = np.full((224, 224, 3), 30, dtype=np.uint8)

            if t2i is None:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=7)
            else:
                attn_512 = agg_fn(t2i[:, tok_idx, :])  # (8,512) → (512,)
                hmap = _attn_to_heatmap(attn_512, camera)
                cell_img = _overlay(img, hmap)
                ax.imshow(cell_img, aspect="equal")

            # Column headers (top row only)
            if row_i == 0:
                ax.set_title(f"F{frame}", color="white", fontsize=8, pad=3)

            # Row labels (first column only)
            if col_i == 0:
                prefix = f"{row_label_prefix}\n" if row_label_prefix else ""
                ax.set_ylabel(
                    f"{prefix}L{layer}",
                    color="white", fontsize=8,
                    rotation=0, labelpad=30, va="center",
                )

    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render(
    root: str,
    outcome: str,
    date: str,
    episode: str,
    available_frames: list[int],
    cf_slugs: list[str],
) -> None:
    """Render the trajectory tab."""

    if not available_frames:
        st.warning("No frames found for this episode.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 3, 1])

    with ctrl_c1:
        camera = st.radio("Camera", ["Exterior", "Wrist"], horizontal=True, key="traj_cam")
        camera_key = "exterior" if camera == "Exterior" else "wrist"

        # Exclude "All heads" — trajectory needs a single map per cell
        agg_names = [n for n in _ALL_AGG_NAMES if n != "All heads"]
        agg = st.selectbox("Head aggregation", agg_names, index=agg_names.index("Max"), key="traj_agg")
        if agg == "Top-K Focused":
            topk_k = st.slider("K heads", 1, 8, 4, key="traj_topk")
        elif agg == "Count Above Threshold":
            count_pct = st.slider("Top %", 1, 50, 10, key="traj_pct")

    with ctrl_c2:
        first_h5 = _rl.h5_path_results(root, outcome, date, episode, available_frames[0])
        all_layers = list_layers_in_h5(first_h5)
        default_layers = [l for l in DEFAULT_LAYERS if l in all_layers] or all_layers[:4]
        selected_layers = st.multiselect(
            "Layers", options=all_layers, default=default_layers, key="traj_layers"
        )

    with ctrl_c3:
        stride = st.select_slider(
            "Stride", options=[1, 2, 4, 8, 16], value=8, key="traj_stride"
        )

    if not selected_layers:
        st.info("Select at least one layer.")
        return

    # ── Frame selection ───────────────────────────────────────────────────────
    default_sel = available_frames[::stride] or available_frames[:1]

    # Quick-select buttons write directly to the multiselect's session state key.
    # This is the only reliable way to programmatically update a multiselect in Streamlit,
    # since `default` is ignored once the widget key already exists in session state.
    ms_key = f"traj_frames_{stride}"
    btn_c1, btn_c2, btn_c3, _ = st.columns([1, 1, 2, 8])
    with btn_c1:
        if st.button("All", key="traj_btn_all"):
            st.session_state[ms_key] = available_frames
    with btn_c2:
        if st.button("None", key="traj_btn_none"):
            st.session_state[ms_key] = []
    with btn_c3:
        if st.button(f"Every {stride}", key="traj_btn_stride"):
            st.session_state[ms_key] = available_frames[::stride]

    selected_frames: list[int] = st.multiselect(
        "Select frames to compare:",
        options=available_frames,
        default=default_sel,
        key=ms_key,
    )

    if not selected_frames:
        st.info("Select at least one frame.")
        return

    selected_frames = sorted(selected_frames)

    if len(selected_frames) > MAX_FRAMES_WARN:
        st.warning(f"{len(selected_frames)} frames selected — rendering may be slow. Consider reducing.")

    # ── Token selector ────────────────────────────────────────────────────────
    meta = load_meta(_rl.h5_path_results(root, outcome, date, episode, selected_frames[0]))
    n_real: int = meta.get("n_real_tokens", len(meta.get("token_texts", [])))
    real_texts = meta.get("token_texts", [])[:n_real]

    if not real_texts:
        st.warning("No token texts found.")
        return

    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    # Deduplicate labels (same logic as grid_heatmap)
    raw_labels = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(real_texts)]
    seen: dict[str, int] = {}
    token_labels: list[str] = []
    for lbl in raw_labels:
        if raw_labels.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            token_labels.append(f"{lbl}#{seen[lbl]}")
        else:
            token_labels.append(lbl)

    selected_label = st.pills(
        "Token (shared across all frames):",
        options=token_labels,
        default=token_labels[min(3, n_real - 1)],
        selection_mode="single",
        key="traj_tok",
    )
    if selected_label is None:
        st.info("Click a token to visualize its attention.")
        return
    tok_idx = token_labels.index(selected_label)

    # ── Frame thumbnail strip ─────────────────────────────────────────────────
    with st.expander("Frame thumbnails", expanded=True):
        for chunk_start in range(0, len(selected_frames), CHUNK_SIZE):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            thumb_cols = st.columns(len(chunk))
            for i, frame in enumerate(chunk):
                path = _rl.h5_path_results(root, outcome, date, episode, frame)
                imgs = load_images(path)
                thumb = imgs.get(camera_key)
                if thumb is not None:
                    thumb_small = cv2.resize(thumb, (THUMB_SIZE, THUMB_SIZE))
                else:
                    thumb_small = np.full((THUMB_SIZE, THUMB_SIZE, 3), 40, dtype=np.uint8)
                with thumb_cols[i]:
                    st.image(thumb_small, caption=f"F{frame}", use_container_width=True)

    # ── Resolve aggregation function ──────────────────────────────────────────
    if agg in _SIMPLE_AGGS:
        agg_fn = _SIMPLE_AGGS[agg]
    elif agg == "Top-K Focused":
        agg_fn = _make_topk_fn(topk_k)
    else:
        agg_fn = _make_count_fn(count_pct)
    if agg_fn is None:  # "All heads" was filtered out, but guard anyway
        agg_fn = lambda x: x.mean(axis=0)

    # ── Load attention data ───────────────────────────────────────────────────
    with st.spinner(f"Loading {len(selected_frames)} × {len(selected_layers)} attention slices…"):
        t2i_data: dict[tuple[int, int], np.ndarray | None] = {}
        imgs_by_frame: dict[int, np.ndarray | None] = {}
        for frame in selected_frames:
            path = _rl.h5_path_results(root, outcome, date, episode, frame)
            imgs_by_frame[frame] = load_images(path).get(camera_key)
            for layer in selected_layers:
                t2i_data[(frame, layer)] = load_text_to_img(path, layer)

    # ── Main trajectory figure ────────────────────────────────────────────────
    with st.spinner("Rendering trajectory grid…"):
        for chunk_idx, chunk_start in enumerate(range(0, len(selected_frames), CHUNK_SIZE)):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            fig = _build_trajectory_figure(
                frames=chunk,
                layers=selected_layers,
                t2i_data=t2i_data,
                imgs=imgs_by_frame,
                tok_idx=tok_idx,
                camera=camera_key,
                agg_fn=agg_fn,
            )
            caption = f'Trajectory — "{selected_label}" — {camera} — {agg}' if chunk_idx == 0 else None
            st.image(_fig_to_bytes(fig), caption=caption, use_container_width=True)

    # ── Counterfactual section ────────────────────────────────────────────────
    if not cf_slugs:
        return

    with st.expander(f"Counterfactual comparison ({', '.join(cf_slugs)})", expanded=False):
        st.caption(
            "Same token & frames, different prompt. "
            "Pick a layer, then see each variant's attention and Δ vs main."
        )

        cf_layer = st.selectbox(
            "Layer for CF comparison",
            options=selected_layers,
            index=min(1, len(selected_layers) - 1),
            key="traj_cf_layer",
        )

        # Load main attention for the CF layer (reuse already-loaded if available)
        main_t2i_cf: dict[tuple[int, int], np.ndarray | None] = {
            (f, cf_layer): t2i_data.get((f, cf_layer))
            for f in selected_frames
        }

        for slug in cf_slugs:
            st.markdown(f"**`{slug}`**")
            cf_t2i: dict[tuple[int, int], np.ndarray | None] = {}
            cf_imgs: dict[int, np.ndarray | None] = {}
            for frame in selected_frames:
                cf_path = _rl.h5_path_results(root, outcome, date, episode, frame, slug=slug)
                if os.path.exists(cf_path):
                    cf_imgs[frame] = load_images(cf_path).get(camera_key)
                    cf_t2i[(frame, cf_layer)] = load_text_to_img(cf_path, cf_layer)
                else:
                    # File missing for this frame — use blank
                    cf_imgs[frame] = imgs_by_frame.get(frame)
                    cf_t2i[(frame, cf_layer)] = None

            for chunk_start in range(0, len(selected_frames), CHUNK_SIZE):
                chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
                fig_cf = _build_trajectory_figure(
                    frames=chunk,
                    layers=[cf_layer],
                    t2i_data=cf_t2i,
                    imgs=cf_imgs,
                    tok_idx=tok_idx,
                    camera=camera_key,
                    agg_fn=agg_fn,
                    row_label_prefix=slug,
                )
                st.image(_fig_to_bytes(fig_cf), use_container_width=True)

        # Δ (main − first slug) heatmap
        st.markdown(f"**Δ main − `{cf_slugs[0]}`**")
        delta_t2i: dict[tuple[int, int], np.ndarray | None] = {}
        for frame in selected_frames:
            main_arr = main_t2i_cf.get((frame, cf_layer))
            cf0_path = _rl.h5_path_results(root, outcome, date, episode, frame, slug=cf_slugs[0])
            cf0_arr = load_text_to_img(cf0_path, cf_layer) if os.path.exists(cf0_path) else None
            if main_arr is not None and cf0_arr is not None:
                # Align n_text dim (may differ if prompts differ in length)
                n = min(main_arr.shape[1], cf0_arr.shape[1])
                delta_t2i[(frame, cf_layer)] = main_arr[:, :n, :] - cf0_arr[:, :n, :]
            else:
                delta_t2i[(frame, cf_layer)] = None

        # For delta, clamp tok_idx to available tokens
        delta_tok = min(tok_idx, min(
            (arr.shape[1] - 1 for arr in delta_t2i.values() if arr is not None),
            default=tok_idx,
        ))

        for chunk_idx, chunk_start in enumerate(range(0, len(selected_frames), CHUNK_SIZE)):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            fig_delta = _build_trajectory_figure(
                frames=chunk,
                layers=[cf_layer],
                t2i_data=delta_t2i,
                imgs=imgs_by_frame,
                tok_idx=delta_tok,
                camera=camera_key,
                agg_fn=agg_fn,
                row_label_prefix="Δ",
            )
            caption = "Red = more attention in main, Blue = less" if chunk_idx == 0 else None
            st.image(_fig_to_bytes(fig_delta), caption=caption, use_container_width=True)
